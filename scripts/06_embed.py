#!/usr/bin/env python3
"""Phase B: BioCLIP 2 image embeddings for the POC dataset.

Every row of data/dataset.csv (sorted by specimen_code, so the order is
deterministic) is embedded once with the frozen BioCLIP 2 image tower
(open_clip, hf-hub:imageomics/bioclip-2, ViT-L/14, 768-d) and L2-normalised.

Outputs, always written together and in identical row order:

- data/embeddings.npy        float32 array, N x 768; row i belongs to row i
                             of the index
- data/embeddings_index.csv  specimen_code, genus, species, subfamily,
                             split, image_path

Resumable: if both outputs exist they are loaded, only specimen codes not yet
embedded are computed, and both files are rewritten atomically (tmp + rename)
in specimen_code order. Images that fail to open are logged and skipped;
they get NO row (no zero vector), so a later run retries them. Integrity is
checked after writing (npy rows == index rows) and the SHA256 of the index
is logged so downstream steps can pin the exact embedding set.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import load_config, quiet_logging, setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

EMB_PATH = DATA / "embeddings.npy"
INDEX_PATH = DATA / "embeddings_index.csv"
INDEX_COLS = ["specimen_code", "genus", "species", "subfamily", "split", "image_path"]
LOG_EVERY_BATCHES = 10

log = logging.getLogger("embed")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_existing(dataset: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Return (index, embeddings) already on disk, restricted to specimen codes
    still present in the dataset. Empty index / None when starting fresh."""
    empty = pd.DataFrame(columns=INDEX_COLS)
    if not (EMB_PATH.exists() and INDEX_PATH.exists()):
        if EMB_PATH.exists() != INDEX_PATH.exists():
            log.warning("only one of %s / %s exists; starting fresh",
                        EMB_PATH.name, INDEX_PATH.name)
        return empty, None
    index = pd.read_csv(INDEX_PATH)
    embs = np.load(EMB_PATH)
    if len(index) != len(embs) or list(index.columns) != INDEX_COLS:
        log.warning("existing outputs inconsistent (%d index rows, %d embedding rows); "
                    "starting fresh", len(index), len(embs))
        return empty, None
    keep = index["specimen_code"].isin(dataset["specimen_code"]).to_numpy()
    if (~keep).any():
        log.warning("%d embedded specimens are no longer in the dataset; dropped",
                    int((~keep).sum()))
    log.info("resuming: %d embeddings already on disk", int(keep.sum()))
    return index[keep].reset_index(drop=True), embs[keep]


def load_model(name: str):
    import open_clip  # slow import; keep it out of --help paths

    torch.set_num_threads(os.cpu_count() or 1)
    t0 = time.perf_counter()
    with quiet_logging():  # open_clip + huggingface_hub chatter
        model, _, preprocess = open_clip.create_model_and_transforms(name)
    model.eval()
    log.info("model %s loaded in %.1fs (%d torch threads)",
             name, time.perf_counter() - t0, torch.get_num_threads())
    return model, preprocess


def open_image(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception as exc:  # truncated file, not an image, missing…
        log.warning("cannot open %s: %s", path, exc)
        return None


def embed(model, preprocess, todo: pd.DataFrame, batch_size: int
          ) -> tuple[pd.DataFrame, np.ndarray, list[dict]]:
    """Embed every row of `todo`; return (index rows, L2-normalised
    embeddings, skipped rows). Skipped rows are absent from both outputs."""
    rows: list[dict] = []
    chunks: list[np.ndarray] = []
    skipped: list[dict] = []
    n_batches = (len(todo) + batch_size - 1) // batch_size
    t_start = time.perf_counter()
    t_tick, n_tick = t_start, 0

    for b in range(n_batches):
        batch = todo.iloc[b * batch_size:(b + 1) * batch_size]
        tensors, kept = [], []
        for row in batch.itertuples(index=False):
            im = open_image(ROOT / row.image_path)
            if im is None:
                skipped.append({"specimen_code": row.specimen_code,
                                "image_path": row.image_path})
                continue
            tensors.append(preprocess(im))
            kept.append(row)
        if tensors:
            with torch.no_grad():
                feats = model.encode_image(torch.stack(tensors))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            chunks.append(feats.to(torch.float32).cpu().numpy())
            rows += [{c: getattr(r, c) for c in INDEX_COLS} for r in kept]
            n_tick += len(kept)

        if (b + 1) % LOG_EVERY_BATCHES == 0 or b + 1 == n_batches:
            now = time.perf_counter()
            done = min((b + 1) * batch_size, len(todo))
            rate = n_tick / max(now - t_tick, 1e-9)
            log.info("batch %d/%d  %d/%d images  %.2f img/s (recent)  %.0fs elapsed",
                     b + 1, n_batches, done, len(todo), rate, now - t_start)
            t_tick, n_tick = now, 0

    embs = (np.concatenate(chunks) if chunks
            else np.zeros((0, 0), dtype=np.float32))
    return pd.DataFrame(rows, columns=INDEX_COLS), embs, skipped


def write_atomic(index: pd.DataFrame, embs: np.ndarray) -> None:
    """Write both outputs via tmp files, then rename. Not a two-file
    transaction, but each file is either the old or the new version."""
    emb_tmp = EMB_PATH.with_suffix(".npy.tmp")
    idx_tmp = INDEX_PATH.with_suffix(".csv.tmp")
    with emb_tmp.open("wb") as fh:
        np.save(fh, embs)
    index.to_csv(idx_tmp, index=False)
    os.replace(emb_tmp, EMB_PATH)
    os.replace(idx_tmp, INDEX_PATH)


def main() -> None:
    setup_logging(REPORTS / "06_embed.log")
    cfg = load_config(ROOT / "config.yaml")
    model_name = cfg["embed_model"]
    batch_size = int(cfg["embed_batch_size"])
    t_run = time.perf_counter()

    src = DATA / "dataset.csv"
    if not src.exists():
        log.error("%s missing; run 03_build_dataset.py --available-only first", src)
        sys.exit(1)
    dataset = pd.read_csv(src).sort_values("specimen_code").reset_index(drop=True)
    if not dataset["specimen_code"].is_unique:
        log.error("specimen_code is not unique in %s", src)
        sys.exit(1)
    log.info("dataset: %d images", len(dataset))

    old_index, old_embs = load_existing(dataset)
    todo = dataset[~dataset["specimen_code"].isin(old_index["specimen_code"])]
    log.info("to embed: %d images (batch size %d)", len(todo), batch_size)

    new_index, new_embs, skipped = pd.DataFrame(columns=INDEX_COLS), None, []
    if not todo.empty:
        model, preprocess = load_model(model_name)
        new_index, new_embs, skipped = embed(model, preprocess, todo, batch_size)

    parts_idx = [p for p in (old_index, new_index) if not p.empty]
    parts_emb = [e for e in (old_embs, new_embs) if e is not None and len(e)]
    if not parts_idx:
        log.error("nothing embedded and nothing on disk; no outputs written")
        sys.exit(1)
    index = pd.concat(parts_idx, ignore_index=True)
    embs = np.concatenate(parts_emb).astype(np.float32, copy=False)
    order = np.argsort(index["specimen_code"].to_numpy(), kind="stable")
    index = index.iloc[order].reset_index(drop=True)
    embs = np.ascontiguousarray(embs[order])

    write_atomic(index, embs)

    # integrity: re-read what was written, not what is in memory
    check_index = pd.read_csv(INDEX_PATH)
    check_embs = np.load(EMB_PATH, mmap_mode="r")
    assert len(check_index) == check_embs.shape[0], (
        f"row mismatch: index {len(check_index)} vs embeddings {check_embs.shape[0]}")
    assert check_embs.dtype == np.float32, check_embs.dtype
    assert check_index["specimen_code"].is_unique
    log.info("wrote %s %s and %s (%d rows)", EMB_PATH.name, check_embs.shape,
             INDEX_PATH.name, len(check_index))
    log.info("index sha256: %s", sha256_of(INDEX_PATH))

    if skipped:
        log.warning("%d image(s) skipped (no row written):", len(skipped))
        for s in skipped:
            log.warning("  %s  %s", s["specimen_code"], s["image_path"])
    log.info("done: %d newly embedded, %d reused, %d skipped, %d total, %.0fs",
             len(new_index), len(old_index), len(skipped), len(index),
             time.perf_counter() - t_run)


if __name__ == "__main__":
    main()

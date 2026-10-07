#!/usr/bin/env python3
"""Item 3, step 2: BioCLIP 2 embeddings of four image versions for the
masking experiment.

Model, preprocessing and L2 normalisation are those of scripts/06_embed.py
(pinned_model_name, open_clip preprocess). Rows follow data/dataset.csv
sorted by specimen_code, minus the images with exclude=1 in
reports/segmentation/exclude.csv. Row i of every .npy belongs to row i of
the index.

Versions:
- original         data/images/<genus>/<stem>.jpg, re-embedded
- ant_grey         data/segmented/<genus>/<stem>_ant_grey.png
- ant_erased       data/segmented/<genus>/<stem>_ant_erased.png
- scalebar_erased  data/segmented/<genus>/<stem>_scalebar_erased.png; for
                   the images where no scale bar was found (no such file)
                   the original is used and no_scalebar=1

Reads:  config.yaml, data/dataset.csv, data/segmented/manifest.csv,
        data/segmented/, data/images/, reports/segmentation/exclude.csv,
        data/embeddings.npy and data/embeddings_index.csv (sanity check only)
Writes: data/masking/embeddings_<version>.npy (float32, N x 768),
        data/masking/embeddings_index.csv (specimen_code, genus, split,
        image_source, partial, no_scalebar, detail_image),
        reports/masking/30_embed_variants.log

Device: MPS when available, else CPU with every core. Images are decoded
and preprocessed in a DataLoader (cores - 2 workers). Batch sizes 32 and 64
are timed on the first images of the original version; the faster is used
for all four versions. Sanity check: cosine similarity between the
re-embedded originals and data/embeddings.npy (median and minimum).
Run with HF_HUB_OFFLINE=1.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import load_config, pinned_model_name, quiet_logging, setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEG = DATA / "segmented"
OUT = DATA / "masking"
REPORT = ROOT / "reports" / "masking"
EXCLUDE_CSV = ROOT / "reports" / "segmentation" / "exclude.csv"

VERSIONS = ["original", "ant_grey", "ant_erased", "scalebar_erased"]
INDEX_COLS = ["specimen_code", "genus", "split", "image_source",
              "partial", "no_scalebar", "detail_image"]
BATCH_CANDIDATES = [32, 64]
TIMING_IMAGES = 256  # per candidate, after one warm-up batch
SEED = 42

log = logging.getLogger("embed_variants")


class Images(Dataset):
    """Paths to preprocessed tensors; decoding runs in the worker processes."""

    def __init__(self, paths: list[str], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.paths[i]) as im:
            return self.preprocess(im.convert("RGB"))


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    torch.set_num_threads(os.cpu_count() or 1)
    log.warning("MPS unavailable: falling back to CPU (%d torch threads)",
                torch.get_num_threads())
    return torch.device("cpu")


def load_model(name: str, device: torch.device):
    import open_clip  # slow import

    t0 = time.perf_counter()
    with quiet_logging():
        model, _, preprocess = open_clip.create_model_and_transforms(name)
    model.eval().to(device)
    log.info("model %s loaded on %s in %.1fs", name, device, time.perf_counter() - t0)
    return model, preprocess


def build_rows() -> pd.DataFrame:
    """Dataset rows in specimen_code order, without the excluded images, with
    the flag columns and one path column per version."""
    ds = pd.read_csv(DATA / "dataset.csv").sort_values("specimen_code").reset_index(drop=True)
    man = pd.read_csv(SEG / "manifest.csv")[["specimen_code", "scalebar_status"]]
    exc = pd.read_csv(EXCLUDE_CSV)
    assert ds["specimen_code"].is_unique and set(man["specimen_code"]) == set(ds["specimen_code"])

    excluded = set(exc.loc[exc["exclude"] == 1, "specimen_code"])
    partial = set(exc.loc[exc["partial"] == 1, "specimen_code"])
    detail = set(exc.loc[exc["detail_image"].notna(), "specimen_code"])

    rows = ds.merge(man, on="specimen_code", how="left", validate="one_to_one")
    dropped = rows[rows["specimen_code"].isin(excluded)]
    log.info("dataset: %d images; left out (exclude=1): %d", len(rows), len(dropped))
    for r in dropped.itertuples(index=False):
        log.info("  excluded %s  %s", r.specimen_code, r.genus)
    rows = rows[~rows["specimen_code"].isin(excluded)].reset_index(drop=True)

    rows["partial"] = rows["specimen_code"].isin(partial).astype(int)
    rows["detail_image"] = rows["specimen_code"].isin(detail).astype(int)

    stems = rows["image_path"].map(lambda p: Path(p).stem)
    rows["path_original"] = [str(ROOT / p) for p in rows["image_path"]]
    for v in VERSIONS[1:]:
        rows[f"path_{v}"] = [str(SEG / g / f"{s}_{v}.png") for g, s in zip(rows["genus"], stems)]
    has_bar_file = rows["path_scalebar_erased"].map(lambda p: Path(p).exists())
    rows["no_scalebar"] = (~has_bar_file).astype(int)
    mismatch = (rows["no_scalebar"] == 1) != (rows["scalebar_status"] == "not_found")
    assert not mismatch.any(), rows.loc[mismatch, "specimen_code"].tolist()
    rows.loc[~has_bar_file, "path_scalebar_erased"] = rows.loc[~has_bar_file, "path_original"]

    for v in VERSIONS:
        missing = [p for p in rows[f"path_{v}"] if not Path(p).exists()]
        assert not missing, f"{v}: {len(missing)} missing, e.g. {missing[:3]}"
    log.info("kept %d images: partial=%d, detail_image=%d, no_scalebar=%d "
             "(scalebar_erased uses the original for these)",
             len(rows), rows["partial"].sum(), rows["detail_image"].sum(),
             rows["no_scalebar"].sum())
    return rows


def make_loader(paths, preprocess, batch_size, workers) -> DataLoader:
    return DataLoader(Images(paths, preprocess), batch_size=batch_size, shuffle=False,
                      num_workers=workers, persistent_workers=False,
                      prefetch_factor=4 if workers else None)


def run(model, loader, device) -> tuple[np.ndarray, float]:
    """Embed every image of the loader in order; return (embeddings, seconds)."""
    chunks = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for x in loader:
            f = model.encode_image(x.to(device))
            f = f / f.norm(dim=-1, keepdim=True)
            chunks.append(f.to(torch.float32).cpu().numpy())
    if device.type == "mps":
        torch.mps.synchronize()
    return np.concatenate(chunks), time.perf_counter() - t0


def time_batches(model, preprocess, paths, device, workers) -> int:
    """Time each candidate batch size on the same images; return the faster."""
    rates = {}
    for bs in BATCH_CANDIDATES:
        sub = paths[:bs + TIMING_IMAGES]
        loader = make_loader(sub, preprocess, bs, workers)
        it = iter(loader)
        with torch.inference_mode():
            model.encode_image(next(it).to(device))  # warm-up batch, not timed
            if device.type == "mps":
                torch.mps.synchronize()
            t0, n = time.perf_counter(), 0
            for x in it:
                model.encode_image(x.to(device))
                n += len(x)
            if device.type == "mps":
                torch.mps.synchronize()
        rates[bs] = n / (time.perf_counter() - t0)
        log.info("batch size %d: %.1f img/s on %d images", bs, rates[bs], n)
        del loader, it
    best = max(rates, key=rates.get)
    log.info("batch size chosen: %d", best)
    return best


def sanity_check(rows: pd.DataFrame, emb_original: np.ndarray) -> None:
    ref = np.load(DATA / "embeddings.npy", mmap_mode="r")
    ref_index = pd.read_csv(DATA / "embeddings_index.csv")
    pos = pd.Series(np.arange(len(ref_index)), index=ref_index["specimen_code"])
    ref_rows = np.asarray(ref[pos.loc[rows["specimen_code"]].to_numpy()])
    cos = np.sum(ref_rows * emb_original, axis=1)
    worst = np.argsort(cos)[:5]
    log.info("sanity check, re-embedded original vs data/embeddings.npy (%d images): "
             "cosine median %.6f, minimum %.6f, 1st percentile %.6f",
             len(cos), np.median(cos), cos.min(), np.percentile(cos, 1))
    for i in worst:
        log.info("  lowest  %s  %s  %.6f", rows["specimen_code"].iat[i],
                 rows["genus"].iat[i], cos[i])


def main() -> None:
    setup_logging(REPORT / "30_embed_variants.log")
    torch.manual_seed(SEED)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        log.error("set HF_HUB_OFFLINE=1 (no network step)")
        sys.exit(1)
    t_run = time.perf_counter()
    cfg = load_config(ROOT / "config.yaml")
    device = pick_device()
    workers = max((os.cpu_count() or 1) - 2, 0)
    log.info("torch %s, device %s, %d DataLoader workers", torch.__version__, device, workers)

    rows = build_rows()
    model, preprocess = load_model(pinned_model_name(cfg), device)
    batch_size = time_batches(model, preprocess, rows["path_original"].tolist(), device, workers)

    OUT.mkdir(parents=True, exist_ok=True)
    timings = []
    for v in VERSIONS:
        loader = make_loader(rows[f"path_{v}"].tolist(), preprocess, batch_size, workers)
        emb, secs = run(model, loader, device)
        assert emb.shape == (len(rows), 768) and emb.dtype == np.float32, emb.shape
        assert np.isfinite(emb).all(), v
        np.save(OUT / f"embeddings_{v}.npy", emb)
        timings.append((v, len(emb), secs))
        log.info("%s: %d images in %.1fs, %.1f img/s -> %s", v, len(emb), secs,
                 len(emb) / secs, (OUT / f"embeddings_{v}.npy").relative_to(ROOT))
        if v == "original":
            sanity_check(rows, emb)

    rows[INDEX_COLS].to_csv(OUT / "embeddings_index.csv", index=False)
    log.info("wrote %s (%d rows)", (OUT / "embeddings_index.csv").relative_to(ROOT), len(rows))

    log.info("summary (batch size %d, device %s):", batch_size, device)
    log.info("  %-16s %6s %8s %8s", "version", "images", "seconds", "img/s")
    for v, n, s in timings:
        log.info("  %-16s %6d %8.1f %8.1f", v, n, s, n / s)
    log.info("done in %.0fs", time.perf_counter() - t_run)


if __name__ == "__main__":
    main()

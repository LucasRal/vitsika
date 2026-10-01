#!/usr/bin/env python3
"""RISE saliency maps for the BioCLIP + probe pipeline (Petsiuk et al. 2018,
arXiv:1806.07421): which part of a photograph does the deployed classifier
actually use when it names a genus?

RISE is black-box: N random binary masks (s x s cell grid, keep probability
p, upsampled bilinearly to the input size with a random sub-cell shift, as in
the paper) are multiplied onto the input image; each masked image goes
through the frozen BioCLIP 2 tower and the calibrated probe, and the saliency
map is the probability-weighted average of the masks. Bright regions are the
pixels whose visibility drives the probability of the TARGET genus.

    .venv/bin/python scripts/12_rise.py --specimen casent0104549            # target = top-1
    .venv/bin/python scripts/12_rise.py --specimen casent0104549 --genus Anochetus
    .venv/bin/python scripts/12_rise.py --specimen casent0104549 \
        --genus Odontomachus --genus Anochetus                              # A-vs-B on one photo
    .venv/bin/python scripts/12_rise.py --gallery                           # the full report set
    .venv/bin/python scripts/12_rise.py --render-only                       # redraw from saved arrays

Computing and drawing are separate passes. Computing saves each saliency
array (arrays/<specimen>_<Target>.npy) and merges the run into runs.json;
drawing then redraws EVERY run in runs.json in two modes, each with a colour
bar under the map:
    per-map   (<name>.png)         the map stretched to its own min/max, bar
                                   ticked with those real values; shows structure
    shared    (<name>_shared.png)  deviation from the map's own baseline (its
                                   mean) on one diverging scale [-W, +W], with
                                   W = max |deviation| over all maps on the
                                   page; shows which maps carry signal at all

Outputs (reports/rise/):
    <specimen>_<Target>.png, <specimen>_<Target>_shared.png   3 panels each
    web/<specimen>_<Target>_overlay.png, web/..._overlay_shared.png, web/<specimen>_photo.png
    arrays/<specimen>_<Target>.npy
    runs.json                       per-run metadata (+ shared_w), merged across invocations
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gbif_client import load_config, quiet_logging, setup_logging  # noqa: E402
import api.probe  # noqa: E402,F401  (joblib needs the class to unpickle data/probe.pkl)

DATA = ROOT / "data"
REPORTS = ROOT / "reports"
OUT = REPORTS / "rise"
WEB_OUT = OUT / "web"
ARRAYS = OUT / "arrays"
RUNS_JSON = OUT / "runs.json"
PARAMS = ("n_masks", "grid", "p_keep", "probs", "seed")

N_MASKS = 400          # forward passes per run (paper uses 4k-8k on GPU; 400 is
S_CELLS = 7            # readable and CPU-practical, ~3 min per run)
P_KEEP = 0.5
SIDE = 224             # BioCLIP input size
BATCH = 16
LOG_EVERY = 5          # batches
DIVERGING = "RdBu_r"   # shared deviation scale: blue below baseline, red above
OVERLAY_ALPHA = 0.4

# The report set. targets=None means the probe's top-1; two genera give the
# 2018 deck's A-vs-B comparison on the same photograph.
GALLERY: list[tuple[str, tuple[str, ...] | None]] = [
    ("casent0104549", ("Odontomachus", "Anochetus")),   # trap-jaw pair,
    ("casent0102933", ("Anochetus", "Odontomachus")),   # both directions
    ("casent0101358", None),                            # typical Camponotus
    ("casent0499794", ("Technomyrmex", "Tapinoma")),    # our shared-confusion
    ("casent0453991", ("Tapinoma", "Technomyrmex")),    # pair, both directions
    ("casent0005580", None),                            # best genus, trap jaws
]

log = logging.getLogger("rise")


def letterbox(im: Image.Image) -> torch.Tensor:
    """Unit-range (3, SIDE, SIDE) tensor of the image padded to a square with
    its median border colour and resized. open_clip's preprocess is
    Compose([Resize, CenterCrop, ..., ToTensor, Normalize]); RISE masks the
    image itself (black patches), so normalisation must come after the mask,
    and the deployed centre-crop would cut off the long mandibles the analysis
    is about, so the image is letterboxed instead. The full-image probability
    is logged so the deviation from the deployed crop is visible."""
    from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor
    resize = Compose([Resize((SIDE, SIDE), interpolation=InterpolationMode.BICUBIC), ToTensor()])
    a = np.asarray(im)
    border = np.concatenate([a[0].reshape(-1, 3), a[-1].reshape(-1, 3),
                             a[:, 0].reshape(-1, 3), a[:, -1].reshape(-1, 3)])
    bg = tuple(int(v) for v in np.median(border, axis=0))
    side = max(im.size)
    canvas = Image.new("RGB", (side, side), bg)
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return resize(canvas)


def find_normalize(preprocess):
    """The Normalize step of open_clip's preprocess, applied after masking."""
    from torchvision.transforms import Normalize
    norms = [t for t in preprocess.transforms if isinstance(t, Normalize)]
    if len(norms) != 1:
        raise SystemExit(f"expected exactly one Normalize in preprocess, got {len(norms)}")
    return norms[0]


def make_masks(rng: np.random.Generator, n: int = N_MASKS, p: float = P_KEEP) -> np.ndarray:
    """N binary cell grids upsampled bilinearly with a random sub-cell shift
    (Petsiuk et al. sec. 3.2). Returns float32 (n, SIDE, SIDE) in [0, 1]."""
    cell = int(np.ceil(SIDE / S_CELLS))
    up = (S_CELLS + 1) * cell
    grids = (rng.random((n, S_CELLS, S_CELLS)) < p).astype(np.float32)
    t = torch.nn.functional.interpolate(
        torch.from_numpy(grids)[:, None], size=(up, up), mode="bilinear", align_corners=False)
    masks = np.empty((n, SIDE, SIDE), dtype=np.float32)
    for i in range(n):
        dx, dy = rng.integers(0, cell, size=2)
        masks[i] = t[i, 0, dy:dy + SIDE, dx:dx + SIDE].numpy()
    return masks


def rise(model, normalize, base_unit: torch.Tensor, masks: np.ndarray,
         proba_fn, target_idx: int, p_keep: float) -> tuple[np.ndarray, np.ndarray]:
    """Probability of the target class for every masked image; returns
    (saliency SIDE x SIDE, per-mask probabilities)."""
    n = len(masks)
    probs = np.empty(n, dtype=np.float64)
    t0 = time.perf_counter()
    with torch.no_grad():
        for b0 in range(0, n, BATCH):
            m = torch.from_numpy(masks[b0:b0 + BATCH])[:, None]      # (b,1,H,W)
            batch = normalize(base_unit[None] * m)
            feats = model.encode_image(batch)
            feats = (feats / feats.norm(dim=-1, keepdim=True)).to(torch.float32).cpu().numpy()
            probs[b0:b0 + len(m)] = proba_fn(feats)[:, target_idx]
            done = b0 // BATCH + 1
            if done % LOG_EVERY == 0 or b0 + BATCH >= n:
                el = time.perf_counter() - t0
                eta = el / (b0 + len(m)) * (n - b0 - len(m))
                log.info("  %d/%d masks, %.0fs elapsed, ETA %.0fs", b0 + len(m), n, el, eta)
    # per-pixel Monte-Carlo estimate of E[P | pixel visible]; dividing by the
    # actual coverage sum(m_i) rather than n*p removes edge/coverage artefacts
    cover = masks.reshape(n, -1).sum(axis=0)
    sal = ((probs @ masks.reshape(n, -1)) / np.maximum(cover, 1e-6)).reshape(SIDE, SIDE)
    return sal, probs


def colour_bar(fig, ax, mappable, ticks: list[float], labels: list[str],
               label: str, fontsize: float) -> None:
    """Small horizontal bar under `ax`, ticked with real values."""
    cb = fig.colorbar(mappable, ax=ax, orientation="horizontal", fraction=0.045,
                      pad=0.03, aspect=35, ticks=ticks)
    cb.ax.set_xticklabels(labels, fontsize=fontsize)
    cb.ax.tick_params(length=2, pad=1.5)
    cb.set_label(label, fontsize=fontsize)
    cb.outline.set_linewidth(0.4)


def render(base_unit: torch.Tensor, sal: np.ndarray, code: str, true_genus: str,
           target: str, p_full: float, n_masks: int, out_3panel: Path,
           out_overlay: Path, out_photo: Path, shared_w: float | None = None) -> None:
    """Per-map mode (shared_w=None): the map stretched to its own min/max on
    'jet', bar ticked with those real values; shows structure. Shared mode:
    deviation from the map's own baseline (its mean) on a diverging map over
    [-W, +W], W common to every map on the page; shows which maps carry any
    signal at all (a flat map stays at the neutral middle colour)."""
    img = base_unit.permute(1, 2, 0).numpy()
    lo, hi, baseline = float(sal.min()), float(sal.max()), float(sal.mean())
    if shared_w is None:
        heat = (sal - lo) / (hi - lo) if hi > lo else np.zeros_like(sal)
        cmap, vmin, vmax = "jet", 0.0, 1.0
        ticks, labels = [0.0, 1.0], [f"{lo:.3f}", f"{hi:.3f}"]
        bar_label = "saliency, per-map scale"
        heat_title = f"saliency, {n_masks} masks, {S_CELLS}×{S_CELLS} grid"
    else:
        heat = sal - baseline
        cmap, vmin, vmax = DIVERGING, -shared_w, shared_w
        ticks = [-shared_w, 0.0, shared_w]
        labels = [f"−{shared_w:.3f}", "0", f"+{shared_w:.3f}"]
        bar_label = f"deviation from baseline\nbaseline P = {baseline:.3f}"
        heat_title = "deviation from baseline, shared scale"

    fig, axes = plt.subplots(1, 3, figsize=(12, 5.0))
    axes[0].imshow(img)
    axes[0].set_title(f"$\\it{{{true_genus}}}$ · {code}", fontsize=12)
    axes[1].imshow(img)
    axes[1].imshow(heat, cmap=cmap, alpha=OVERLAY_ALPHA, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"RISE for $\\it{{{target}}}$ (full-image P = {p_full:.3f})", fontsize=12)
    opaque = axes[2].imshow(heat, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title(heat_title, fontsize=12)
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    # the overlay's own mappable carries alpha; the bar uses the opaque one
    colour_bar(fig, axes[1], opaque, ticks, labels, bar_label, fontsize=8)
    colour_bar(fig, axes[2], opaque, ticks, labels, bar_label, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_3panel, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4.5))
    ax.imshow(img)
    ax.imshow(heat, cmap=cmap, alpha=OVERLAY_ALPHA, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    opaque = ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap)
    colour_bar(fig, ax, opaque, ticks, labels, bar_label, fontsize=7)
    fig.savefig(out_overlay, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    if not out_photo.exists():
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img)
        ax.set_axis_off()
        fig.savefig(out_photo, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def compute(args, ds: pd.DataFrame) -> dict:
    """RISE for the requested runs: saves the saliency arrays and merges the
    run metadata into runs.json (a re-run of one specimen replaces its own
    (specimen, target) entries and keeps the rest of the gallery)."""
    cfg = load_config(ROOT / "config.yaml")
    torch.set_num_threads(os.cpu_count() or 1)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    import joblib
    import open_clip
    t = time.perf_counter()
    with quiet_logging():
        model, _, preprocess = open_clip.create_model_and_transforms(cfg["embed_model"])
    model.eval()
    log.info("model %s loaded in %.1fs", cfg["embed_model"], time.perf_counter() - t)
    normalize = find_normalize(preprocess)
    probe = joblib.load(DATA / "probe.pkl")
    proba_fn = probe.predict_proba if args.probs == "calibrated" else probe.base.predict_proba
    classes = [str(c) for c in probe.classes_]

    runs = [(args.specimen, tuple(args.genus) if args.genus else None)] \
        if not args.gallery else GALLERY
    rng = np.random.default_rng(args.seed)
    masks = make_masks(rng, args.n_masks, args.p_keep)
    log.info("%d masks, %dx%d grid, p=%.2f, probs=%s, mean coverage %.3f",
             args.n_masks, S_CELLS, S_CELLS, args.p_keep, args.probs, float(masks.mean()))

    results = []
    for code, targets in runs:
        row = ds.loc[code]
        base_unit = letterbox(Image.open(ROOT / row["image_path"]).convert("RGB"))
        with torch.no_grad():
            feats = model.encode_image(normalize(base_unit)[None])
            feats = (feats / feats.norm(dim=-1, keepdim=True)).to(torch.float32).cpu().numpy()
        full_probs = proba_fn(feats)[0]
        top1 = classes[int(full_probs.argmax())]
        for target in targets or (top1,):
            if target not in classes:
                raise SystemExit(f"unknown genus {target!r}; classes: {classes}")
            ti = classes.index(target)
            p_full = float(full_probs[ti])
            log.info("%s (%s): RISE for %s, full-image P=%.4f",
                     code, row["genus"], target, p_full)
            sal, probs = rise(model, normalize, base_unit, masks, proba_fn, ti, args.p_keep)
            np.save(ARRAYS / f"{code}_{target}.npy", sal.astype(np.float32))
            results.append({
                "specimen_code": code, "true_genus": str(row["genus"]),
                "species": str(row["species"]), "target": target,
                "top1": top1, "p_full": round(p_full, 4),
                "p_masked_mean": round(float(probs.mean()), 4),
                "p_masked_max": round(float(probs.max()), 4),
                "creator": str(row["creator"]), "figure": f"{code}_{target}.png",
            })
            log.info("computed %s -> %s (masked P: mean %.3f, max %.3f)",
                     code, target, probs.mean(), probs.max())

    meta = {k: getattr(args, k) for k in ("n_masks", "p_keep", "probs", "seed")}
    meta["grid"] = S_CELLS
    meta["runs"] = results
    if RUNS_JSON.exists() and not args.gallery:
        old = json.loads(RUNS_JSON.read_text())
        differ = {k: old.get(k) for k in PARAMS if old.get(k) != meta[k]}
        if differ:
            log.warning("runs.json parameters differ from this run (%s); keeping the "
                        "old entries anyway", differ)
        fresh = {(r["specimen_code"], r["target"]): r for r in results}
        merged = [fresh.pop((r["specimen_code"], r["target"]), r) for r in old.get("runs", [])]
        meta["runs"] = merged + list(fresh.values())
    log.info("computed %d runs this invocation, %d in runs.json", len(results), len(meta["runs"]))
    return meta


def draw_all(meta: dict, ds: pd.DataFrame) -> dict:
    """Redraw every run in runs.json from its saved array, in per-map and in
    shared-deviation mode; W (the shared half-width) is the max |deviation
    from baseline| over all maps, so it is recomputed page-wide every time."""
    loaded = []
    for r in meta["runs"]:
        path = ARRAYS / f"{r['specimen_code']}_{r['target']}.npy"
        if not path.exists():
            log.warning("no saliency array for %s -> %s, not drawn", r["specimen_code"], r["target"])
            continue
        sal = np.load(path).astype(np.float64)
        r["baseline"] = round(float(sal.mean()), 4)
        r["sal_min"], r["sal_max"] = round(float(sal.min()), 4), round(float(sal.max()), 4)
        r["dev_max"] = round(float(np.abs(sal - sal.mean()).max()), 4)
        loaded.append((r, sal))
    if not loaded:
        raise SystemExit("nothing to draw: no saliency arrays found")
    w = max(float(np.abs(sal - sal.mean()).max()) for _, sal in loaded)
    meta["shared_w"] = round(w, 4)
    log.info("shared deviation scale: W = %.4f over %d maps", w, len(loaded))

    for r, sal in loaded:
        code, target = r["specimen_code"], r["target"]
        row = ds.loc[code]
        base_unit = letterbox(Image.open(ROOT / row["image_path"]).convert("RGB"))
        common = (base_unit, sal, code, str(row["genus"]), target, r["p_full"], meta["n_masks"])
        render(*common, OUT / f"{code}_{target}.png", WEB_OUT / f"{code}_{target}_overlay.png",
               WEB_OUT / f"{code}_photo.png")
        render(*common, OUT / f"{code}_{target}_shared.png",
               WEB_OUT / f"{code}_{target}_overlay_shared.png",
               WEB_OUT / f"{code}_photo.png", shared_w=w)
        log.info("drew %s -> %s: baseline %.4f, range %.4f – %.4f, max |dev| %.4f",
                 code, target, r["baseline"], r["sal_min"], r["sal_max"], r["dev_max"])
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specimen", help="specimen_code from data/dataset.csv")
    ap.add_argument("--genus", action="append",
                    help="target genus, repeatable (default: the probe's top-1)")
    ap.add_argument("--gallery", action="store_true", help="run the full report set")
    ap.add_argument("--render-only", action="store_true",
                    help="skip computing; redraw every run in runs.json from its saved array")
    ap.add_argument("--n-masks", type=int, default=N_MASKS)
    ap.add_argument("--p-keep", type=float, default=P_KEEP)
    ap.add_argument("--probs", choices=("calibrated", "raw"), default="calibrated",
                    help="weight masks by the deployed calibrated probabilities or by "
                         "the raw (T=1) probe softmax; the calibrated ones saturate "
                         "near 0/1 and can leave RISE with no signal")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (args.gallery or args.specimen or args.render_only):
        ap.error("--specimen, --gallery or --render-only required")

    setup_logging(REPORTS / "12_rise.log")
    for d in (OUT, WEB_OUT, ARRAYS):
        d.mkdir(exist_ok=True)
    ds = pd.read_csv(DATA / "dataset.csv").set_index("specimen_code")

    if args.render_only:
        if not RUNS_JSON.exists():
            raise SystemExit(f"{RUNS_JSON} missing; compute first")
        meta = json.loads(RUNS_JSON.read_text())
    else:
        meta = compute(args, ds)
    meta = draw_all(meta, ds)
    RUNS_JSON.write_text(json.dumps(meta, indent=2) + "\n")
    log.info("done: %d runs in runs.json, W = %.4f -> %s", len(meta["runs"]), meta["shared_w"], OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Segment every image of data/dataset.csv with rembg, masks used as-is.

Decision after the trial (reports/segmentation_trial/): rembg's u2net matte
is the segmentation; no card or pin cleaning (scripts/25_clean_mask.py and
26 are kept but not used). For each image, in data/segmented/<genus>/
(gitignored like data/images), with the original's file stem:

  <stem>_mask.png           binary mask, alpha >= 128, every component kept
  <stem>_ant_grey.png       ant only on uniform grey (128, 128, 128), RGB
  <stem>_ant_rgba.png       ant only, RGBA (alpha = the binary mask)
  <stem>_ant_erased.png     original with the mask dilated by ERASE_DILATE px
                            replaced by the image's median border colour
  <stem>_scalebar_erased.png original with the scale-bar box (from
                            reports/confounds/image_stats.csv) replaced by the
                            median border colour; skipped and flagged when no
                            bar was detected

and data/segmented/manifest.csv (also copied to reports/segmentation/), one
row per image: specimen_code, genus, split, view, image_path, width, height,
mask_area_fraction, bbox (x0,y0,x1,y1), n_components (8-connected, any
size) and n_components_30px, mask_touches_border and which edges, the
residue estimates, the scale-bar box and the flags.

Residues, lengths at 1024 px width and scaled by width / 1024:
  card_residue   pale, low-saturation pixels inside the mask: luminance
                 (0.299 R + 0.587 G + 0.114 B) > 150 and chroma (max RGB -
                 min RGB) < 60, after a 2 px opening to drop hairs and
                 highlights. Count and fraction of the mask. A pale ant
                 scores high here without any card; the number is a prompt
                 to look, not a verdict.
  pin_residue    dark, grey, elongated thin parts of the mask that reach a
                 frame edge: thin = mask minus its opening by a 25 px disc,
                 opened by 3 px; a thin component counts when its median
                 luminance < 100 and chroma < 30, it touches the frame and
                 its length / width >= 3 (principal axis). Count and
                 fraction of the mask. Dark legs reaching the frame score
                 here too.

Flags: empty (mask < EMPTY_FRAC of the image), large (mask > LARGE_FRAC of
the image), frame_filling (mask touches all four edges; with large, the
sign of an inverted mask), no_scalebar.

Usage: 27_segment_all.py [--resume] [--limit N]
Network: none (the u2net model is cached in ~/.rembg/models/).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dataset.csv"
STATS = ROOT / "reports" / "confounds" / "image_stats.csv"
OUT = ROOT / "data" / "segmented"
REPORT = ROOT / "reports" / "segmentation"

REMBG_MODEL = "u2net"
ALPHA_T = 128               # alpha >= ALPHA_T is ant
GREY = (128, 128, 128)
ERASE_DILATE = 15           # px, mask dilation for the ant-erased image
BORDER = 10                 # px frame used for the median border colour
BAR_PAD_X, BAR_ABOVE, BAR_BELOW = 20, 10, 50   # scale-bar box margins at 1024 px width (the mm label sits below the bar)
BAR_TRUNC_MIN = 0.12        # a truncated bar is erased over at least this share of the width
CARD_LUM, CARD_CHROMA, CARD_OPEN = 150, 60, 2
PIN_LUM, PIN_CHROMA, PIN_BODY_RADIUS, PIN_OPEN, PIN_MIN_ELONG, PIN_MIN_AREA = 100, 30, 25, 3, 3.0, 100
EMPTY_FRAC, LARGE_FRAC = 0.005, 0.80
SPECK = 30
EIGHT = np.ones((3, 3), bool)

log = logging.getLogger("segment_all")


def disc(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def border_colour(rgb: np.ndarray) -> tuple[int, int, int]:
    b = BORDER
    frame = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                            rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(frame, axis=0))


def scalebar_box(s: pd.Series, w: int, h: int) -> tuple[int, int, int, int] | None:
    """Box (x0, y0, x1, y1) around the detected bar and its label, or None."""
    status = s["scalebar_status"]
    if status not in ("found", "truncated"):
        return None
    k = w / 1024
    cx, cy = s["scalebar_x_frac"] * w, s["scalebar_y_frac"] * h
    if status == "found":
        half = s["scalebar_px"] / 2
        thick = s["scalebar_thickness_px"]
    else:                       # bar touches the left or right edge; the centre is known, the length is not
        half = max(cx if cx < w / 2 else w - cx, BAR_TRUNC_MIN * w / 2)
        thick = 6 * k
    x0, x1 = cx - half - BAR_PAD_X * k, cx + half + BAR_PAD_X * k
    y0, y1 = cy - thick / 2 - BAR_ABOVE * k, cy + thick / 2 + BAR_BELOW * k
    return (int(max(0, x0)), int(max(0, y0)), int(min(w, np.ceil(x1))), int(min(h, np.ceil(y1))))


def components(mask: np.ndarray, min_area: int = 1) -> tuple[int, np.ndarray, int]:
    lab, n = ndimage.label(mask, structure=EIGHT)
    if n == 0:
        return 0, lab, 0
    sizes = np.bincount(lab.ravel())[1:]
    return n, lab, int((sizes >= min_area).sum())


def card_residue(rgb: np.ndarray, mask: np.ndarray, k: float) -> int:
    lum = luminance(rgb.astype(float))
    chroma = rgb.max(-1).astype(int) - rgb.min(-1).astype(int)
    pale = mask & (lum > CARD_LUM) & (chroma < CARD_CHROMA)
    pale = ndimage.binary_opening(pale, structure=disc(max(1, int(round(CARD_OPEN * k)))))
    return int(pale.sum())


def pin_residue(rgb: np.ndarray, mask: np.ndarray, k: float) -> int:
    h, w = mask.shape
    core = ndimage.binary_opening(mask, structure=disc(int(round(PIN_BODY_RADIUS * k))))
    thin = ndimage.binary_opening(mask & ~core, structure=disc(max(1, int(round(PIN_OPEN * k)))))
    if not thin.any():
        return 0
    lum = luminance(rgb.astype(float))
    chroma = rgb.max(-1).astype(int) - rgb.min(-1).astype(int)
    lab, n = ndimage.label(thin, structure=EIGHT)
    total = 0
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        comp = lab[sl] == i
        area = int(comp.sum())
        if area < PIN_MIN_AREA * k * k:
            continue
        ys, xs = np.nonzero(comp)
        ys, xs = ys + sl[0].start, xs + sl[1].start
        if not (xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1):
            continue
        if np.median(lum[ys, xs]) >= PIN_LUM or np.median(chroma[ys, xs]) >= PIN_CHROMA:
            continue
        pts = np.stack([xs, ys], 1).astype(float)
        pts -= pts.mean(0)
        _, vec = np.linalg.eigh(np.cov(pts.T))
        length = float(np.ptp(pts @ vec[:, 1]))
        if length / max(area / max(length, 1), 1e-9) >= PIN_MIN_ELONG:
            total += area
    return total


def process(rgb: np.ndarray, alpha: np.ndarray, stats: pd.Series, paths: dict[str, Path]) -> dict:
    h, w = rgb.shape[:2]
    k = w / 1024
    mask = alpha >= ALPHA_T
    area = int(mask.sum())
    frac = area / (h * w)
    bg = border_colour(rgb)

    Image.fromarray((mask * 255).astype(np.uint8)).save(paths["mask"], compress_level=3)
    grey = np.where(mask[..., None], rgb, np.array(GREY, np.uint8)).astype(np.uint8)
    Image.fromarray(grey).save(paths["ant_grey"], compress_level=3)
    rgba = np.dstack([rgb, (mask * 255).astype(np.uint8)])
    Image.fromarray(rgba, "RGBA").save(paths["ant_rgba"], compress_level=3)
    grown = ndimage.binary_dilation(mask, structure=disc(ERASE_DILATE)) if area else mask
    erased = np.where(grown[..., None], np.array(bg, np.uint8), rgb).astype(np.uint8)
    Image.fromarray(erased).save(paths["ant_erased"], compress_level=3)
    box = scalebar_box(stats, w, h)
    if box is not None:
        x0, y0, x1, y1 = box
        sb = rgb.copy()
        sb[y0:y1, x0:x1] = bg
        Image.fromarray(sb).save(paths["scalebar_erased"], compress_level=3)

    n_all, lab, n_30 = components(mask, SPECK)
    if area:
        ys, xs = np.nonzero(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        edges = [e for e, v in (("top", mask[0]), ("bottom", mask[-1]), ("left", mask[:, 0]), ("right", mask[:, -1])) if v.any()]
    else:
        bbox, edges = (0, 0, 0, 0), []
    card = card_residue(rgb, mask, k) if area else 0
    pin = pin_residue(rgb, mask, k) if area else 0
    flags = []
    if frac < EMPTY_FRAC:
        flags.append("empty")
    if frac > LARGE_FRAC:
        flags.append("large")
    if len(edges) == 4:
        flags.append("frame_filling")
    if box is None:
        flags.append("no_scalebar")
    return {"width": w, "height": h, "mask_px": area, "mask_area_fraction": round(frac, 5),
            "bbox": ",".join(map(str, bbox)), "n_components": n_all, "n_components_30px": n_30,
            "mask_touches_border": bool(edges), "border_edges": ";".join(edges),
            "card_residue_px": card, "card_residue_frac": round(card / max(area, 1), 5),
            "pin_residue_px": pin, "pin_residue_frac": round(pin / max(area, 1), 5),
            "border_rgb": ",".join(map(str, bg)), "scalebar_status": stats["scalebar_status"],
            "scalebar_box": ",".join(map(str, box)) if box else "",
            "flags": ";".join(flags)}


KINDS = ("mask", "ant_grey", "ant_rgba", "ant_erased", "scalebar_erased")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--resume", action="store_true", help="skip images whose five outputs and manifest row exist")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N images (smoke test)")
    a = ap.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    setup_logging(REPORT / "27_segment_all.log")
    d = pd.read_csv(DATASET).sort_values("image_path").reset_index(drop=True)
    stats = pd.read_csv(STATS).set_index("image_path")
    if a.limit:
        d = d.head(a.limit)
    manifest_path = OUT / "manifest.csv"
    done: dict[str, dict] = {}
    if a.resume and manifest_path.exists():
        prev = pd.read_csv(manifest_path, keep_default_na=False)
        done = {r["image_path"]: r for r in prev.to_dict("records")}
        log.info("resume: %d rows already in %s", len(done), manifest_path)

    from rembg import new_session, remove
    t0 = time.perf_counter()
    session = new_session(REMBG_MODEL)
    log.info("rembg %s loaded in %.1f s; %d images", REMBG_MODEL, time.perf_counter() - t0, len(d))

    rows, times = [], []
    t_start = time.perf_counter()
    for i, r in enumerate(d.itertuples(), start=1):
        stem = Path(r.image_path).stem
        gdir = OUT / r.genus
        paths = {kind: gdir / f"{stem}_{kind}.png" for kind in KINDS}
        if r.image_path in done and all(paths[k].exists() for k in KINDS if k != "scalebar_erased"):
            rows.append(done[r.image_path])
            continue
        gdir.mkdir(parents=True, exist_ok=True)
        with Image.open(ROOT / r.image_path) as im:
            pil = im.convert("RGB")
        rgb = np.array(pil)
        t1 = time.perf_counter()
        alpha = np.asarray(remove(pil, session=session))[..., 3]
        t_rembg = time.perf_counter() - t1
        t2 = time.perf_counter()
        info = process(rgb, alpha, stats.loc[r.image_path], paths)
        t_post = time.perf_counter() - t2
        times.append(t_rembg)
        row = {"specimen_code": r.specimen_code, "genus": r.genus, "split": r.split, "view": r.view,
               "image_path": r.image_path, **info, "rembg_s": round(t_rembg, 3), "post_s": round(t_post, 3)}
        rows.append(row)
        if info["flags"] or i % 50 == 0:
            log.info("%4d/%d %s %s: mask %.1f %%, %d comp, border %s, card %.3f, pin %.3f, flags [%s], rembg %.2f s, post %.2f s",
                     i, len(d), r.genus, r.specimen_code, 100 * info["mask_area_fraction"], info["n_components"],
                     info["border_edges"] or "no", info["card_residue_frac"], info["pin_residue_frac"],
                     info["flags"], t_rembg, t_post)
        if i % 100 == 0:
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
    m = pd.DataFrame(rows)
    m.to_csv(manifest_path, index=False)
    shutil.copy(manifest_path, REPORT / "manifest.csv")
    wall = time.perf_counter() - t_start
    log.info("done: %d rows, wall %.0f s, rembg %.2f s per image (min %.2f, max %.2f) over %d new images",
             len(m), wall, float(np.mean(times)) if times else 0, min(times, default=0), max(times, default=0), len(times))
    fl = m["flags"].fillna("").astype(str)
    for f in ("empty", "large", "frame_filling", "no_scalebar"):
        log.info("flag %-14s %d", f, fl.str.contains(f).sum())
    log.info("card residue > 5 %%: %d; pin residue > 5 %%: %d", (m.card_residue_frac > 0.05).sum(), (m.pin_residue_frac > 0.05).sum())


if __name__ == "__main__":
    main()

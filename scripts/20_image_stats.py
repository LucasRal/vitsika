#!/usr/bin/env python3
"""Per-image photo-condition measurements for the confound analysis.

For every row of data/dataset.csv, writes one row to
reports/confounds/image_stats.csv. Offline; reads only data/ and writes only
reports/confounds/.

Measured from the pixels (luminance = 0.299 R + 0.587 G + 0.114 B):

  border_r, border_g, border_b   median RGB of the border region: every pixel
                                 within 5 % of the width (left/right) or 5 % of
                                 the height (top/bottom) of an edge
  border_brightness              mean luminance of that region (0-255)
  warm_cool                      border_r - border_b (> 0 warm, < 0 cool)
  width, height, aspect, file_bytes
  dark_frac                      share of the image with luminance below half
                                 the border's median luminance (ant + pin +
                                 scale bar; relative, so exposure-independent)
  dark_frac_abs                  share with luminance below 60 (absolute)
  scalebar_*                     see detect_scalebar()

The scale bar's mm label (0.2 / 0.5 / 1 / 2 mm ...) is drawn into the
pixels only; neither the image file name nor the metadata (image_title,
image_url, GBIF fields) carries it, and no OCR is available offline, so
scalebar_mm is left empty for every row. scalebar_px alone does not give the
magnification.

Joined columns: genus, subfamily, creator, image_source, typeStatus,
is_type, rightsHolder, rights_year_start / rights_year_end (the year range in
rightsHolder, e.g. "California Academy of Sciences, 2000-2008"), split, and
from data/image_manifest.csv sha256, source_url, match_method, source3
(commons_original / commons_thumbnail / gbif_cache) and is_commons_thumb_1280.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_DIR = ROOT / "reports" / "confounds"
OUT = OUT_DIR / "image_stats.csv"

BORDER = 0.05
DARK_ABS = 60
# scale bar: a solid near-black horizontal rectangle near the bottom
BAR_LUM = 60          # bar pixels are near black
BAR_REGION = 0.35     # bottom 35 % of the image (some bars sit at ~70 % height)
BAR_MIN_LEN = 20      # px
BAR_THICK = (2, 16)   # px
BAR_MIN_ELONG = 5.0   # length / thickness
BAR_MIN_FILL = 0.50   # share of the bounding box that is dark (end ticks and
                      # anti-aliased edge rows lower it)
BAR_FULL_ROW = 0.97   # a row counts as full if dark over this share of the length
BAR_MIN_FULL_ROWS = 2 # ... and the bar needs at least this many full rows
BAR_CLEAR = 0.80      # share of the rows just above/below the bar that is not dark
# second style: a thin (1-4 px) straight line, darker or lighter than its
# surroundings, usually with end ticks (older AntWeb images)
LINE_K = 3            # compare each row with the rows 3 px above and below
LINE_T = 10           # ... and require a same-sign difference above this (0-255)
LINE_MIN_LEN = 60     # px
LINE_MAX_TH = 4       # px
LINE_ROW = 0.90       # at least one row covered over this share of the length

log = logging.getLogger("image_stats")


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def border_mask(h: int, w: int) -> np.ndarray:
    bh, bw = max(1, round(BORDER * h)), max(1, round(BORDER * w))
    m = np.zeros((h, w), bool)
    m[:bh], m[-bh:], m[:, :bw], m[:, -bw:] = True, True, True, True
    return m


def detect_thin_line(lum: np.ndarray) -> dict | None:
    """Longest thin horizontal line in the bottom BAR_REGION of the image.

    A row belongs to a line where it differs from both the row LINE_K above
    and the row LINE_K below in the same direction by more than LINE_T; a
    step edge (card edge, shadow) differs from one side only and is ignored.
    """
    h, w = lum.shape
    y0 = max(int(h * (1 - BAR_REGION)), LINE_K)
    band = lum[y0 - LINE_K:]
    mid = band[LINE_K:-LINE_K]
    up, dn = mid - band[:-2 * LINE_K], mid - band[2 * LINE_K:]
    resp = np.where(np.sign(up) == np.sign(dn), np.sign(up) * np.minimum(abs(up), abs(dn)), 0)
    best = None
    for sign in (1, -1):
        labels, _ = ndimage.label(sign * resp > LINE_T)
        for i, sl in enumerate(ndimage.find_objects(labels), start=1):
            ys, xs = sl
            th, ln = ys.stop - ys.start, xs.stop - xs.start
            if ln < LINE_MIN_LEN or th > LINE_MAX_TH:
                continue
            if (labels[sl] == i).mean(axis=1).max() < LINE_ROW:
                continue
            if best is None or ln > best["len"]:
                best = {"len": ln, "thick": th, "x0": xs.start, "x1": xs.stop,
                        "y": y0 + (ys.start + ys.stop) / 2, "edge": xs.start == 0 or xs.stop == w}
    return best


def detect_scalebar(lum: np.ndarray) -> dict:
    """Find the scale bar in the bottom 35 % of the image.

    Candidates are connected components of near-black pixels that form a
    solid, thin, horizontal rectangle (at least two rows dark over its whole
    length) with non-dark pixels directly above and below (so not part of
    the ant, the pin or a dark background). If none is found, the thin-line
    style is tried (detect_thin_line). scalebar_style says which style
    matched: black_bar, thin_line or none. Status:
      found      exactly one candidate, not touching the left/right edge
      truncated  the best candidate touches the left or right image edge
                 (cropped images): its length is a lower bound only
      ambiguous  several candidates of different lengths
      not_found  none
    Length is reported for 'found' only; other statuses leave it empty.
    """
    h, w = lum.shape
    y0 = int(h * (1 - BAR_REGION))
    region = lum[y0:]
    dark = region < BAR_LUM
    labels, n = ndimage.label(dark)
    cands = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        ys, xs = sl
        th, ln = ys.stop - ys.start, xs.stop - xs.start
        if ln < BAR_MIN_LEN or not (BAR_THICK[0] <= th <= BAR_THICK[1]) or ln / th < BAR_MIN_ELONG:
            continue
        comp = labels[sl] == i
        if comp.mean() < BAR_MIN_FILL or (comp.mean(axis=1) >= BAR_FULL_ROW).sum() < BAR_MIN_FULL_ROWS:
            continue
        above = dark[max(ys.start - 2, 0):ys.start, xs]
        below = dark[ys.stop:ys.stop + 2, xs]
        clear = [1 - b.mean() for b in (above, below) if b.size]
        if clear and min(clear) < BAR_CLEAR:
            continue
        cands.append({"len": ln, "thick": th, "x0": xs.start, "x1": xs.stop,
                      "y": y0 + (ys.start + ys.stop) / 2,
                      "edge": xs.start == 0 or xs.stop == w})
    out = {"scalebar_style": "none", "scalebar_status": "not_found", "scalebar_px": np.nan,
           "scalebar_frac_width": np.nan, "scalebar_thickness_px": np.nan,
           "scalebar_x_frac": np.nan, "scalebar_y_frac": np.nan,
           "scalebar_n_candidates": len(cands), "scalebar_mm": np.nan}
    if cands:
        out["scalebar_style"] = "black_bar"
        best = max(cands, key=lambda c: c["len"])
        others = [c for c in cands if c is not best and c["len"] >= 0.5 * best["len"]]
    else:
        best, others = detect_thin_line(lum), []
        if best is None:
            return out
        out["scalebar_style"] = "thin_line"
    status = "truncated" if best["edge"] else ("ambiguous" if others else "found")
    out.update({"scalebar_status": status,
                "scalebar_x_frac": round((best["x0"] + best["x1"]) / 2 / w, 3),
                "scalebar_y_frac": round(best["y"] / h, 3)})
    if status == "found":
        out.update({"scalebar_px": best["len"], "scalebar_frac_width": round(best["len"] / w, 4),
                    "scalebar_thickness_px": best["thick"]})
    return out


def measure(path: Path) -> dict:
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    lum = luminance(rgb)
    bm = border_mask(h, w)
    med = np.median(rgb[bm], axis=0)
    bg_lum_med = float(np.median(lum[bm]))
    rec = {"width": w, "height": h, "aspect": round(w / h, 4), "file_bytes": path.stat().st_size,
           "border_r": float(med[0]), "border_g": float(med[1]), "border_b": float(med[2]),
           "border_brightness": round(float(lum[bm].mean()), 2),
           "warm_cool": float(med[0] - med[2]),
           "dark_frac": round(float((lum < 0.5 * bg_lum_med).mean()), 5),
           "dark_frac_abs": round(float((lum < DARK_ABS).mean()), 5)}
    rec.update(detect_scalebar(lum))
    return rec


def rights_years(s: str) -> tuple[float, float]:
    m = re.search(r"(\d{4})\s*-\s*(\d{4})", str(s))
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"(\d{4})", str(s))
    return (float(m.group(1)),) * 2 if m else (np.nan, np.nan)


def source3(row: pd.Series) -> str:
    if row["image_source"] == "gbif_cache":
        return "gbif_cache"
    if str(row["match_method"]).startswith("Commons server thumbnail"):
        return "commons_thumbnail"
    return "commons_original"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT_DIR / "20_image_stats.log")
    df = pd.read_csv(DATA / "dataset.csv")
    man = pd.read_csv(DATA / "image_manifest.csv")
    assert list(man["image_path"]) == list(df["image_path"]), "manifest out of step with dataset.csv"

    rows = [measure(ROOT / p) for p in tqdm(df["image_path"], unit="img")]
    stats = pd.DataFrame(rows)

    meta = df[["specimen_code", "genus", "subfamily", "species", "creator", "image_source",
               "typeStatus", "rightsHolder", "split", "view", "image_path"]].copy()
    meta["is_type"] = meta["typeStatus"].notna()
    years = meta["rightsHolder"].map(rights_years)
    meta["rights_year_start"] = [y[0] for y in years]
    meta["rights_year_end"] = [y[1] for y in years]
    meta = meta.join(man[["sha256", "source_url", "match_method"]])
    meta["source3"] = meta.apply(source3, axis=1)
    meta["is_commons_thumb_1280"] = (meta["source3"] == "commons_thumbnail") & (stats["width"] == 1280)
    out = pd.concat([meta, stats], axis=1)
    out.to_csv(OUT, index=False)

    log.info("wrote %s: %d rows, %d columns", OUT, len(out), out.shape[1])
    log.info("source3: %s", out["source3"].value_counts().to_dict())
    log.info("commons thumbnails at 1280 px: %d", int(out["is_commons_thumb_1280"].sum()))
    log.info("scale bar style x status:\n%s", pd.crosstab(out["scalebar_style"], out["scalebar_status"]).to_string())
    log.info("border brightness: median %.1f, IQR %.1f-%.1f; warm_cool median %.1f, IQR %.1f-%.1f",
             out["border_brightness"].median(), *out["border_brightness"].quantile([.25, .75]),
             out["warm_cool"].median(), *out["warm_cool"].quantile([.25, .75]))
    log.info("rights years: start %s, end %s", sorted(out["rights_year_start"].dropna().unique()),
             sorted(out["rights_year_end"].dropna().unique()))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Second segmentation step: remove the card point and the pin from a rembg mask.

Follows the trial in reports/segmentation_trial/ (rembg is the primary
segmenter). For every row of a selection CSV (default: the five trial
images), reads <masks>/<specimen>_rembg_alpha.png and writes to <out>:

  <k>_<genus>_<specimen>.jpg   contact sheet: original | rembg mask | removed
                               regions (card yellow, pin red, dropped pieces
                               magenta, kept pale blobs outlined green) |
                               final ant only on grey 128
  masks/<specimen>_*.png       card, pin and clean masks
  candidates.csv               every card and pin candidate with its features,
                               the tests it passed or failed, and an empty
                               true_label column to fill by hand
  margins.csv                  candidate tests within MARGIN of a threshold
  summary.csv                  pixels removed per image, frame-edge contact
  25_clean_mask.log

Usage: 25_clean_mask.py [--selection CSV] [--masks DIR] [--out DIR]

Rules (all lengths in pixels at 1024 px width, scaled by width / 1024):

  mask     rembg alpha > 0.5.
  card     pale pixels inside the mask (luminance > CARD_LUM), opened by
           2 px, in two tiers by chroma (max RGB - min RGB):
             white tier, chroma < CARD_WHITE_CHROMA: white or grey card
               points. Accepted when area >= WHITE_MIN_AREA of the image,
               largest inscribed disc radius >= WHITE_MIN_RADIUS, solidity
               >= WHITE_MIN_SOLIDITY, the outline has a straight run
               >= WHITE_MIN_EDGE and at least CARD_MIN_BACKGROUND of the
               outline borders the background rather than the ant (a card
               point sticks out of the ant; pale bands on a gaster do not).
             cream tier, CARD_WHITE_CHROMA <= chroma < CARD_MAX_CHROMA: the
               lit cream card of the Pheidole image has chroma up to 90,
               which overlaps pale yellow ant parts, so the shape tests are
               strict: area >= CREAM_MIN_AREA, radius >= CREAM_MIN_RADIUS,
               solidity >= CREAM_MIN_SOLIDITY, a straight run
               >= max(CREAM_MIN_EDGE, CREAM_EDGE_FRAC * sqrt(area)) and the
               same background test.
           The straight run is the Hough peak of the component outline: the
           most outline pixels on one straight line (gaps allowed, so a card
           edge interrupted by a leg still counts). It is what keeps
           pale ant parts: a gaster or femur has a curved outline, a card
           point has straight cut edges.
           Shaded card: an accepted card is then grown into connected mask
           pixels of similar hue (within GROW_HUE_TOL degrees of the card's
           hue; any hue when the pixel or the card is nearly grey), with
           chroma at most the card's median + GROW_CHROMA_TOL and luminance
           >= GROW_MIN_LUM, never into the body core (see pin). This takes
           the darker part of a card next to the ant, which the absolute
           CARD_LUM test misses.
           Pale blobs that fail the tests are "glue": kept and counted.
  pin      body core = mask opened with a disc of radius BODY_RADIUS (the
           thick parts: head, mesosoma, gaster). Thin parts = mask minus
           core, opened by THIN_OPEN px to drop hairs. A thin connected
           component is a pin when it is long (length >= PIN_MIN_LEN), not
           wider than PIN_MAX_WIDTH (width = area / length), elongated
           (length / width >= PIN_MIN_ELONG), straight (share of pixels
           within 0.75 width of the fitted axis >= PIN_MIN_STRAIGHT), of
           even width (coefficient of variation of the width along the axis
           <= PIN_MAX_WIDTH_CV), touches the image border (every pin leaves
           the frame) and is dark and grey (median luminance <= PIN_MAX_LUM,
           median chroma <= PIN_MAX_CHROMA). Only thin-part pixels are
           removed: where the pin passes behind or through the body, the
           core is left untouched by construction.
  keep     after removal, the largest connected component plus every
           component reachable from it through gaps <= BRIDGE px (repeated
           dilation), so legs and antennae separated by a removed region
           stay. Everything else is dropped and counted.

Every candidate gets the same feature set (area, inscribed radius,
solidity, straight edge, background share, chroma, luminance, elongation)
whatever its kind, so the two rules can be compared on hand labels.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "reports" / "segmentation_trial"

ALPHA_FG = 0.5
CARD_LUM = 150
CARD_WHITE_CHROMA = 50
CARD_MAX_CHROMA = 95
WHITE_MIN_AREA, WHITE_MIN_RADIUS, WHITE_MIN_SOLIDITY, WHITE_MIN_EDGE = 0.001, 8, 0.45, 50
CREAM_MIN_AREA, CREAM_MIN_RADIUS, CREAM_MIN_SOLIDITY, CREAM_MIN_EDGE, CREAM_EDGE_FRAC = 0.01, 18, 0.6, 70, 0.6
CARD_MIN_BACKGROUND = 0.15
EDGE_BIN = 3
BG_TOL = 4
CARD_GROW = 3
GROW_MIN_LUM = 110
GROW_HUE_TOL = 15           # degrees
GROW_CHROMA_TOL = 20
GROW_GREY = 15              # chroma below this: no usable hue
GROW_MAX_ITER = 600
GLUE_MIN_AREA = 150         # smaller pale blobs are not even listed
BODY_RADIUS = 40
THIN_OPEN = 3
THIN_MIN_AREA = 300
PIN_MIN_LEN = 100
PIN_MAX_WIDTH = 80
PIN_MIN_ELONG = 3.0
PIN_MIN_STRAIGHT = 0.90
PIN_MAX_WIDTH_CV = 0.25
PIN_MAX_LUM = 130
PIN_MAX_CHROMA = 25
BRIDGE = 6
SPECK = 30
MARGIN = 0.10               # tests within this relative distance of a threshold are listed

GREY = 128
PANEL_W = 640
CAPTION_H = 24
CARD_COLOR, PIN_COLOR, DROP_COLOR, GLUE_COLOR = (237, 161, 0), (227, 73, 72), (232, 123, 164), (27, 175, 122)
FEATURES = ["area", "area_frac", "radius", "solidity", "edge", "background_share", "median_chroma",
            "median_lum", "elongation", "length", "width", "straight", "width_cv", "touches_border"]

log = logging.getLogger("clean_mask")


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def hue_deg(rgb: np.ndarray) -> np.ndarray:
    """HSV hue in degrees, 0 where the pixel is grey."""
    r, g, b = (rgb[..., i].astype(float) for i in range(3))
    mx, mn = rgb.max(-1).astype(float), rgb.min(-1).astype(float)
    d = np.where(mx > mn, mx - mn, 1.0)
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4))
    return np.where(mx > mn, h * 60.0, 0.0)


def disc(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def components(mask: np.ndarray, min_area: int = 1) -> list[np.ndarray]:
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        c = lab == i
        if c.sum() >= min_area:
            out.append(c)
    return out


def longest_edge(comp: np.ndarray) -> float:
    """Straight-edge support: the most outline pixels on one straight line
    (EDGE_BIN px wide, angles in 0.5 degree steps). Gaps along the line do not
    matter, so a card edge interrupted by a leg still counts; a curved outline
    only offers short tangent chords."""
    outline = comp & ~ndimage.binary_erosion(comp)
    ys, xs = np.nonzero(outline)
    best = 0
    for theta in np.linspace(0, np.pi, 360, endpoint=False):
        d = np.floor((xs * np.cos(theta) + ys * np.sin(theta)) / EDGE_BIN).astype(int)
        best = max(best, int(np.bincount(d - d.min()).max()))
    return float(best)


def background_share(comp: np.ndarray, mask: np.ndarray) -> float:
    """Share of the component outline within BG_TOL px of the background."""
    outline = comp & ~ndimage.binary_erosion(comp)
    near_bg = ndimage.binary_dilation(~mask, structure=disc(BG_TOL))
    return float((outline & near_bg).sum() / max(outline.sum(), 1))


def shape_features(comp: np.ndarray, mask: np.ndarray, lum: np.ndarray, chroma: np.ndarray) -> dict:
    props = measure.regionprops(comp.astype(np.uint8))[0]
    h, w = mask.shape
    area = int(comp.sum())
    return {"area": area, "area_frac": area / (h * w),
            "radius": float(ndimage.distance_transform_edt(comp).max()),
            "solidity": float(props.solidity), "edge": longest_edge(comp),
            "background_share": background_share(comp, mask),
            "median_chroma": float(np.median(chroma[comp])), "median_lum": float(np.median(lum[comp])),
            "elongation": float(props.axis_major_length / max(props.axis_minor_length, 1e-9))}


def check(tests: dict[str, tuple[float, float, str]]) -> tuple[bool, str, list[dict]]:
    """tests: name -> (value, threshold, '>=' or '<='). Returns accepted, failed names, margin rows."""
    failed, rows = [], []
    for name, (value, thr, op) in tests.items():
        ok = value >= thr if op == ">=" else value <= thr
        if not ok:
            failed.append(name)
        rel = abs(value - thr) / thr if thr else (0.0 if value == thr else np.inf)
        rows.append({"test": name, "value": value, "threshold": thr, "op": op, "passed": ok, "rel_distance": rel})
    return not failed, ",".join(failed), rows


def grow_shaded(seed: np.ndarray, rgb: np.ndarray, mask: np.ndarray, core: np.ndarray,
                lum: np.ndarray, chroma: np.ndarray, hue: np.ndarray) -> np.ndarray:
    """Grow an accepted card into connected, similar-hue, darker card pixels (not into the core)."""
    card_chroma = float(np.median(chroma[seed]))
    if card_chroma < GROW_GREY:
        hue_ok = np.ones_like(mask)
    else:
        vec = np.exp(1j * np.deg2rad(hue[seed]))
        card_hue = np.rad2deg(np.angle(vec.mean())) % 360
        diff = np.abs((hue - card_hue + 180) % 360 - 180)
        hue_ok = (diff <= GROW_HUE_TOL) | (chroma < GROW_GREY)
    allowed = mask & ~core & (lum >= GROW_MIN_LUM) & hue_ok & (chroma <= card_chroma + GROW_CHROMA_TOL)
    grown = seed.copy()
    for _ in range(GROW_MAX_ITER):
        new = ndimage.binary_dilation(grown) & allowed | grown
        if new.sum() == grown.sum():
            break
        grown = new
    return grown


def card_candidates(rgb: np.ndarray, mask: np.ndarray, core: np.ndarray, s: float) -> list[dict]:
    lum = luminance(rgb.astype(float))
    chroma = rgb.max(-1).astype(int) - rgb.min(-1).astype(int)
    hue = hue_deg(rgb)
    pale = mask & (lum > CARD_LUM) & (chroma < CARD_MAX_CHROMA)
    pale = ndimage.binary_opening(pale, structure=disc(2))
    h, w = mask.shape
    out = []
    for comp in components(pale, int(GLUE_MIN_AREA * s * s)):
        f = shape_features(comp, mask, lum, chroma)
        if f["median_chroma"] < CARD_WHITE_CHROMA:
            tier = "white"
            tests = {"area": (f["area"], WHITE_MIN_AREA * h * w, ">="), "radius": (f["radius"], WHITE_MIN_RADIUS * s, ">="),
                     "solidity": (f["solidity"], WHITE_MIN_SOLIDITY, ">="), "edge": (f["edge"], WHITE_MIN_EDGE * s, ">="),
                     "background": (f["background_share"], CARD_MIN_BACKGROUND, ">=")}
        else:
            tier = "cream"
            tests = {"area": (f["area"], CREAM_MIN_AREA * h * w, ">="), "radius": (f["radius"], CREAM_MIN_RADIUS * s, ">="),
                     "solidity": (f["solidity"], CREAM_MIN_SOLIDITY, ">="),
                     "edge": (f["edge"], max(CREAM_MIN_EDGE * s, CREAM_EDGE_FRAC * np.sqrt(f["area"])), ">="),
                     "background": (f["background_share"], CARD_MIN_BACKGROUND, ">=")}
        accepted, failed, rows = check(tests)
        region, grown_px = comp, 0
        if accepted:
            if CARD_GROW:
                near = ndimage.binary_dilation(comp, structure=disc(int(round(CARD_GROW * s))))
                region = comp | (near & mask & (lum > CARD_LUM - 10) & (chroma < CARD_MAX_CHROMA + 15))
            region = grow_shaded(region, rgb, mask, core, lum, chroma, hue)
            grown_px = int(region.sum() - comp.sum())
        out.append({"kind": "card", "tier": tier, "mask": region, **f, "edge_needed": tests["edge"][1],
                    "grown_px": grown_px, "accepted": accepted, "failed": failed, "checks": rows})
    return out


def pin_candidates(rgb: np.ndarray, mask: np.ndarray, core: np.ndarray, s: float) -> list[dict]:
    lum = luminance(rgb.astype(float))
    chroma = rgb.max(-1).astype(int) - rgb.min(-1).astype(int)
    thin = ndimage.binary_opening(mask & ~core, structure=disc(int(round(THIN_OPEN * s))))
    h, w = mask.shape
    out = []
    for comp in components(thin, int(THIN_MIN_AREA * s * s)):
        f = shape_features(comp, mask, lum, chroma)
        ys, xs = np.nonzero(comp)
        pts = np.stack([xs, ys], 1).astype(float)
        pts -= pts.mean(0)
        _, vec = np.linalg.eigh(np.cov(pts.T))
        major, minor = pts @ vec[:, 1], pts @ vec[:, 0]
        length = float(major.max() - major.min())
        width = float(len(pts) / max(length, 1))
        straight = float((np.abs(minor) <= 0.75 * width).mean())
        bins = np.minimum(((major - major.min()) / max(length, 1) * 10).astype(int), 9)
        profile = np.bincount(bins, minlength=10) / (length / 10)   # width per tenth of the length
        width_cv = float(profile[1:-1].std() / max(profile[1:-1].mean(), 1e-9))  # ends excluded
        border = bool(xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1)
        tests = {"length": (length, PIN_MIN_LEN * s, ">="), "width": (width, PIN_MAX_WIDTH * s, "<="),
                 "elongation": (length / width, PIN_MIN_ELONG, ">="), "straight": (straight, PIN_MIN_STRAIGHT, ">="),
                 "even_width": (width_cv, PIN_MAX_WIDTH_CV, "<="), "border": (float(border), 1.0, ">="),
                 "lum": (f["median_lum"], PIN_MAX_LUM, "<="), "chroma": (f["median_chroma"], PIN_MAX_CHROMA, "<=")}
        accepted, failed, rows = check(tests)
        out.append({"kind": "pin", "tier": "", "mask": comp, **f, "length": length, "width": width,
                    "straight": straight, "width_cv": width_cv, "touches_border": border,
                    "accepted": accepted, "failed": failed, "checks": rows})
    return out


def keep_connected(mask: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    """Largest component plus everything reachable through gaps <= BRIDGE."""
    comps = components(mask)
    if not comps:
        return mask, np.zeros_like(mask)
    keep = max(comps, key=lambda c: c.sum())
    struct = disc(int(round(BRIDGE * s)))
    while True:
        reach = ndimage.binary_dilation(keep, structure=struct) & mask
        new = keep.copy()
        for c in comps:
            if (c & reach).any():
                new |= c
        if new.sum() == keep.sum():
            break
        keep = new
    return keep, mask & ~keep


def n_components(mask: np.ndarray) -> int:
    return len(components(mask, SPECK))


def edge_contact(mask: np.ndarray, s: float) -> str:
    """Which frame edges the mask touches over at least 10 px: legs or antennae reaching the frame."""
    n = int(10 * s)
    sides = {"top": mask[0], "bottom": mask[-1], "left": mask[:, 0], "right": mask[:, -1]}
    return ";".join(k for k, v in sides.items() if v.sum() >= n)


def clean_image(rgb: np.ndarray, alpha: np.ndarray) -> dict:
    h, w = rgb.shape[:2]
    s = w / 1024
    mask = alpha > ALPHA_FG
    core = ndimage.binary_opening(mask, structure=disc(int(round(BODY_RADIUS * s))))
    cards = card_candidates(rgb, mask, core, s)
    card, glue = np.zeros_like(mask), np.zeros_like(mask)
    for c in cards:
        (card if c["accepted"] else glue)[c["mask"]] = True
    pins = pin_candidates(rgb, mask & ~card, core, s)
    pin = np.zeros_like(mask)
    for p in pins:
        if p["accepted"]:
            pin[p["mask"]] = True
    keep, dropped = keep_connected(mask & ~card & ~pin, s)
    return {"mask": mask, "core": core, "cards": cards, "pins": pins, "card": card, "pin": pin,
            "glue": glue, "keep": keep, "dropped": dropped, "scale": s}


def font(size: int = 15):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def panel(im: Image.Image, caption: str, f, width: int = PANEL_W) -> Image.Image:
    im = im.convert("RGB")
    im = im.resize((width, int(round(im.height * width / im.width))), Image.LANCZOS)
    out = Image.new("RGB", (width, im.height + CAPTION_H), "white")
    ImageDraw.Draw(out).text((5, 4), caption, fill="black", font=f)
    out.paste(im, (0, CAPTION_H))
    return out


def on_grey(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(mask[..., None], rgb, GREY).astype(np.uint8))


def removed_view(rgb, res) -> Image.Image:
    out = rgb.astype(float) * 0.55 + 50
    out[~res["mask"]] = out[~res["mask"]] * 0.5
    for region, col in ((res["card"], CARD_COLOR), (res["pin"], PIN_COLOR), (res["dropped"], DROP_COLOR)):
        out[region] = 0.25 * out[region] + 0.75 * np.array(col)
    edge = ndimage.binary_dilation(res["glue"], iterations=2) & ~res["glue"]
    out[edge] = GLUE_COLOR
    return Image.fromarray(out.round().astype(np.uint8))


def contact_sheet(rgb: np.ndarray, res: dict, title: str, f) -> Image.Image:
    panels = [panel(Image.fromarray(rgb), "original", f),
              panel(Image.fromarray(res["mask"]), f"rembg mask, alpha > {ALPHA_FG}", f),
              panel(removed_view(rgb, res), "removed: card yellow, pin red, dropped pieces magenta; kept pale blobs outlined green", f),
              panel(on_grey(rgb, res["keep"]), f"final ant only on grey {GREY}, {res['keep'].sum():,} px, "
                                               f"{n_components(res['keep'])} component(s)", f)]
    ph = max(p.height for p in panels)
    sheet = Image.new("RGB", (len(panels) * PANEL_W + (len(panels) - 1) * 6, ph + 30), "white")
    ImageDraw.Draw(sheet).text((5, 7), title, fill="black", font=f)
    for i, p in enumerate(panels):
        sheet.paste(p, (i * (PANEL_W + 6), 30))
    return sheet


def margins(cands: list[dict]) -> pd.DataFrame:
    """Tests within MARGIN (relative) of their threshold. decisive = the outcome would flip:
    an accepted candidate on any such test, a rejected one when that test is its only failure."""
    rows = []
    for c in cands:
        failed = c["failed"].split(",") if c["failed"] else []
        for r in c["checks"]:
            if r["rel_distance"] <= MARGIN and r["test"] != "border":
                decisive = c["accepted"] or failed == [r["test"]]
                rows.append({"k": c["k"], "genus": c["genus"], "specimen_code": c["specimen_code"],
                             "kind": c["kind"], "tier": c["tier"], "candidate": c["candidate"], "area": c["area"],
                             "accepted": c["accepted"], **{k: r[k] for k in ("test", "value", "threshold", "op", "rel_distance")},
                             "decisive": decisive})
    cols = ["k", "genus", "specimen_code", "kind", "tier", "candidate", "area", "accepted", "test", "value",
            "threshold", "op", "rel_distance", "decisive"]
    return pd.DataFrame(rows, columns=cols).sort_values(["decisive", "rel_distance"], ascending=[False, True])


def run(selection: Path, mask_dir: Path, out: Path) -> pd.DataFrame:
    masks_out = out / "masks"
    masks_out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "25_clean_mask.log")
    sel = pd.read_csv(selection)
    f = font()
    cands, summary = [], []
    for r in sel.itertuples():
        with Image.open(ROOT / r.image_path) as im:
            rgb = np.array(im.convert("RGB"))
        alpha = np.array(Image.open(mask_dir / f"{r.specimen_code}_rembg_alpha.png")) / 255.0
        t0 = time.perf_counter()
        res = clean_image(rgb, alpha)
        t_rules = time.perf_counter() - t0
        mask, card, pin, keep, dropped, glue = (res[k] for k in ("mask", "card", "pin", "keep", "dropped", "glue"))
        contact = edge_contact(keep, res["scale"])
        log.info("%d %s %s: mask %d px; card candidates %d (accepted %d, %d px); pin candidates %d "
                 "(accepted %d, %d px); dropped %d px in %d pieces; pale blobs kept %d (%d px); "
                 "final touches frame: %s; %.2f s",
                 r.k, r.genus, r.specimen_code, mask.sum(), len(res["cards"]),
                 sum(c["accepted"] for c in res["cards"]), card.sum(), len(res["pins"]),
                 sum(p["accepted"] for p in res["pins"]), pin.sum(), dropped.sum(), len(components(dropped)),
                 sum(not c["accepted"] for c in res["cards"]), glue.sum(), contact or "no", t_rules)
        for i, c in enumerate(res["cards"] + res["pins"]):
            c.update({"k": r.k, "genus": r.genus, "specimen_code": r.specimen_code, "candidate": i})
            cands.append(c)
            if c["accepted"] or c["area"] >= 1000:
                log.info("   %s %s %s: %s", c["kind"], c["tier"], "ACCEPT" if c["accepted"] else "reject",
                         ", ".join(f"{k}={c[k]:.2f}" if isinstance(c[k], float) else f"{k}={c[k]}"
                                   for k in FEATURES + ["grown_px", "failed"] if k in c))
        Image.fromarray(card).save(masks_out / f"{r.specimen_code}_card.png")
        Image.fromarray(pin).save(masks_out / f"{r.specimen_code}_pin.png")
        Image.fromarray(keep).save(masks_out / f"{r.specimen_code}_clean.png")
        title = (f"{r.k}. {r.genus} {r.specimen_code} ({r.image_source}, {r.creator}): rembg mask {mask.sum():,} px; "
                 f"removed card {card.sum():,}, pin {pin.sum():,}, dropped pieces {dropped.sum():,} px; "
                 f"pale blobs kept {glue.sum():,} px. Rules {t_rules:.1f} s.")
        contact_sheet(rgb, res, title, f).save(out / f"{r.k}_{r.genus}_{r.specimen_code}.jpg", quality=90)
        summary.append({"k": r.k, "genus": r.genus, "specimen_code": r.specimen_code, "image_source": r.image_source,
                        "creator": r.creator, "mask_px": int(mask.sum()), "card_px": int(card.sum()),
                        "card_regions": sum(c["accepted"] for c in res["cards"]),
                        "card_grown_px": sum(c["grown_px"] for c in res["cards"] if c["accepted"]),
                        "pin_px": int(pin.sum()), "pin_regions": sum(p["accepted"] for p in res["pins"]),
                        "dropped_px": int(dropped.sum()), "dropped_pieces": len(components(dropped)),
                        "glue_px": int(glue.sum()), "glue_blobs": sum(not c["accepted"] for c in res["cards"]),
                        "clean_px": int(keep.sum()), "clean_components": n_components(keep),
                        "removed_share": round(1 - keep.sum() / mask.sum(), 4),
                        "final_touches_frame": contact, "rules_s": round(t_rules, 2)})
    cols = ["k", "genus", "specimen_code", "candidate", "kind", "tier"] + FEATURES + ["edge_needed", "grown_px",
                                                                                      "accepted", "failed"]
    df = pd.DataFrame([{k: c.get(k, "") for k in cols} for c in cands])
    for col in df.columns:
        if df[col].dtype == float:
            df[col] = df[col].round(3)
    df["true_label"] = ""
    df.to_csv(out / "candidates.csv", index=False)
    margins(cands).to_csv(out / "margins.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(out / "summary.csv", index=False)
    log.info("wrote %s: %d images, %d candidates", out, len(summ), len(df))
    return summ


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selection", type=Path, default=TRIAL / "selection.csv")
    ap.add_argument("--masks", type=Path, default=TRIAL / "masks")
    ap.add_argument("--out", type=Path, default=TRIAL / "clean")
    a = ap.parse_args()
    run(a.selection, a.masks, a.out)


if __name__ == "__main__":
    main()

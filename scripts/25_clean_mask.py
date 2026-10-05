#!/usr/bin/env python3
"""Second segmentation step: remove the card point and the pin from a rembg mask.

Follows the trial in reports/segmentation_trial/ (rembg is the primary
segmenter). Reads each rembg alpha in reports/segmentation_trial/masks/ for
the five trial images and writes to reports/segmentation_trial/clean/:

  <k>_<genus>_<specimen>.jpg   contact sheet: original | rembg mask | removed
                               regions (card yellow, pin red, dropped pieces
                               magenta, kept glue outlined cyan) | final ant
                               only on grey 128 | SAM 2 refinement and verdict
  masks/<specimen>_*.png       card, pin, clean and SAM 2 masks
  candidates.csv               every card and pin candidate with its features
                               and the rule that accepted or rejected it
  summary.csv                  pixels removed per image, refinement verdict
  25_clean_mask.log

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
           point has straight cut edges. Accepted components are grown by
           CARD_GROW px into neighbouring pale mask pixels. Pale blobs that
           fail the tests are "glue": kept and counted.
  pin      body = mask opened with a disc of radius BODY_RADIUS (the thick
           core: head, mesosoma, gaster). Thin parts = mask minus body,
           opened by THIN_OPEN px to drop hairs. A thin connected component
           is a pin when it is long (length >= PIN_MIN_LEN), not wider than
           PIN_MAX_WIDTH (width = area / length), elongated (length / width
           >= PIN_MIN_ELONG), straight (share of pixels within 0.75 width of
           the fitted axis >= PIN_MIN_STRAIGHT), of even width (coefficient
           of variation of the width along the axis <= PIN_MAX_WIDTH_CV),
           touches the image border (every pin leaves the frame) and is
           dark and grey (median luminance <= PIN_MAX_LUM, median chroma
           <= PIN_MAX_CHROMA). Only thin-part pixels are removed: where the
           pin passes behind or through the body, the body region is left
           untouched by construction.
  keep     after removal, the largest connected component plus every
           component reachable from it through gaps <= BRIDGE px (repeated
           dilation), so legs and antennae separated by a removed region
           stay. Everything else is dropped and counted.
  SAM 2    optional refinement (trial row C): sam2.1 tiny, box = padded box
           of the clean mask, up to 3 positive points at the thickest spots
           of the clean mask, one negative point at the thickest spot of
           every removed card or pin component. Kept only when IoU with the
           clean mask > SAM_MIN_IOU and it has fewer connected components
           (components >= 30 px). Set SAM2_SKIP=1 to skip it.
"""
from __future__ import annotations

import logging
import os
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
OUT = TRIAL / "clean"
MASKS = OUT / "masks"
CKPT = Path(os.environ.get("SAM2_CHECKPOINT",
                           ROOT.parent / "mg-ants-cache" / "sam2" / "sam2.1_hiera_tiny.pt"))
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

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
SAM_MIN_IOU = 0.85
SAM_POS_POINTS = 3
SAM_POS_SEP = 150
BOX_PAD = 0.02

GREY = 128
PANEL_W = 520
CAPTION_H = 24
CARD_COLOR, PIN_COLOR, DROP_COLOR, GLUE_COLOR = (237, 161, 0), (227, 73, 72), (232, 123, 164), (27, 175, 122)

log = logging.getLogger("clean_mask")


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


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


def card_candidates(rgb: np.ndarray, mask: np.ndarray, s: float) -> list[dict]:
    lum, chroma = luminance(rgb.astype(float)), rgb.max(-1).astype(int) - rgb.min(-1).astype(int)
    pale = mask & (lum > CARD_LUM) & (chroma < CARD_MAX_CHROMA)
    pale = ndimage.binary_opening(pale, structure=disc(2))
    h, w = mask.shape
    out = []
    for comp in components(pale, int(GLUE_MIN_AREA * s * s)):
        area = int(comp.sum())
        props = measure.regionprops(comp.astype(np.uint8))[0]
        radius = float(ndimage.distance_transform_edt(comp).max())
        edge = longest_edge(comp)
        bg_share = background_share(comp, mask)
        med_chroma = float(np.median(chroma[comp]))
        if med_chroma < CARD_WHITE_CHROMA:
            tier, need_edge = "white", WHITE_MIN_EDGE * s
            tests = {"area": area >= WHITE_MIN_AREA * h * w, "radius": radius >= WHITE_MIN_RADIUS * s,
                     "solidity": props.solidity >= WHITE_MIN_SOLIDITY, "edge": edge >= need_edge,
                     "background": bg_share >= CARD_MIN_BACKGROUND}
        else:
            tier, need_edge = "cream", max(CREAM_MIN_EDGE * s, CREAM_EDGE_FRAC * np.sqrt(area))
            tests = {"area": area >= CREAM_MIN_AREA * h * w, "radius": radius >= CREAM_MIN_RADIUS * s,
                     "solidity": props.solidity >= CREAM_MIN_SOLIDITY, "edge": edge >= need_edge,
                     "background": bg_share >= CARD_MIN_BACKGROUND}
        accepted = all(tests.values())
        region = comp
        if accepted and CARD_GROW:
            grow = ndimage.binary_dilation(comp, structure=disc(int(round(CARD_GROW * s))))
            region = comp | (grow & mask & (lum > CARD_LUM - 10) & (chroma < CARD_MAX_CHROMA + 15))
        out.append({"kind": "card", "mask": region, "tier": tier, "area": area, "area_frac": area / (h * w),
                    "radius": radius, "solidity": float(props.solidity), "edge": edge,
                    "edge_needed": need_edge, "background_share": bg_share, "median_lum": float(np.median(lum[comp])),
                    "median_chroma": med_chroma, "accepted": accepted,
                    "failed": ",".join(k for k, v in tests.items() if not v)})
    return out


def pin_candidates(rgb: np.ndarray, mask: np.ndarray, s: float) -> tuple[list[dict], np.ndarray]:
    lum, chroma = luminance(rgb.astype(float)), rgb.max(-1).astype(int) - rgb.min(-1).astype(int)
    body = ndimage.binary_opening(mask, structure=disc(int(round(BODY_RADIUS * s))))
    thin = ndimage.binary_opening(mask & ~body, structure=disc(int(round(THIN_OPEN * s))))
    h, w = mask.shape
    out = []
    for comp in components(thin, int(THIN_MIN_AREA * s * s)):
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
        tests = {"length": length >= PIN_MIN_LEN * s, "width": width <= PIN_MAX_WIDTH * s,
                 "elongation": length / width >= PIN_MIN_ELONG, "straight": straight >= PIN_MIN_STRAIGHT,
                 "even_width": width_cv <= PIN_MAX_WIDTH_CV, "border": border,
                 "lum": np.median(lum[comp]) <= PIN_MAX_LUM, "chroma": np.median(chroma[comp]) <= PIN_MAX_CHROMA}
        out.append({"kind": "pin", "mask": comp, "area": int(comp.sum()), "length": length, "width": width,
                    "elongation": length / width, "straight": straight, "width_cv": width_cv,
                    "touches_border": border,
                    "median_lum": float(np.median(lum[comp])), "median_chroma": float(np.median(chroma[comp])),
                    "accepted": all(tests.values()), "failed": ",".join(k for k, v in tests.items() if not v)})
    return out, body


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


def thickest_points(mask: np.ndarray, n: int, sep: float) -> list[tuple[int, int]]:
    dist = ndimage.distance_transform_edt(mask)
    pts: list[tuple[int, int]] = []
    for _ in range(n):
        if dist.max() <= 0:
            break
        y, x = np.unravel_index(int(dist.argmax()), dist.shape)
        pts.append((int(x), int(y)))
        yy, xx = np.ogrid[:dist.shape[0], :dist.shape[1]]
        dist[(xx - x) ** 2 + (yy - y) ** 2 < sep * sep] = 0
    return pts


def font(size: int = 15):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def panel(im: Image.Image, caption: str, f) -> Image.Image:
    im = im.convert("RGB")
    im = im.resize((PANEL_W, int(round(im.height * PANEL_W / im.width))), Image.LANCZOS)
    out = Image.new("RGB", (PANEL_W, im.height + CAPTION_H), "white")
    ImageDraw.Draw(out).text((5, 4), caption, fill="black", font=f)
    out.paste(im, (0, CAPTION_H))
    return out


def on_grey(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(mask[..., None], rgb, GREY).astype(np.uint8))


def removed_view(rgb, mask, card, pin, dropped, glue) -> Image.Image:
    out = rgb.astype(float) * 0.55 + 50
    out[~mask] = out[~mask] * 0.5
    for region, col in ((card, CARD_COLOR), (pin, PIN_COLOR), (dropped, DROP_COLOR)):
        out[region] = 0.25 * out[region] + 0.75 * np.array(col)
    edge = ndimage.binary_dilation(glue, iterations=2) & ~glue
    out[edge] = GLUE_COLOR
    return Image.fromarray(out.round().astype(np.uint8))


def sam_prompts_view(rgb, mask, box, pos, neg) -> Image.Image:
    im = on_grey(rgb, mask)
    d = ImageDraw.Draw(im)
    d.rectangle(box, outline=(235, 104, 52), width=4)
    for x, y in neg:
        d.line([(x - 10, y - 10), (x + 10, y + 10)], fill=PIN_COLOR, width=5)
        d.line([(x - 10, y + 10), (x + 10, y - 10)], fill=PIN_COLOR, width=5)
    for x, y in pos:
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=GLUE_COLOR, outline="white", width=2)
    return im


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MASKS.mkdir(exist_ok=True)
    setup_logging(OUT / "25_clean_mask.log")
    sel = pd.read_csv(TRIAL / "selection.csv")

    predictor = None
    if not os.environ.get("SAM2_SKIP"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        t0 = time.perf_counter()
        predictor = SAM2ImagePredictor(build_sam2(SAM2_CFG, str(CKPT), device="cpu"))
        log.info("SAM 2 load: %.1f s", time.perf_counter() - t0)

    f = font()
    cands, summary = [], []
    for r in sel.itertuples():
        with Image.open(ROOT / r.image_path) as im:
            rgb = np.array(im.convert("RGB"))
        h, w = rgb.shape[:2]
        s = w / 1024
        alpha = np.array(Image.open(TRIAL / "masks" / f"{r.specimen_code}_rembg_alpha.png")) / 255.0
        mask = alpha > ALPHA_FG
        t0 = time.perf_counter()

        cards = card_candidates(rgb, mask, s)
        card = np.zeros_like(mask)
        glue = np.zeros_like(mask)
        for c in cards:
            (card if c["accepted"] else glue)[c["mask"]] = True
        pins, body = pin_candidates(rgb, mask & ~card, s)
        pin = np.zeros_like(mask)
        for p in pins:
            if p["accepted"]:
                pin[p["mask"]] = True
        for c in cards + pins:
            cands.append({"k": r.k, "genus": r.genus, "specimen_code": r.specimen_code,
                          **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if k != "mask"}})
        cleaned = mask & ~card & ~pin
        keep, dropped = keep_connected(cleaned, s)
        t_rules = time.perf_counter() - t0
        log.info("%d %s %s: mask %d px; card candidates %d (accepted %d, %d px); pin candidates %d "
                 "(accepted %d, %d px); dropped %d px in %d pieces; glue blobs kept %d (%d px); %.2f s",
                 r.k, r.genus, r.specimen_code, mask.sum(), len(cards), sum(c["accepted"] for c in cards),
                 card.sum(), len(pins), sum(p["accepted"] for p in pins), pin.sum(), dropped.sum(),
                 len(components(dropped)), sum(not c["accepted"] for c in cards), glue.sum(), t_rules)
        for c in cards + pins:
            log.info("   %s %s: %s", c["kind"], "ACCEPT" if c["accepted"] else "reject",
                     ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in c.items() if k not in ("mask", "kind", "accepted")))

        Image.fromarray(card).save(MASKS / f"{r.specimen_code}_card.png")
        Image.fromarray(pin).save(MASKS / f"{r.specimen_code}_pin.png")
        Image.fromarray(keep).save(MASKS / f"{r.specimen_code}_clean.png")

        sam_mask, iou, verdict, t_sam, pos, neg, box = None, np.nan, "skipped", np.nan, [], [], None
        if predictor is not None:
            ys, xs = np.nonzero(keep)
            px, py = int(w * BOX_PAD), int(h * BOX_PAD)
            box = (max(0, xs.min() - px), max(0, ys.min() - py), min(w - 1, xs.max() + px), min(h - 1, ys.max() + py))
            pos = thickest_points(keep, SAM_POS_POINTS, SAM_POS_SEP * s)
            neg = [thickest_points(c["mask"], 1, 1)[0] for c in cards + pins if c["accepted"]]
            t0 = time.perf_counter()
            predictor.set_image(rgb)
            pts = pos + neg
            m, score, _ = predictor.predict(box=np.array(box), point_coords=np.array(pts, float),
                                            point_labels=np.array([1] * len(pos) + [0] * len(neg)),
                                            multimask_output=False)
            t_sam = time.perf_counter() - t0
            sam_mask = m[0].astype(bool)
            iou = float((sam_mask & keep).sum() / (sam_mask | keep).sum())
            fewer = n_components(sam_mask) < n_components(keep)
            verdict = "kept" if (iou > SAM_MIN_IOU and fewer) else "not kept"
            Image.fromarray(sam_mask).save(MASKS / f"{r.specimen_code}_sam2.png")
            log.info("   SAM 2: score %.2f, IoU with clean %.3f, components %d vs %d, %.2f s -> %s",
                     score[0], iou, n_components(sam_mask), n_components(keep), t_sam, verdict)

        final = sam_mask if verdict == "kept" else keep
        Image.fromarray(final).save(MASKS / f"{r.specimen_code}_final.png")
        title = (f"{r.k}. {r.genus} {r.specimen_code}: rembg mask {mask.sum():,} px; removed card {card.sum():,}, "
                 f"pin {pin.sum():,}, dropped pieces {dropped.sum():,} px; glue kept {glue.sum():,} px. "
                 f"Rules {t_rules:.2f} s.")
        panels = [panel(Image.fromarray(rgb), "original", f),
                  panel(Image.fromarray(mask), f"rembg mask, alpha > {ALPHA_FG}", f),
                  panel(removed_view(rgb, mask, card, pin, dropped, glue),
                        "removed: card yellow, pin red, dropped pieces magenta; glue kept, outlined green", f),
                  panel(on_grey(rgb, keep), f"clean ant only on grey {GREY}, {keep.sum():,} px, "
                                            f"{n_components(keep)} component(s)", f)]
        if sam_mask is not None:
            panels.append(panel(sam_prompts_view(rgb, sam_mask, box, pos, neg),
                                f"SAM 2 refinement: IoU {iou:.3f}, {n_components(sam_mask)} comp. -> {verdict}", f))
        ph = max(p.height for p in panels)
        sheet = Image.new("RGB", (len(panels) * PANEL_W + (len(panels) - 1) * 6, ph + 30), "white")
        ImageDraw.Draw(sheet).text((5, 7), title, fill="black", font=f)
        for i, p in enumerate(panels):
            sheet.paste(p, (i * (PANEL_W + 6), 30))
        sheet.save(OUT / f"{r.k}_{r.genus}_{r.specimen_code}.jpg", quality=90)

        summary.append({"k": r.k, "genus": r.genus, "specimen_code": r.specimen_code, "mask_px": int(mask.sum()),
                        "card_px": int(card.sum()), "pin_px": int(pin.sum()), "dropped_px": int(dropped.sum()),
                        "dropped_pieces": len(components(dropped)), "glue_px": int(glue.sum()),
                        "glue_blobs": sum(not c["accepted"] for c in cards),
                        "clean_px": int(keep.sum()), "clean_components": n_components(keep),
                        "removed_share": round(1 - keep.sum() / mask.sum(), 4), "rules_s": round(t_rules, 3),
                        "sam2_iou": round(iou, 3) if sam_mask is not None else "",
                        "sam2_components": n_components(sam_mask) if sam_mask is not None else "",
                        "sam2_s": round(t_sam, 2) if sam_mask is not None else "", "sam2_verdict": verdict,
                        "final": "sam2" if verdict == "kept" else "rules"})
    pd.DataFrame(cands).to_csv(OUT / "candidates.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()

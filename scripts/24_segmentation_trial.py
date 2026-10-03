#!/usr/bin/env python3
"""Segmentation trial on five images: SAM 2 (box prompt) against rembg.

A dry run before segmenting the whole dataset. Picks five images from
data/dataset.csv with seed 7, from five different genera, always including
Camponotus (a large ant) and one of Strumigenys / Syllophopsis (small ants),
and writes to reports/segmentation_trial/:

  <k>_<genus>_<specimen>.jpg   contact sheet, one row per prompt version:
                               original | SAM 2 mask overlay | SAM 2 ant only
                               on grey | rembg ant only on grey (row A);
                               prompts | overlay | ant only | rembg alpha
                               matte (row B); prompts | overlay | ant only |
                               where SAM 2 and rembg disagree (row C)
  masks/<specimen>_*.png       the binary masks (SAM 2) and alpha (rembg)
  selection.csv                the five images and how they were drawn
  runtime.csv                  per-image timings, boxes, scores, mask areas
  24_segmentation_trial.log

SAM 2: sam2.1_hiera_tiny (the smallest checkpoint, 39 M parameters), CPU,
multimask_output=False. Three prompt versions per image:

  A  automatic box only: the bounding box of the "foreground" pixels, those
     darker than DARK_REL times the background luminance or more saturated
     than CHROMA_MIN, after dropping specks and excluding the scale bar (the
     bottom-right corner plus the bar found by scripts/20_image_stats.py and
     its label). The saturation test is there because pale ants
     (Syllophopsis, Pheidole) are not dark: the dark-pixel rule alone boxes
     the pin instead. Both boxes are logged.
  B  the same box plus points placed by hand after looking at A: negative
     points on the pin and the card point where A had leaked onto them, and
     one positive point on the mesosoma. The positive point is not in the
     brief; it was added because with a near-full-frame box SAM 2 returned
     the background instead of the ant, and negative points alone cannot
     undo that. The negatives-only version (B0) is run and recorded too
     (mask file and runtime.csv columns) but not drawn.
  C  automatic alternative: box and one positive point derived from the
     rembg matte (box of alpha > 0.5, point at the thickest part of that
     mask), no hand input.

rembg: the default u2net model through onnxruntime on CPU, no prompt.

Timings are wall-clock seconds after one warm-up call per model, on the
machine's CPU; model load times are logged separately.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "segmentation_trial"
MASKS = OUT / "masks"
CKPT = Path(os.environ.get("SAM2_CHECKPOINT",
                           ROOT.parent / "mg-ants-cache" / "sam2" / "sam2.1_hiera_tiny.pt"))
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
REMBG_MODEL = "u2net"

SEED = 7
N_IMAGES = 5
LARGE = "Camponotus"
SMALL = ("Strumigenys", "Syllophopsis")

BORDER = 0.05        # border region for the background colour, as in 20_image_stats.py
DARK_REL = 0.75      # foreground if luminance < DARK_REL * background luminance
CHROMA_MIN = 40      # or if max(RGB) - min(RGB) > CHROMA_MIN (pale yellow ants)
MIN_CC = 0.0005      # drop connected components smaller than this share of the image
BOX_PAD = 0.02       # pad the box by this share of width / height
CORNER = (0.75, 0.85)  # exclude x > 75 % of width and y > 85 % of height (scale bar corner)
BAR_MARGIN = (20, 45, 12)  # around the detected bar: sideways, above (label), below
ALPHA_FG = 0.5       # rembg alpha above this counts as ant

GREY = (128, 128, 128)
PANEL_W = 640
CAPTION_H = 26
MASK_TINT = (42, 120, 214)
BOX_COLOR = (235, 104, 52)
NEG_COLOR = (227, 73, 72)
POS_COLOR = (27, 175, 122)

# version B points, (x, y) in original pixels, placed by hand after version A
NEG_POINTS: dict[str, list[tuple[int, int]]] = {
    "casent0102441": [(730, 60), (560, 530), (700, 600)],   # pin top; card point, twice
    "casent0044971": [(430, 60), (380, 720)],               # blurred pin, top and bottom
    "casent0005967": [(770, 40), (630, 740), (615, 335)],   # pin top, pin bottom, card point
    "casent0101703": [(870, 330), (380, 620)],              # glue on the card, twice
    "casent0101624": [(850, 450), (980, 60)],               # card point, blurred pin
}
POS_POINTS: dict[str, list[tuple[int, int]]] = {           # mesosoma
    "casent0102441": [(420, 230)],
    "casent0044971": [(820, 200)],
    "casent0005967": [(470, 220)],
    "casent0101703": [(300, 300)],
    "casent0101624": [(430, 290)],
}

log = logging.getLogger("segmentation_trial")


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def select_images(d: pd.DataFrame) -> pd.DataFrame:
    """Seeded draw: Camponotus, one small genus, then 3 more distinct genera."""
    rng = np.random.default_rng(SEED)
    small = SMALL[rng.integers(len(SMALL))]
    others = [g for g in sorted(d["genus"].unique()) if g not in (LARGE, small)]
    genera = [LARGE, small] + list(rng.choice(others, N_IMAGES - 2, replace=False))
    rows = []
    for k, g in enumerate(genera, 1):
        sub = d[d["genus"] == g].sort_values("image_path").reset_index(drop=True)
        r = sub.iloc[rng.integers(len(sub))].copy()
        r["k"] = k
        r["why"] = {LARGE: "large ant, required", small: "small ant, required"}.get(g, "random genus")
        rows.append(r)
    return pd.DataFrame(rows)


def padded_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if not mask.any():
        return None
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    px, py = int(w * BOX_PAD), int(h * BOX_PAD)
    return (int(max(0, xs.min() - px)), int(max(0, ys.min() - py)),
            int(min(w - 1, xs.max() + px)), int(min(h - 1, ys.max() + py)))


def foreground_box(rgb: np.ndarray, bar: dict | None) -> dict:
    """Version A: automatic box from the non-background pixels; also the dark-only box."""
    h, w = rgb.shape[:2]
    bw, bh = int(round(w * BORDER)), int(round(h * BORDER))
    border = np.zeros((h, w), bool)
    border[:bh], border[-bh:], border[:, :bw], border[:, -bw:] = True, True, True, True
    bg = np.median(rgb[border], axis=0)
    lum = luminance(rgb.astype(float))
    dark = lum < DARK_REL * luminance(bg)
    chroma = rgb.max(-1).astype(int) - rgb.min(-1).astype(int)
    exclude = np.zeros((h, w), bool)
    exclude[int(h * CORNER[1]):, int(w * CORNER[0]):] = True
    if bar is not None:
        half = (bar["px"] / 2 if np.isfinite(bar["px"]) else 60) + BAR_MARGIN[0]
        x0, x1 = int(max(0, bar["x"] - half)), int(min(w, bar["x"] + half))
        y0, y1 = int(max(0, bar["y"] - BAR_MARGIN[1])), int(min(h, bar["y"] + BAR_MARGIN[2]))
        exclude[y0:y1, x0:x1] = True

    def clean(mask: np.ndarray) -> np.ndarray:
        m = ndimage.binary_opening(mask & ~exclude, iterations=2)
        lab, n = ndimage.label(m)
        if n == 0:
            return m
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        return np.isin(lab, 1 + np.flatnonzero(sizes >= MIN_CC * h * w))

    return {"background_rgb": tuple(int(v) for v in bg),
            "box": padded_box(clean(dark | (chroma > CHROMA_MIN))), "box_dark_only": padded_box(clean(dark))}


def rembg_prompt(alpha: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Version C: box of the rembg matte and its thickest point (largest inscribed disc)."""
    fg = alpha > ALPHA_FG
    dist = ndimage.distance_transform_edt(fg)
    y, x = np.unravel_index(int(dist.argmax()), dist.shape)
    return padded_box(fg), (int(x), int(y))


def font(size: int = 16):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def draw_prompts(d: ImageDraw.ImageDraw, box, pos=(), neg=()) -> None:
    if box is not None:
        d.rectangle(box, outline=BOX_COLOR, width=4)
    for x, y in neg:
        d.line([(x - 12, y - 12), (x + 12, y + 12)], fill=NEG_COLOR, width=5)
        d.line([(x - 12, y + 12), (x + 12, y - 12)], fill=NEG_COLOR, width=5)
        d.ellipse([x - 16, y - 16, x + 16, y + 16], outline="white", width=2)
    for x, y in pos:
        d.ellipse([x - 11, y - 11, x + 11, y + 11], fill=POS_COLOR, outline="white", width=3)


def overlay(rgb: np.ndarray, mask: np.ndarray, box=None, pos=(), neg=()) -> Image.Image:
    out = rgb.astype(float)
    out[mask] = 0.5 * out[mask] + 0.5 * np.array(MASK_TINT)
    edge = mask ^ ndimage.binary_erosion(mask, iterations=2)
    out[edge] = MASK_TINT
    im = Image.fromarray(out.round().astype(np.uint8))
    draw_prompts(ImageDraw.Draw(im), box, pos, neg)
    return im


def disagreement(rgb: np.ndarray, sam: np.ndarray, rembg_fg: np.ndarray) -> Image.Image:
    """Grey image; blue = SAM 2 only, orange = rembg only, untinted = both agree."""
    out = np.repeat(luminance(rgb.astype(float))[..., None], 3, axis=-1) * 0.6 + 60
    out[sam & ~rembg_fg] = MASK_TINT
    out[rembg_fg & ~sam] = BOX_COLOR
    return Image.fromarray(out.round().astype(np.uint8))


def on_grey(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    """alpha in [0, 1]: ant pixels over a uniform mid grey."""
    a = alpha[..., None].astype(float)
    out = a * rgb + (1 - a) * np.array(GREY, float)
    return Image.fromarray(out.round().astype(np.uint8))


def panel(im: Image.Image, caption: str, f) -> Image.Image:
    im = im.convert("RGB")
    im = im.resize((PANEL_W, int(round(im.height * PANEL_W / im.width))), Image.LANCZOS)
    out = Image.new("RGB", (PANEL_W, im.height + CAPTION_H), "white")
    ImageDraw.Draw(out).text((6, 5), caption, fill="black", font=f)
    out.paste(im, (0, CAPTION_H))
    return out


def sheet(rows: list[list[Image.Image]], title: str, f) -> Image.Image:
    gap = 6
    ph = max(p.height for r in rows for p in r)
    out = Image.new("RGB", (4 * PANEL_W + 3 * gap, 34 + len(rows) * (ph + gap)), "white")
    ImageDraw.Draw(out).text((6, 8), title, fill="black", font=f)
    for i, r in enumerate(rows):
        for j, p in enumerate(r):
            out.paste(p, (j * (PANEL_W + gap), 34 + i * (ph + gap)))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MASKS.mkdir(exist_ok=True)
    setup_logging(OUT / "24_segmentation_trial.log")
    import torch
    from rembg import new_session, remove
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    log.info("torch %s, %d threads, cpu only", torch.__version__, torch.get_num_threads())

    d = pd.read_csv(ROOT / "data" / "dataset.csv")
    pick = select_images(d)
    pick[["k", "genus", "subfamily", "specimen_code", "view", "image_path", "image_source",
          "creator", "split", "why"]].to_csv(OUT / "selection.csv", index=False)
    log.info("selected (seed %d):\n%s", SEED,
             pick[["k", "genus", "specimen_code", "split", "why"]].to_string(index=False))
    stats = pd.read_csv(ROOT / "reports" / "confounds" / "image_stats.csv").set_index("image_path")

    t0 = time.perf_counter()
    predictor = SAM2ImagePredictor(build_sam2(SAM2_CFG, str(CKPT), device="cpu"))
    log.info("SAM 2 load (%s): %.1f s", CKPT.name, time.perf_counter() - t0)
    t0 = time.perf_counter()
    session = new_session(REMBG_MODEL)
    log.info("rembg session (%s): %.1f s", REMBG_MODEL, time.perf_counter() - t0)
    warm = np.full((256, 256, 3), 200, np.uint8)
    predictor.set_image(warm)
    predictor.predict(box=np.array([20, 20, 200, 200]), multimask_output=False)
    remove(Image.fromarray(warm), session=session)

    def sam(box, pos=(), neg=()) -> tuple[np.ndarray, float, float]:
        pts = list(pos) + list(neg)
        kw = {}
        if pts:
            kw = {"point_coords": np.array(pts, float), "point_labels": np.array([1] * len(pos) + [0] * len(neg))}
        t0 = time.perf_counter()
        m, score, _ = predictor.predict(box=np.array(box), multimask_output=False, **kw)
        return m[0].astype(bool), float(score[0]), time.perf_counter() - t0

    f = font()
    records = []
    for r in pick.itertuples():
        with Image.open(ROOT / r.image_path) as im:
            rgb = np.array(im.convert("RGB"))
        h, w = rgb.shape[:2]
        s = stats.loc[r.image_path]
        bar = None
        if s["scalebar_status"] in ("found", "truncated"):
            bar = {"x": s["scalebar_x_frac"] * w, "y": s["scalebar_y_frac"] * h, "px": s["scalebar_px"]}
        fg = foreground_box(rgb, bar)
        box = fg["box"]
        log.info("%d %s %s: %dx%d, background %s, box A %s, dark-only box %s",
                 r.k, r.genus, r.specimen_code, w, h, fg["background_rgb"], box, fg["box_dark_only"])

        t0 = time.perf_counter()
        predictor.set_image(rgb)
        t_embed = time.perf_counter() - t0
        mask_a, score_a, t_a = sam(box)
        pos, neg = POS_POINTS.get(r.specimen_code, []), NEG_POINTS.get(r.specimen_code, [])
        mask_b0, score_b0, t_b0 = sam(box, [], neg)
        mask_b, score_b, t_b = sam(box, pos, neg)

        t0 = time.perf_counter()
        rgba = remove(Image.fromarray(rgb), session=session)
        t_rembg = time.perf_counter() - t0
        alpha = np.asarray(rgba)[..., 3] / 255.0
        rembg_fg = alpha > ALPHA_FG
        box_c, pos_c = rembg_prompt(alpha)
        mask_c, score_c, t_c = sam(box_c, [pos_c])

        Image.fromarray(mask_a).save(MASKS / f"{r.specimen_code}_sam2_A_box.png")
        Image.fromarray(mask_b0).save(MASKS / f"{r.specimen_code}_sam2_B0_box_negatives.png")
        Image.fromarray(mask_b).save(MASKS / f"{r.specimen_code}_sam2_B_box_points.png")
        Image.fromarray(mask_c).save(MASKS / f"{r.specimen_code}_sam2_C_rembg_box.png")
        Image.fromarray((alpha * 255).round().astype(np.uint8)).save(MASKS / f"{r.specimen_code}_rembg_alpha.png")

        title = (f"{r.k}. {r.genus} {r.specimen_code} ({w}x{h}, {r.image_source}). "
                 f"SAM 2 tiny, CPU: image embedding {t_embed:.2f} s, then {t_a:.2f} s per prompt. "
                 f"rembg {REMBG_MODEL}: {t_rembg:.2f} s. Grey = removed.")
        blank = np.zeros((h, w), bool)
        rows = [
            [panel(Image.fromarray(rgb), "original", f),
             panel(overlay(rgb, mask_a, box), f"A  SAM 2, automatic box (orange): mask, score {score_a:.2f}", f),
             panel(on_grey(rgb, mask_a), f"A  SAM 2 ant only on grey, {mask_a.mean():.1%} of image", f),
             panel(on_grey(rgb, alpha), f"rembg ant only on grey, {rembg_fg.mean():.1%} of image", f)],
            [panel(overlay(rgb, blank, box, pos, neg), "B  prompts: box, positive point (green), negatives (x)", f),
             panel(overlay(rgb, mask_b, box, pos, neg), f"B  SAM 2, box + points: mask, score {score_b:.2f}", f),
             panel(on_grey(rgb, mask_b), f"B  SAM 2 ant only on grey, {mask_b.mean():.1%} of image", f),
             panel(Image.fromarray((alpha * 255).astype(np.uint8)), "rembg alpha matte", f)],
            [panel(overlay(rgb, blank, box_c, [pos_c]), "C  prompts from rembg: box + thickest point, automatic", f),
             panel(overlay(rgb, mask_c, box_c, [pos_c]), f"C  SAM 2, rembg box + point: mask, score {score_c:.2f}", f),
             panel(on_grey(rgb, mask_c), f"C  SAM 2 ant only on grey, {mask_c.mean():.1%} of image", f),
             panel(disagreement(rgb, mask_c, rembg_fg), "C vs rembg: blue = SAM 2 only, orange = rembg only", f)],
        ]
        sheet(rows, title, f).save(OUT / f"{r.k}_{r.genus}_{r.specimen_code}.jpg", quality=90)

        inter, union = (mask_c & rembg_fg).sum(), (mask_c | rembg_fg).sum()
        records.append({"k": r.k, "genus": r.genus, "specimen_code": r.specimen_code, "width": w, "height": h,
                        "background_rgb": "%d,%d,%d" % fg["background_rgb"],
                        "box_A": "%d,%d,%d,%d" % box, "box_dark_only": "%d,%d,%d,%d" % fg["box_dark_only"],
                        "box_C": "%d,%d,%d,%d" % box_c, "point_C": "%d,%d" % pos_c,
                        "pos_points_B": ";".join(f"{x},{y}" for x, y in pos),
                        "neg_points_B": ";".join(f"{x},{y}" for x, y in neg),
                        "sam2_embed_s": round(t_embed, 3),
                        "sam2_A_predict_s": round(t_a, 3), "sam2_B0_predict_s": round(t_b0, 3),
                        "sam2_B_predict_s": round(t_b, 3),
                        "sam2_C_predict_s": round(t_c, 3), "rembg_s": round(t_rembg, 3),
                        "sam2_A_score": round(score_a, 3), "sam2_B0_score": round(score_b0, 3),
                        "sam2_B_score": round(score_b, 3),
                        "sam2_C_score": round(score_c, 3),
                        "sam2_A_frac": round(float(mask_a.mean()), 4),
                        "sam2_B0_frac": round(float(mask_b0.mean()), 4),
                        "sam2_B_frac": round(float(mask_b.mean()), 4),
                        "sam2_C_frac": round(float(mask_c.mean()), 4), "rembg_frac": round(float(rembg_fg.mean()), 4),
                        "iou_C_rembg": round(float(inter / union), 3) if union else ""})
        log.info("   embed %.2f s; A %.2f s score %.2f %.1f %%; B0 score %.2f %.1f %%; "
                 "B %.2f s score %.2f %.1f %%; rembg %.2f s %.1f %%; C %.2f s score %.2f %.1f %%, "
                 "IoU with rembg %.2f",
                 t_embed, t_a, score_a, 100 * mask_a.mean(), score_b0, 100 * mask_b0.mean(),
                 t_b, score_b, 100 * mask_b.mean(),
                 t_rembg, 100 * rembg_fg.mean(), t_c, score_c, 100 * mask_c.mean(), inter / union)
    out = pd.DataFrame(records)
    out.to_csv(OUT / "runtime.csv", index=False)
    log.info("mean per image: SAM 2 embedding %.2f s + %.2f s per prompt; rembg %.2f s",
             out["sam2_embed_s"].mean(), out["sam2_A_predict_s"].mean(), out["rembg_s"].mean())


if __name__ == "__main__":
    main()

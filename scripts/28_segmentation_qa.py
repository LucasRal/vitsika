#!/usr/bin/env python3
"""QA sheets for the full rembg segmentation (scripts/27_segment_all.py).

Reads data/segmented/manifest.csv and writes to reports/segmentation/qa/:

  <genus>_<n>.jpg      20 images per sheet, by genus in specimen order:
                       original | ant only on grey, with the specimen code,
                       mask share, component count and flags in the caption
  worst40.jpg          the 40 images with the highest automatic score
                       (10 for each of the flags empty, large, frame_filling;
                       10 for a mask under 5 % of the image ("small", added
                       here, not a manifest flag); plus card residue fraction,
                       pin residue fraction and components over 30 px / 50)
                       with the score in the caption
  handcheck_<n>.jpg    60 random images (seed 42), 20 per sheet, numbered for
                       the hand check; the template reports/segmentation/
                       handcheck.csv gets one row per image with an empty
                       rating column (good / partial / bad) to fill by hand
  worst40.csv          the scored list

Reads only; nothing in data/ is modified.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEG = ROOT / "data" / "segmented"
REPORT = ROOT / "reports" / "segmentation"
QA = REPORT / "qa"
PER_SHEET, COLS = 20, 4
TILE_W = 256                 # width of each of the two panels in a pair
CAP_H = 30
HAND_N, HAND_SEED = 60, 42
WORST_N = 40
SMALL_FRAC = 0.05            # mask share below which the QA score treats the mask as small

log = logging.getLogger("segmentation_qa")


def font(size: int = 13):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def pair(r, caption: str, f) -> Image.Image:
    stem = Path(r.image_path).stem
    orig = Image.open(ROOT / r.image_path).convert("RGB")
    ant = Image.open(SEG / r.genus / f"{stem}_ant_grey.png").convert("RGB")
    h = int(round(orig.height * TILE_W / orig.width))
    out = Image.new("RGB", (2 * TILE_W + 4, h + CAP_H), "white")
    d = ImageDraw.Draw(out)
    d.text((3, 2), caption[:60], fill="black", font=f)
    if len(caption) > 60:
        d.text((3, 15), caption[60:120], fill="black", font=f)
    out.paste(orig.resize((TILE_W, h), Image.LANCZOS), (0, CAP_H))
    out.paste(ant.resize((TILE_W, h), Image.LANCZOS), (TILE_W + 4, CAP_H))
    return out


def sheet(rows: pd.DataFrame, captions: list[str], title: str, path: Path, f) -> None:
    tiles = [pair(r, c, f) for r, c in zip(rows.itertuples(), captions)]
    tw = 2 * TILE_W + 4
    th = max(t.height for t in tiles)
    n_rows = int(np.ceil(len(tiles) / COLS))
    out = Image.new("RGB", (COLS * (tw + 8), 26 + n_rows * (th + 8)), "white")
    ImageDraw.Draw(out).text((6, 6), title, fill="black", font=font(15))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % COLS) * (tw + 8), 26 + (i // COLS) * (th + 8)))
    out.save(path, quality=80)


def caption(r) -> str:
    fl = r.flags if isinstance(r.flags, str) else ""
    return (f"{r.specimen_code} mask {100 * r.mask_area_fraction:.0f}% comp {r.n_components_30px} "
            f"card {100 * r.card_residue_frac:.0f}% pin {100 * r.pin_residue_frac:.0f}%"
            + (f" [{fl}]" if fl else ""))


def main() -> None:
    QA.mkdir(parents=True, exist_ok=True)
    setup_logging(REPORT / "28_segmentation_qa.log")
    m = pd.read_csv(SEG / "manifest.csv", keep_default_na=False)
    m["flags"] = m["flags"].astype(str)
    small = m["mask_area_fraction"] < SMALL_FRAC
    m.loc[small, "flags"] = (m.loc[small, "flags"] + ";small").str.strip(";")
    log.info("%d masks under %.0f %% of the image, tagged small", small.sum(), 100 * SMALL_FRAC)
    f = font()
    n_sheets = 0
    for genus, g in m.sort_values(["genus", "specimen_code"]).groupby("genus", sort=True):
        for i in range(0, len(g), PER_SHEET):
            chunk = g.iloc[i:i + PER_SHEET]
            k = i // PER_SHEET + 1
            sheet(chunk, [caption(r) for r in chunk.itertuples()],
                  f"{genus}, sheet {k}: original | rembg ant only on grey. Caption: mask share of the image, "
                  f"components over 30 px, card and pin residue shares of the mask, flags.",
                  QA / f"{genus}_{k}.jpg", f)
            n_sheets += 1
    log.info("%d genus sheets", n_sheets)

    score = (10 * m["flags"].str.contains("empty|large|frame_filling|small").astype(int)
             + m["card_residue_frac"] + m["pin_residue_frac"] + np.minimum(m["n_components_30px"], 10) / 50)
    m["qa_score"] = score.round(4)
    worst = m.sort_values(["qa_score", "specimen_code"], ascending=[False, True]).head(WORST_N)
    worst[["specimen_code", "genus", "image_path", "qa_score", "mask_area_fraction", "n_components_30px",
           "card_residue_frac", "pin_residue_frac", "flags"]].to_csv(QA / "worst40.csv", index=False)
    sheet(worst, [f"#{i + 1} s={r.qa_score:.2f} " + caption(r) for i, r in enumerate(worst.itertuples())],
          f"The {WORST_N} highest automatic scores: 10 per flag (empty, large, frame_filling, small) + card residue "
          f"+ pin residue + min(components, 10) / 50.", QA / "worst40.jpg", f)
    log.info("worst %d: scores %.2f to %.2f", WORST_N, worst.qa_score.max(), worst.qa_score.min())

    rng = np.random.default_rng(HAND_SEED)
    srt = m.sort_values("image_path").reset_index(drop=True)
    pick = srt.iloc[np.sort(rng.choice(len(srt), HAND_N, replace=False))].reset_index(drop=True)
    pick.insert(0, "k", range(1, len(pick) + 1))
    for i in range(0, HAND_N, PER_SHEET):
        chunk = pick.iloc[i:i + PER_SHEET]
        sheet(chunk, [f"#{r.k} " + caption(r) for r in chunk.itertuples()],
              f"Hand check, {HAND_N} random images, seed {HAND_SEED}, sheet {i // PER_SHEET + 1}: "
              "rate good / partial (ant parts missing) / bad (ant mostly missing or mask inverted).",
              QA / f"handcheck_{i // PER_SHEET + 1}.jpg", f)
    tmpl = pick[["k", "specimen_code", "genus", "split", "image_path", "mask_area_fraction", "n_components_30px",
                 "card_residue_frac", "pin_residue_frac", "flags"]].copy()
    tmpl["rating"] = ""
    tmpl["note"] = ""
    out = REPORT / "handcheck.csv"
    if out.exists():
        prev = pd.read_csv(out, keep_default_na=False)
        if list(prev["specimen_code"]) == list(tmpl["specimen_code"]):
            tmpl["rating"], tmpl["note"] = prev["rating"].values, prev["note"].values
            log.info("kept the %d ratings already in %s", (prev["rating"] != "").sum(), out)
    tmpl.to_csv(out, index=False)
    log.info("hand-check template: %s", out)


if __name__ == "__main__":
    main()

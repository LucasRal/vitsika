#!/usr/bin/env python3
"""Measurement run of rembg + mask cleaning on 30 images.

Draws 30 images from data/dataset.csv with seed 11 so that every genus
(27) is represented, at least 3 are GBIF-cache crops and at least 2 are by
photographers other than April Nobile; runs rembg (u2net, CPU) on each;
then runs scripts/25_clean_mask.py's rules with the current thresholds.
Writes to reports/segmentation_trial/run30/:

  selection.csv                 the 30 images and why each was drawn
  masks/<specimen>_rembg_alpha.png   rembg alpha
  <k>_<genus>_<specimen>.jpg    contact sheets (25_clean_mask.py)
  masks/<specimen>_{card,pin,clean}.png, candidates.csv (with an empty
  true_label column to fill by hand), margins.csv, summary.csv
  summary_sheet.jpg             the 30 final ant-only results as thumbnails
  26_segmentation_run30.log, 25_clean_mask.log

Draw: 3 GBIF-cache images from 3 distinct genera; 2 Commons images by other
photographers from 2 further genera; one random image for each remaining
genus; 3 more images at random from the rest. All with one generator,
seed 11. Offline except that rembg needs its model file (already cached).
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "segmentation_trial" / "run30"
MASKS = OUT / "masks"
SEED = 11
N_IMAGES = 30
N_GBIF = 3
N_OTHER_PHOTOGRAPHER = 2
MAIN_PHOTOGRAPHER = "April Nobile"
REMBG_MODEL = "u2net"
THUMB_W, THUMB_COLS = 420, 6

log = logging.getLogger("run30")


def load_clean_mask():
    spec = importlib.util.spec_from_file_location("clean_mask", Path(__file__).resolve().parent / "25_clean_mask.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_images(d: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    d = d.sort_values("image_path").reset_index(drop=True)
    genera = sorted(d["genus"].unique())
    picked: list[tuple[int, str]] = []
    used_genera: set[str] = set()

    def draw(pool: pd.DataFrame, why: str) -> None:
        i = int(pool.index[rng.integers(len(pool))])
        picked.append((i, why))
        used_genera.add(d.at[i, "genus"])

    gbif = d[d["image_source"] == "gbif_cache"]
    for _ in range(N_GBIF):
        pool = gbif[~gbif["genus"].isin(used_genera)]
        draw(pool, "gbif_cache crop, required")
    other = d[(d["image_source"] == "commons") & (d["creator"] != MAIN_PHOTOGRAPHER)]
    for _ in range(N_OTHER_PHOTOGRAPHER):
        pool = other[~other["genus"].isin(used_genera)]
        draw(pool, "other photographer, required")
    for g in genera:
        if g not in used_genera:
            draw(d[d["genus"] == g], "one per genus")
    taken = {i for i, _ in picked}
    for _ in range(N_IMAGES - len(picked)):
        pool = d[~d.index.isin(taken)]
        draw(pool, "extra, random")
        taken = {i for i, _ in picked}
    rows = d.loc[[i for i, _ in picked]].copy()
    rows["why"] = [w for _, w in picked]
    rows = rows.sort_values(["genus", "specimen_code"]).reset_index(drop=True)
    rows.insert(0, "k", range(1, len(rows) + 1))
    return rows


def summary_sheet(sel: pd.DataFrame, summ: pd.DataFrame, cm) -> None:
    f = cm.font(13)
    tiles = []
    for r, s in zip(sel.itertuples(), summ.itertuples()):
        with Image.open(ROOT / r.image_path) as im:
            rgb = np.array(im.convert("RGB"))
        keep = np.array(Image.open(MASKS / f"{r.specimen_code}_clean.png")).astype(bool)
        tile = cm.on_grey(rgb, keep).resize((THUMB_W, int(THUMB_W * rgb.shape[0] / rgb.shape[1])), Image.LANCZOS)
        cap = (f"{r.k}. {r.genus} {r.specimen_code}{' [gbif]' if r.image_source == 'gbif_cache' else ''}"
               f"{'' if r.creator == MAIN_PHOTOGRAPHER else ' [' + r.creator + ']'}\n"
               f"card {s.card_px:,} px, pin {s.pin_px:,} px, dropped {s.dropped_px:,} px")
        out = Image.new("RGB", (THUMB_W, tile.height + 34), "white")
        ImageDraw.Draw(out).text((4, 2), cap, fill="black", font=f)
        out.paste(tile, (0, 34))
        tiles.append(out)
    th = max(t.height for t in tiles)
    rows = int(np.ceil(len(tiles) / THUMB_COLS))
    sheet = Image.new("RGB", (THUMB_COLS * (THUMB_W + 6), 28 + rows * (th + 6)), "white")
    ImageDraw.Draw(sheet).text((6, 6), f"rembg + 25_clean_mask.py, {len(tiles)} images, seed {SEED}: final ant only on grey. "
                                       "Card and pin pixels removed in each caption.", fill="black", font=cm.font(16))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % THUMB_COLS) * (THUMB_W + 6), 28 + (i // THUMB_COLS) * (th + 6)))
    sheet.save(OUT / "summary_sheet.jpg", quality=88)


def main() -> None:
    MASKS.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT / "26_segmentation_run30.log")
    d = pd.read_csv(ROOT / "data" / "dataset.csv")
    sel = select_images(d)
    cols = ["k", "genus", "subfamily", "specimen_code", "view", "image_path", "image_source", "creator", "split", "why"]
    sel[cols].to_csv(OUT / "selection.csv", index=False)
    log.info("selected %d images, %d genera, %d gbif_cache, %d other photographers:\n%s", len(sel),
             sel["genus"].nunique(), (sel["image_source"] == "gbif_cache").sum(),
             (sel["creator"] != MAIN_PHOTOGRAPHER).sum(),
             sel[["k", "genus", "specimen_code", "image_source", "creator", "why"]].to_string(index=False))

    from rembg import new_session, remove
    session = new_session(REMBG_MODEL)
    times = []
    for r in sel.itertuples():
        with Image.open(ROOT / r.image_path) as im:
            rgb = im.convert("RGB")
        t0 = time.perf_counter()
        rgba = remove(rgb, session=session)
        times.append(time.perf_counter() - t0)
        Image.fromarray(np.asarray(rgba)[..., 3]).save(MASKS / f"{r.specimen_code}_rembg_alpha.png")
    log.info("rembg %s: %.2f s per image (min %.2f, max %.2f)", REMBG_MODEL, np.mean(times), min(times), max(times))

    cm = load_clean_mask()
    summ = cm.run(OUT / "selection.csv", MASKS, OUT)
    summary_sheet(sel, summ, cm)
    log.info("wrote %s", OUT / "summary_sheet.jpg")


if __name__ == "__main__":
    main()

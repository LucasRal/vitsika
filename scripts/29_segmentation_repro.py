#!/usr/bin/env python3
"""Compare two rembg segmentation runs mask by mask (cross-machine reproducibility).

Reads the first run's masks and manifest from data/segmented_run1/ (copied
before the re-run: 500 masks made on the VPS, the first 500 manifest rows in
image_path order, then 736 made on the Mac by a --resume run) and the
current run's from data/segmented/ (all 1,236 made on the Mac by
scripts/27_segment_all.py). Writes to reports/segmentation/:

  29_segmentation_repro.csv   one row per image: specimen_code, genus,
                              run1_machine (vps / mac), mask_px in each run,
                              intersection over union (1 when both masks are
                              empty), changed-pixel share (pixels that differ
                              / image pixels), identical (bool)
  29_segmentation_repro.log   distribution per machine and the worst five

Reads only; nothing in data/ is modified. Network: none.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN1 = ROOT / "data" / "segmented_run1"
RUN2 = ROOT / "data" / "segmented"
REPORT = ROOT / "reports" / "segmentation"
N_VPS = 500                  # the first 500 rows of the run-1 manifest were made on the VPS

log = logging.getLogger("segmentation_repro")


def load(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) >= 128


def main() -> None:
    setup_logging(REPORT / "29_segmentation_repro.log")
    m1 = pd.read_csv(RUN1 / "manifest.csv", keep_default_na=False)
    m2 = pd.read_csv(RUN2 / "manifest.csv", keep_default_na=False).set_index("image_path")
    rows = []
    for i, r in enumerate(m1.itertuples()):
        name = f"{Path(r.image_path).stem}_mask.png"
        a, b = load(RUN1 / r.genus / name), load(RUN2 / r.genus / name)
        inter, union, diff = int((a & b).sum()), int((a | b).sum()), int((a ^ b).sum())
        rows.append({"specimen_code": r.specimen_code, "genus": r.genus,
                     "run1_machine": "vps" if i < N_VPS else "mac",
                     "mask_px_run1": int(a.sum()), "mask_px_run2": int(b.sum()),
                     "iou": round(inter / union, 6) if union else 1.0,
                     "changed_share": round(diff / a.size, 6), "identical": diff == 0,
                     "flags_run1": r.flags, "flags_run2": m2.loc[r.image_path, "flags"]})
    d = pd.DataFrame(rows)
    d.to_csv(REPORT / "29_segmentation_repro.csv", index=False)

    log.info("%d images compared (run 1: %d VPS, %d Mac)", len(d), (d.run1_machine == "vps").sum(), (d.run1_machine == "mac").sum())
    q = [0, 0.01, 0.05, 0.5, 0.95, 1]
    for machine, g in d.groupby("run1_machine", sort=False):
        log.info("%s: %d images, identical %d, IoU < 0.99: %d, < 0.95: %d, < 0.90: %d",
                 machine, len(g), g.identical.sum(), (g.iou < 0.99).sum(), (g.iou < 0.95).sum(), (g.iou < 0.90).sum())
        log.info("  IoU quantiles %s: %s", q, ", ".join(f"{v:.4f}" for v in g.iou.quantile(q)))
        log.info("  changed-pixel share quantiles %s: %s", q, ", ".join(f"{v:.5f}" for v in g.changed_share.quantile(q)))
        log.info("  mean IoU %.4f, mean changed share %.5f", g.iou.mean(), g.changed_share.mean())
        flips = g[g.flags_run1 != g.flags_run2]
        log.info("  flags changed on %d images%s", len(flips),
                 "".join(f"\n    {f.genus} {f.specimen_code}: [{f.flags_run1}] -> [{f.flags_run2}]" for f in flips.itertuples()))
    log.info("worst five by IoU:")
    for r in d.nsmallest(5, "iou").itertuples():
        log.info("  %s %s (%s): IoU %.4f, changed %.4f, mask %d -> %d px",
                 r.genus, r.specimen_code, r.run1_machine, r.iou, r.changed_share, r.mask_px_run1, r.mask_px_run2)


if __name__ == "__main__":
    main()

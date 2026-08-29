#!/usr/bin/env python3
"""Phase 4: download the dataset images.

For each row of data/dataset.csv, fetch the image through the GBIF image
cache: first at image_max_px, then at original size ('x'). If both cache
attempts fail the row is marked download_failed; there is NO antweb.org
fallback (Cloudflare blocks it, and failures are expected to concentrate in
the newest records whose images GBIF has not cached yet).

Images are verified with Pillow and saved to
data/images/<genus>/<specimen_code>_<view>.jpg. Resumable: existing valid
files are skipped. Outputs: data/download_status.csv (per-row outcome),
data/download_errors.csv (failures only) and reports/download_stats.md
(failures broken down by eventDate year and by gbifID decile).
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import GbifClient, cache_url, load_config, setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMAGES = DATA / "images"
REPORTS = ROOT / "reports"

log = logging.getLogger("download")


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def fetch(client: GbifClient, url: str, dest: Path) -> str | None:
    """Download url to dest; return None on success, error string otherwise."""
    try:
        resp = client.get(url)
    except Exception as exc:
        return f"request error: {exc}"
    if resp.status_code != 200:
        return f"HTTP {resp.status_code}"
    if not resp.headers.get("Content-Type", "").startswith("image"):
        return f"not an image ({resp.headers.get('Content-Type')})"
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(resp.content)
    if not is_valid_image(tmp):
        tmp.unlink(missing_ok=True)
        return "Pillow validation failed"
    tmp.rename(dest)
    return None


def md_table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return lines


def write_download_stats(status: pd.DataFrame) -> None:
    lines = ["# Download stats", ""]
    n_failed = int(status["download_failed"].sum())
    lines.append(f"- Rows: **{len(status)}**, failed: **{n_failed}** "
                 f"({n_failed / len(status):.1%})")
    lines.append("")

    lines.append("## Failures by eventDate year")
    lines.append("")
    status["year"] = status["eventDate"].astype(str).str[:4]
    by_year = status.groupby("year").agg(rows=("download_failed", "size"),
                                         failed=("download_failed", "sum"))
    rows = [
        [year, r.rows, r.failed, f"{r.failed / r.rows:.1%}"]
        for year, r in by_year.iterrows()
    ]
    lines += md_table(["eventDate year", "rows", "failed", "fail rate"], rows)
    lines.append("")

    lines.append("## Failures by gbifID decile (proxy for record recency)")
    lines.append("")
    status["gbifID_decile"] = (
        pd.qcut(status["gbifID"].rank(method="first"), 10, labels=False) + 1
    )
    by_dec = status.groupby("gbifID_decile").agg(
        min_id=("gbifID", "min"), max_id=("gbifID", "max"),
        rows=("download_failed", "size"), failed=("download_failed", "sum"),
    )
    rows = [
        [int(dec), r.min_id, r.max_id, r.rows, r.failed, f"{r.failed / r.rows:.1%}"]
        for dec, r in by_dec.iterrows()
    ]
    lines += md_table(
        ["decile", "min gbifID", "max gbifID", "rows", "failed", "fail rate"], rows,
    )
    lines.append("")

    lines.append("## Failures per genus")
    lines.append("")
    by_genus = status.groupby("genus").agg(rows=("download_failed", "size"),
                                           failed=("download_failed", "sum"))
    by_genus = by_genus[by_genus["failed"] > 0].sort_values("failed", ascending=False)
    if by_genus.empty:
        lines.append("None.")
    else:
        lines += md_table(
            ["genus", "rows", "failed", "fail rate"],
            [[g, r.rows, r.failed, f"{r.failed / r.rows:.1%}"] for g, r in by_genus.iterrows()],
        )
    lines.append("")

    lines.append("## Usable images per genus per split (downloads that succeeded)")
    lines.append("")
    usable = status[~status["download_failed"]]
    pivot = usable.pivot_table(index="genus", columns="split", values="specimen_code",
                               aggfunc="count", fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    lines += md_table(
        ["genus", "train", "test", "total"],
        [[g, r.get("train", 0), r.get("test", 0), r["total"]] for g, r in pivot.iterrows()],
    )
    lines.append("")

    out = REPORTS / "download_stats.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("stats written to %s", out)


def main() -> None:
    setup_logging(REPORTS / "04_download.log")
    cfg = load_config(ROOT / "config.yaml")
    client = GbifClient()

    src = DATA / "dataset.csv"
    if not src.exists():
        log.error("%s missing; run 03_build_dataset.py first", src)
        sys.exit(1)
    df = pd.read_csv(src)
    log.info("dataset: %d images to ensure", len(df))

    size = f"{cfg['image_max_px']}x{cfg['image_max_px']}"
    downloaded = skipped = failed = 0
    outcomes: list[dict] = []

    for row in tqdm(df.itertuples(index=False), total=len(df), unit="img"):
        dest = ROOT / str(row.image_path)  # sanitized path built in phase 3
        outcome = {
            "specimen_code": row.specimen_code,
            "genus": row.genus,
            "split": row.split,
            "gbifID": row.gbifID,
            "eventDate": row.eventDate,
            "image_url": row.image_url,
            "image_path": row.image_path,
            "download_failed": False,
            "error": "",
        }
        if dest.exists() and is_valid_image(dest):
            skipped += 1
            outcomes.append(outcome)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)

        # two cache attempts only, no antweb.org fallback (Cloudflare)
        attempts = [
            cache_url(row.gbifID, row.image_url, size),  # cache, resized
            cache_url(row.gbifID, row.image_url, "x"),   # cache, original
        ]
        last_error: str | None = "no attempt"
        for i, url in enumerate(attempts):
            last_error = fetch(client, url, dest)
            if last_error is None:
                downloaded += 1
                break
            if i < len(attempts) - 1:
                time.sleep(1.0)
        if last_error is not None:
            failed += 1
            outcome["download_failed"] = True
            outcome["error"] = last_error
        outcomes.append(outcome)

    status = pd.DataFrame(outcomes)
    status.to_csv(DATA / "download_status.csv", index=False)
    errors = status[status["download_failed"]]
    if not errors.empty:
        errors.to_csv(DATA / "download_errors.csv", index=False)
        log.info("errors written to %s", DATA / "download_errors.csv")
    write_download_stats(status)

    log.info("done: %d downloaded, %d skipped (already present), %d failed",
             downloaded, skipped, failed)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download dataset images directly from antweb.org with a browser-impersonating
client (curl_cffi). Standalone: needs only curl_cffi, pandas, pillow, tqdm and
data/dataset.csv — it can run from a laptop on a residential IP if the VPS is
blocked, then the data/images/ tree is rsync'ed back.

Polite by design: one request at a time, random 0.5-1.0 s sleep between
requests, exponential backoff on 403/429/5xx, and a hard stop after 10
consecutive 403s (that means Cloudflare is blocking this IP — stop and report
rather than hammer).

For each row: try the _med size first (smaller, enough for 1024px training
crops), then _high, then _low. A Cloudflare challenge page ("Just a moment" /
cf-chl markers) is treated as a 403 and never saved as an image.

Usage:
    python scripts/05_download_antweb.py --limit 10   # smoke test
    python scripts/05_download_antweb.py              # full run (resume default)
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
from curl_cffi import requests as cffi_requests
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dataset.csv"
ERRORS_CSV = ROOT / "data" / "download_errors_antweb.csv"
LOG_FILE = ROOT / "reports" / "05_download_antweb.log"

HEADERS = {
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
    "Referer": "https://www.antweb.org/",
}
SIZES = ["med", "high", "low"]
CHALLENGE_MARKERS = (b"Just a moment", b"cf-chl", b"challenges.cloudflare.com")
MAX_CONSECUTIVE_403 = 10
MAX_TRIES_PER_URL = 3

log = logging.getLogger("download_antweb")


def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")],
    )


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def looks_like_challenge(content: bytes) -> bool:
    head = content[:8192]
    return any(marker in head for marker in CHALLENGE_MARKERS)


def size_variants(url: str) -> list[str]:
    """AntWeb image URLs end in _<size>.jpg; try med, then high, then low."""
    m = re.search(r"_(high|med|low)\.jpg$", url)
    if not m:
        return [url]
    return [url[: m.start()] + f"_{size}.jpg" for size in SIZES]


def fetch(session: cffi_requests.Session, url: str, dest: Path) -> tuple[bool, str]:
    """Try one URL with backoff. Returns (success, final_status_string)."""
    backoff = 2.0
    status = "no attempt"
    for attempt in range(MAX_TRIES_PER_URL):
        try:
            resp = session.get(url, headers=HEADERS, timeout=60)
        except Exception as exc:
            status = f"request error: {exc}"
            log.info("GET %s -> %s", url, status)
            time.sleep(backoff)
            backoff *= 2
            continue
        code = resp.status_code
        challenged = code == 200 and looks_like_challenge(resp.content)
        if challenged:
            code = 403  # a challenge page is a block, whatever the HTTP code
        status = f"HTTP {code}" + (" (cf challenge)" if challenged else "")
        log.info("GET %s -> %s", url, status)
        if code == 200:
            ctype = resp.headers.get("Content-Type", "")
            if not ctype.startswith("image"):
                return False, f"not an image ({ctype})"
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(resp.content)
            if not is_valid_image(tmp):
                tmp.unlink(missing_ok=True)
                return False, "Pillow validation failed"
            tmp.rename(dest)
            return True, status
        if code == 404:
            return False, status  # try the next size variant, no retry
        if code in (403, 429) or code >= 500:
            time.sleep(backoff)
            backoff *= 2
            continue
        return False, status
    return False, status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="only process the first N pending rows (0 = all)")
    parser.add_argument("--no-resume", action="store_true",
                        help="re-download even if a valid file exists")
    args = parser.parse_args()

    setup_logging()
    df = pd.read_csv(DATASET)
    if not args.no_resume:
        pending_mask = df["image_path"].map(
            lambda p: not ((ROOT / p).exists() and is_valid_image(ROOT / p))
        )
        df = df[pending_mask]
    if args.limit:
        df = df.head(args.limit)
    log.info("rows to download: %d", len(df))

    session = cffi_requests.Session(impersonate="chrome")
    downloaded = failed = 0
    consecutive_403 = 0
    errors: list[dict] = []

    for row in tqdm(df.itertuples(index=False), total=len(df), unit="img"):
        dest = ROOT / str(row.image_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        success, status = False, "no attempt"
        for url in size_variants(str(row.image_url)):
            success, status = fetch(session, url, dest)
            if success or "403" not in status and "404" not in status:
                break  # stop on success or a non-retriable oddity
            if "403" in status:
                break  # blocked — trying other sizes won't help
        if success:
            downloaded += 1
            consecutive_403 = 0
        else:
            failed += 1
            errors.append({"specimen_code": row.specimen_code,
                           "image_url": row.image_url,
                           "status": status})
            if "403" in status:
                consecutive_403 += 1
                if consecutive_403 >= MAX_CONSECUTIVE_403:
                    log.error(
                        "%d consecutive 403s — Cloudflare is blocking this IP. "
                        "Stopping. Run this script from another network "
                        "(e.g. a laptop on a residential connection).",
                        consecutive_403,
                    )
                    break
            else:
                consecutive_403 = 0
        time.sleep(random.uniform(0.5, 1.0))

    if errors:
        write_header = not ERRORS_CSV.exists()
        with ERRORS_CSV.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["specimen_code", "image_url", "status"])
            if write_header:
                writer.writeheader()
            writer.writerows(errors)
        log.info("failures appended to %s", ERRORS_CSV)

    log.info("done: %d downloaded, %d failed", downloaded, failed)


if __name__ == "__main__":
    main()

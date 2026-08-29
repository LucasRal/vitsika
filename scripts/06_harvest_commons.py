#!/usr/bin/env python3
"""Fallback image source: Wikimedia Commons "Images from AntWeb" category.

antweb.org blocks datacenter IPs outright and the GBIF image cache is cold
(see reports/download_stats.md), but ~33k AntWeb images were bulk-uploaded to
Commons years ago with titles like
"Pachycondyla cambouei casent0451572 profile 1.jpg". Strategy:

1. Enumerate the whole category once (500 titles/call, cached to
   data/commons_files.txt).
2. Parse specimen code / view / shot index from each title and join against
   data/dataset.csv locally, no per-specimen search calls.
3. For pending rows, pick the Commons file matching the row's view (profile
   or dorsal), shot 1 preferred, lowest shot otherwise.
4. Resolve URLs in batches of 50 via prop=imageinfo (1024px thumb) and
   download from upload.wikimedia.org (not Cloudflare-gated).

Resumable: rows with a valid image on disk are skipped; outcomes go to
data/download_status_commons.csv. Commons AntWeb files are CC BY-SA, same
terms as AntWeb's own media.
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
import requests
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

API = "https://commons.wikimedia.org/w/api.php"
CATEGORY = "Category:Images from AntWeb"
HEADERS = {"User-Agent": "mg-ants-poc/0.1 (research POC; AntWeb images via Commons)"}
THUMB_PX = 1024

CODE_RE = re.compile(r"(casent[0-9][\w().-]*|antweb\d+)", re.IGNORECASE)
VIEW_RE = re.compile(r"\b(profile|dorsal|head|label)\b(?:\s+(\d+))?")
VIEW_LETTER = {"profile": "p", "dorsal": "d", "head": "h", "label": "l"}

log = logging.getLogger("commons")


def api_get(session: requests.Session, params: dict, tries: int = 6) -> dict:
    backoff = 5.0
    for attempt in range(tries):
        resp = session.get(API, params={**params, "format": "json"},
                           headers=HEADERS, timeout=60)
        if resp.status_code == 429 or resp.status_code >= 500:
            log.warning("HTTP %d from Commons API; retry in %.0fs", resp.status_code, backoff)
            time.sleep(backoff)
            backoff *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Commons API kept failing after {tries} tries")


def enumerate_category(session: requests.Session) -> list[str]:
    cache = DATA / "commons_files.txt"
    if cache.exists():
        titles = cache.read_text(encoding="utf-8").splitlines()
        log.info("category cache: %d titles from %s", len(titles), cache.name)
        return titles
    titles: list[str] = []
    cont: dict = {}
    while True:
        r = api_get(session, {
            "action": "query", "list": "categorymembers", "cmtitle": CATEGORY,
            "cmtype": "file", "cmlimit": "500", **cont,
        })
        titles += [m["title"] for m in r["query"]["categorymembers"]]
        if len(titles) % 5000 < 500:
            log.info("enumerated %d titles...", len(titles))
        if "continue" in r:
            cont = r["continue"]
            time.sleep(0.3)
        else:
            break
    cache.write_text("\n".join(titles), encoding="utf-8")
    log.info("category enumerated: %d titles (cached)", len(titles))
    return titles


def parse_title(title: str) -> tuple[str, str | None, int] | None:
    """'File:Genus species casent0451572 profile 1.jpg' -> (code, view, shot)."""
    t = title.removeprefix("File:").lower()
    t = re.sub(r"\.jpe?g$", "", t)
    m = CODE_RE.search(t)
    if not m:
        return None
    view, shot = None, 1
    mv = VIEW_RE.search(t)
    if mv:
        view = VIEW_LETTER[mv.group(1)]
        shot = int(mv.group(2) or 1)
    return m.group(1), view, shot


def resolve_urls(session: requests.Session, titles: list[str]) -> dict[str, str]:
    """Batch prop=imageinfo: title -> ORIGINAL file URL. On-demand thumbnails
    (iiurlwidth) get heavily 429-throttled; originals are cache-served.
    We downscale locally instead."""
    urls: dict[str, str] = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        r = api_get(session, {
            "action": "query", "titles": "|".join(batch),
            "prop": "imageinfo", "iiprop": "url",
        })
        normalized = {n["to"]: n["from"] for n in r["query"].get("normalized", [])}
        for page in r["query"]["pages"].values():
            title = normalized.get(page.get("title"), page.get("title"))
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("url")
            if url:
                urls[title] = url
        time.sleep(0.5)
    return urls


def download_with_backoff(session: requests.Session, url: str,
                          tries: int = 8) -> requests.Response:
    backoff = 30.0
    for attempt in range(tries):
        resp = session.get(url, headers=HEADERS, timeout=120)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = min(900.0, float(resp.headers.get("Retry-After") or backoff))
            log.warning("HTTP %d on download; waiting %.0fs", resp.status_code, wait)
            time.sleep(wait)
            backoff *= 2
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="only process the first N pending rows (0 = all)")
    args = parser.parse_args()

    setup_logging(REPORTS / "06_harvest_commons.log")
    session = requests.Session()

    df = pd.read_csv(DATA / "dataset.csv")
    pending = df[df["image_path"].map(
        lambda p: not ((ROOT / p).exists() and is_valid_image(ROOT / p)))]
    if args.limit:
        pending = pending.head(args.limit)
    log.info("dataset rows: %d, pending (no valid image on disk): %d", len(df), len(pending))

    titles = enumerate_category(session)

    # specimen_code -> list of (view, shot, title)
    by_code: dict[str, list[tuple[str | None, int, str]]] = {}
    for title in titles:
        parsed = parse_title(title)
        if parsed:
            code, view, shot = parsed
            by_code.setdefault(code, []).append((view, shot, title))
    log.info("parsed %d distinct specimen codes from Commons titles", len(by_code))

    # choose the best-matching Commons file for each pending row
    chosen: list[tuple[pd.Series, str]] = []
    for _, row in pending.iterrows():
        cands = by_code.get(str(row["specimen_code"]).lower(), [])
        same_view = [(shot, t) for view, shot, t in cands if view == row["view"]]
        if not same_view:
            continue
        same_view.sort()  # shot 1 first, then lowest shot
        chosen.append((row, same_view[0][1]))
    log.info("Commons match for %d of %d pending rows", len(chosen), len(pending))

    urls = resolve_urls(session, [t for _, t in chosen])
    log.info("resolved %d file URLs", len(urls))

    downloaded = failed = 0
    outcomes: list[dict] = []
    for row, title in tqdm(chosen, unit="img"):
        outcome = {"specimen_code": row["specimen_code"], "genus": row["genus"],
                   "commons_title": title, "image_path": row["image_path"],
                   "ok": False, "error": ""}
        url = urls.get(title)
        if not url:
            outcome["error"] = "no URL resolved"
            failed += 1
            outcomes.append(outcome)
            continue
        dest = ROOT / str(row["image_path"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            resp = download_with_backoff(session, url)
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(resp.content)
            if not is_valid_image(tmp):
                tmp.unlink(missing_ok=True)
                raise ValueError("Pillow validation failed")
            with Image.open(tmp) as im:  # downscale locally to THUMB_PX max
                im = im.convert("RGB")
                im.thumbnail((THUMB_PX, THUMB_PX))
                im.save(dest, "JPEG", quality=90)
            tmp.unlink(missing_ok=True)
            outcome["ok"] = True
            downloaded += 1
        except Exception as exc:
            outcome["error"] = str(exc)
            failed += 1
        outcomes.append(outcome)
        time.sleep(random.uniform(4.0, 6.0))

    status_path = DATA / "download_status_commons.csv"
    with status_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(outcomes[0].keys()) if outcomes
                                else ["specimen_code", "genus", "commons_title",
                                      "image_path", "ok", "error"])
        writer.writeheader()
        writer.writerows(outcomes)
    log.info("done: %d downloaded, %d failed; status in %s", downloaded, failed, status_path)


if __name__ == "__main__":
    main()

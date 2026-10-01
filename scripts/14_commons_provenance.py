#!/usr/bin/env python3
"""Recover which Wikimedia Commons file each "commons" image came from.

06_harvest_commons.py chose one Commons file per specimen (same view, lowest
shot) but recorded its choice only for its last run (13 rows): the status
file was rewritten on every run. This script re-applies that selection to
every image_source == "commons" row of data/dataset.csv, using the cached
category listing data/commons_files.txt and the harvester's own
parse_title(), then confirms each candidate against the file on disk:

1. Resolve the candidate's original URL, SHA-1, size and upload timestamp
   (prop=imageinfo, 50 titles per call).
2. Download the original once into --cache-dir (outside the repo; SHA-1
   checked against Commons). Re-runs read the cache, not the network.
3. Compare with data/images/...:
   exact         the disk file is byte-identical to (or decodes to the same
                 pixels as) one of: the original; the harvester's downscale
                 of it (RGB, thumbnail 1024, JPEG q90); or the thumbnail
                 Commons serves for a 1024 px request (MediaWiki rounds that
                 up to its 1280 px step; the current harvester never fetches
                 these, so the 83 files that are 1280 wide most likely come
                 from an earlier, uncommitted version of it, whose docstring
                 still describes 1024 px thumbnails);
   visual_match  same aspect ratio, and after resizing the original to the
                 disk size: PSNR >= PSNR_MIN dB and pHash distance <= PHASH_MAX;
   unconfirmed   anything else (no candidate, download failed, mismatch).
   When the first-ranked candidate does not match, the other same-view
   candidates of that specimen are tried; candidate_rank says which matched.

Nothing in data/images is modified. Output: data/commons_provenance.csv
(one row per commons image; rows already present are skipped, so the run
is resumable). Network: commons.wikimedia.org API and upload.wikimedia.org.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy.fft import dctn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
OUT = DATA / "commons_provenance.csv"
DEFAULT_CACHE = ROOT.parent / "mg-ants-cache" / "commons_originals"
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "mg-ants-poc/0.2 (https://vitsika.lucas-ralambo.com; "
                         "research reproducibility check of AntWeb images on Commons)"}
PSNR_MIN = 30.0      # dB; an independent re-encode of the same photo sits well above
PHASH_MAX = 6        # of 64 bits
ASPECT_TOL = 0.01    # relative difference of width/height

log = logging.getLogger("provenance")


def load_harvester():
    """06_harvest_commons.py as a module, for parse_title() and THUMB_PX."""
    spec = importlib.util.spec_from_file_location("harvest06", ROOT / "scripts/06_harvest_commons.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def candidates_by_row(h06, rows: pd.DataFrame) -> dict[str, list[str]]:
    """specimen_code -> same-view Commons titles in the harvester's order
    (shot 1 first, then lowest shot; ties by title, as list.sort() did)."""
    titles = (DATA / "commons_files.txt").read_text(encoding="utf-8").splitlines()
    by_code: dict[str, list[tuple[str | None, int, str]]] = {}
    for title in titles:
        parsed = h06.parse_title(title)
        if parsed:
            code, view, shot = parsed
            by_code.setdefault(code, []).append((view, shot, title))
    out = {}
    for row in rows.itertuples():
        cands = by_code.get(str(row.specimen_code).lower(), [])
        same_view = sorted((shot, t) for view, shot, t in cands if view == row.view)
        out[row.specimen_code] = [t for _, t in same_view]
    return out


def api_get(session: requests.Session, params: dict, tries: int = 6) -> dict:
    backoff = 5.0
    for _ in range(tries):
        resp = session.get(API, params={**params, "format": "json"}, headers=HEADERS, timeout=60)
        if resp.status_code == 429 or resp.status_code >= 500:
            log.warning("HTTP %d from Commons API; retry in %.0fs", resp.status_code, backoff)
            time.sleep(backoff)
            backoff *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Commons API kept failing after {tries} tries")


def image_info(session: requests.Session, titles: list[str]) -> dict[str, dict]:
    """title -> {url, sha1, width, height, size, timestamp} of the current file version."""
    info: dict[str, dict] = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        r = api_get(session, {"action": "query", "titles": "|".join(batch),
                              "prop": "imageinfo", "iiprop": "url|sha1|size|timestamp"})
        normalized = {n["to"]: n["from"] for n in r["query"].get("normalized", [])}
        for page in r["query"]["pages"].values():
            title = normalized.get(page.get("title"), page.get("title"))
            ii = (page.get("imageinfo") or [None])[0]
            if ii:
                info[title] = ii
        time.sleep(0.5)
    return info


def download(session: requests.Session, url: str, dest: Path, sha1: str, tries: int = 8) -> bytes:
    if dest.exists():
        data = dest.read_bytes()
        if hashlib.sha1(data).hexdigest() == sha1:
            return data
    backoff = 30.0
    for _ in range(tries):
        resp = session.get(url, headers=HEADERS, timeout=120)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = min(900.0, float(resp.headers.get("Retry-After") or backoff))
            log.warning("HTTP %d on download; waiting %.0fs", resp.status_code, wait)
            time.sleep(wait)
            backoff *= 2
            continue
        resp.raise_for_status()
        data = resp.content
        got = hashlib.sha1(data).hexdigest()
        if got != sha1:
            raise ValueError(f"SHA-1 mismatch: Commons says {sha1}, got {got}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        time.sleep(random.uniform(4.0, 6.0))   # same pacing as the harvester
        return data
    raise RuntimeError(f"download kept failing: {url}")


def server_thumbnail(session: requests.Session, title: str, width: int, cache_dir: Path) -> bytes:
    """The JPEG Commons serves for prop=imageinfo&iiurlwidth=<width>."""
    r = api_get(session, {"action": "query", "titles": title, "prop": "imageinfo",
                          "iiprop": "url", "iiurlwidth": str(width)})
    ii = list(r["query"]["pages"].values())[0]["imageinfo"][0]
    dest = cache_dir / "thumbs" / f"{hashlib.sha1(title.encode()).hexdigest()}_{ii['thumbwidth']}.jpg"
    if dest.exists():
        return dest.read_bytes()
    resp = session.get(ii["thumburl"], headers=HEADERS, timeout=120)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    time.sleep(random.uniform(4.0, 6.0))
    return resp.content


def same_image(a: bytes, b: bytes) -> str | None:
    """'byte-identical', 'pixel-identical' or None."""
    if hashlib.sha256(a).digest() == hashlib.sha256(b).digest():
        return "byte-identical"
    ia, ib = Image.open(io.BytesIO(a)).convert("RGB"), Image.open(io.BytesIO(b)).convert("RGB")
    if ia.size == ib.size and np.array_equal(np.asarray(ia), np.asarray(ib)):
        return "pixel-identical"
    return None


def harvester_downscale(original: bytes, thumb_px: int) -> bytes:
    """Exactly what 06_harvest_commons.py writes for a downloaded original."""
    with Image.open(io.BytesIO(original)) as im:
        im = im.convert("RGB")
        im.thumbnail((thumb_px, thumb_px))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def phash(im: Image.Image) -> np.ndarray:
    """64-bit DCT perceptual hash (32x32 grey, 8x8 low frequencies, median)."""
    a = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    low = dctn(a, norm="ortho")[:8, :8].ravel()
    return low > np.median(low[1:])


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def compare(disk_bytes: bytes, original: bytes, thumb_px: int, thumbnail=None) -> dict:
    """Evidence that the disk file came from this original. `thumbnail` is a
    zero-argument callable returning Commons' server thumbnail (network)."""
    exact = {"match_confidence": "exact", "psnr_db": float("inf"), "phash_distance": 0}
    how = same_image(disk_bytes, original)
    if how:
        return {**exact, "method": f"original file, {how}"}
    how = same_image(disk_bytes, harvester_downscale(original, thumb_px))
    if how:
        return {**exact, "method": f"harvester downscale, {how}"}
    disk = Image.open(io.BytesIO(disk_bytes)).convert("RGB")
    orig = Image.open(io.BytesIO(original)).convert("RGB")
    if thumbnail is not None and max(orig.size) > thumb_px:
        how = same_image(disk_bytes, thumbnail())
        if how:
            return {**exact, "method": f"Commons server thumbnail, {how}"}
    ar_d, ar_o = disk.width / disk.height, orig.width / orig.height
    dist = int(np.count_nonzero(phash(disk) != phash(orig)))
    if abs(ar_d - ar_o) / ar_o > ASPECT_TOL:
        return {"match_confidence": "unconfirmed", "method": "aspect ratio differs",
                "psnr_db": float("nan"), "phash_distance": dist}
    resized = orig.resize(disk.size, Image.LANCZOS)
    p = psnr(np.asarray(resized), np.asarray(disk))
    ok = p >= PSNR_MIN and dist <= PHASH_MAX
    return {"match_confidence": "visual_match" if ok else "unconfirmed",
            "method": "resized original vs disk" if ok else "pixels differ",
            "psnr_db": round(p, 2), "phash_distance": dist}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                        help=f"where Commons originals are kept (default {DEFAULT_CACHE})")
    parser.add_argument("--limit", type=int, default=0, help="process at most N pending rows")
    parser.add_argument("--codes", nargs="*", default=[], help="only these specimen codes")
    args = parser.parse_args()
    setup_logging(REPORTS / "14_commons_provenance.log")

    h06 = load_harvester()
    df = pd.read_csv(DATA / "dataset.csv")
    rows = df[df["image_source"] == "commons"]
    if args.codes:
        rows = rows[rows["specimen_code"].isin(args.codes)]
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["specimen_code"])
    pending = rows[~rows["specimen_code"].isin(done["specimen_code"])]
    if args.limit:
        pending = pending.head(args.limit)
    log.info("commons rows %d, already done %d, pending %d", len(rows),
             rows["specimen_code"].isin(done["specimen_code"]).sum(), len(pending))
    if pending.empty:
        return

    cands = candidates_by_row(h06, pending)
    session = requests.Session()
    all_titles = sorted({t for ts in cands.values() for t in ts})
    info = image_info(session, all_titles)
    log.info("imageinfo for %d of %d candidate titles", len(info), len(all_titles))

    results: list[dict] = []
    for i, row in enumerate(pending.itertuples(), 1):
        titles = cands.get(row.specimen_code, [])
        disk_bytes = (ROOT / row.image_path).read_bytes()
        rec = {"specimen_code": row.specimen_code, "genus": row.genus, "view": row.view,
               "image_path": row.image_path, "n_candidates": len(titles), "candidate_rank": None,
               "commons_file": None, "commons_url": None, "commons_sha1": None,
               "commons_width": None, "commons_height": None, "commons_upload_timestamp": None,
               "match_confidence": "unconfirmed", "method": "no same-view Commons file",
               "psnr_db": None, "phash_distance": None, "rank1_psnr_db": None, "error": ""}
        for rank, title in enumerate(titles, 1):
            ii = info.get(title)
            if ii is None:
                rec["error"] += f"rank {rank}: no imageinfo; "
                continue
            try:
                original = download(session, ii["url"], args.cache_dir / ii["sha1"], ii["sha1"])
            except Exception as exc:  # noqa: BLE001
                rec["error"] += f"rank {rank}: {exc}; "
                continue
            try:
                ev = compare(disk_bytes, original, h06.THUMB_PX,
                             thumbnail=lambda t=title: server_thumbnail(session, t, h06.THUMB_PX, args.cache_dir))
            except Exception as exc:  # noqa: BLE001 (thumbnail request failed: fall back to pixels)
                rec["error"] += f"rank {rank}: thumbnail: {exc}; "
                ev = compare(disk_bytes, original, h06.THUMB_PX)
            if rank == 1:
                rec["rank1_psnr_db"] = ev["psnr_db"]
            if rank == 1 or ev["match_confidence"] != "unconfirmed":
                rec.update(candidate_rank=rank, commons_file=title.removeprefix("File:"),
                           commons_url=ii["url"], commons_sha1=ii["sha1"],
                           commons_width=ii.get("width"), commons_height=ii.get("height"),
                           commons_upload_timestamp=ii.get("timestamp"), **ev)
            if ev["match_confidence"] != "unconfirmed":
                break
        results.append(rec)
        if i % 25 == 0 or i == len(pending):
            new = pd.DataFrame(results)
            pd.concat([done, new], ignore_index=True).to_csv(OUT, index=False)
            counts = pd.concat([done, new])["match_confidence"].value_counts().to_dict()
            log.info("%d/%d processed; totals so far %s", i, len(pending), counts)


if __name__ == "__main__":
    main()

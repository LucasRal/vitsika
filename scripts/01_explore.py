#!/usr/bin/env python3
"""Phase 1: explore the AntWeb dataset on GBIF before harvesting.

Verifies the dataset facts, counts imaged Madagascar records, builds the
genus facet table (cached to data/genera.csv), inspects sample records,
checks the one-record-per-view assumption and tests one image download via
the GBIF image cache. Everything printed is mirrored to
reports/01_explore.txt. Safe to re-run: the genus name resolution is cached.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import (  # noqa: E402
    GbifClient,
    cache_url,
    is_antweb_url,
    load_config,
    setup_logging,
    specimen_code,
    view_from_url,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

log = logging.getLogger("explore")


def section(title: str) -> None:
    log.info("\n%s\n%s", title, "=" * len(title))


def occurrence_count(client: GbifClient, **params: Any) -> int:
    return int(client.get_json("occurrence/search", {**params, "limit": 0})["count"])


def fetch_subfamily(client: GbifClient, cfg: dict, genus_key: str) -> str:
    """GBIF's backbone has no subfamily rank; AntWeb ships it inside
    higherClassification on occurrence records (...;formicidae;<subfamily>)."""
    res = client.get_json(
        "occurrence/search",
        {
            "datasetKey": cfg["dataset_key"],
            "familyKey": cfg["family_key"],
            "genusKey": genus_key,
            "limit": 1,
        },
    )
    if res["results"]:
        parts = (res["results"][0].get("higherClassification") or "").split(";")
        if "formicidae" in parts:
            i = parts.index("formicidae")
            if i + 1 < len(parts):
                return parts[i + 1]
    return ""


def build_genera_csv(client: GbifClient, cfg: dict) -> list[dict[str, str]]:
    """Facet imaged MG records on GENUS_KEY; resolve and cache genus names."""
    path = DATA / "genera.csv"
    cached: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open() as fh:
            cached = {row["genusKey"]: row for row in csv.DictReader(fh)}
        log.info("(loaded %d cached genera from %s)", len(cached), path.name)

    res = client.get_json(
        "occurrence/search",
        {
            "datasetKey": cfg["dataset_key"],
            "country": cfg["country"],
            "familyKey": cfg["family_key"],
            "mediaType": "StillImage",
            "limit": 0,
            "facet": "GENUS_KEY",
            "facetLimit": 300,
            "GENUS_KEY.facetLimit": 300,
        },
    )
    facets = res["facets"][0]["counts"] if res.get("facets") else []
    rows: list[dict[str, str]] = []
    for item in facets:
        key, count = str(item["name"]), int(item["count"])
        if key in cached:
            row = cached[key]
            row["n_records"] = str(count)
        else:
            sp = client.get_json(f"species/{key}")
            row = {
                "genusKey": key,
                "genus": sp.get("canonicalName") or sp.get("scientificName", ""),
                "subfamily": fetch_subfamily(client, cfg, key),
                "n_records": str(count),
            }
        rows.append(row)

    rows.sort(key=lambda r: int(r["n_records"]), reverse=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["genusKey", "genus", "subfamily", "n_records"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_genus_table(rows: list[dict[str, str]], threshold: int) -> None:
    log.info("%-4s %-24s %-20s %10s", "#", "genus", "subfamily", "n_records")
    for i, r in enumerate(rows[:40], 1):
        log.info("%-4d %-24s %-20s %10s", i, r["genus"], r["subfamily"], r["n_records"])
    n_ok = sum(1 for r in rows if int(r["n_records"]) >= threshold)
    log.info("")
    log.info("%d genera with imaged MG records; %d have >= %d records", len(rows), n_ok, threshold)


def sample_records(client: GbifClient, cfg: dict, **extra: Any) -> list[dict]:
    res = client.get_json(
        "occurrence/search",
        {
            "datasetKey": cfg["dataset_key"],
            "country": cfg["country"],
            "familyKey": cfg["family_key"],
            "mediaType": "StillImage",
            "limit": 5,
            **extra,
        },
    )
    return res["results"]


def print_sample_records(recs: list[dict]) -> None:
    log.info("Full JSON of first record:")
    log.info("%s", json.dumps(recs[0], indent=2, ensure_ascii=False))
    log.info("")
    for r in recs:
        log.info(
            "gbifID=%s occurrenceID=%s catalogNumber=%s",
            r.get("key"), r.get("occurrenceID"), r.get("catalogNumber"),
        )
        log.info(
            "  genus=%s species=%s taxonRank=%s sex=%s",
            r.get("genus"), r.get("species"), r.get("taxonRank"), r.get("sex"),
        )
        log.info(
            "  stateProvince=%s locality=%s coords=(%s, %s)",
            r.get("stateProvince"), r.get("locality"),
            r.get("decimalLatitude"), r.get("decimalLongitude"),
        )
        for m in r.get("media", []):
            log.info(
                "  media: %s | title=%s | creator=%s | license=%s",
                m.get("identifier"), m.get("title"), m.get("creator"),
                (m.get("license") or "")[:70],
            )


def check_one_record_per_view(client: GbifClient, cfg: dict, catalog_number: str) -> None:
    code = specimen_code(catalog_number)
    log.info("catalogNumber %r -> specimen_code %r", catalog_number, code)
    res = client.get_json(
        "occurrence/search",
        {"datasetKey": cfg["dataset_key"], "q": code, "limit": 50},
    )
    matches = [
        r for r in res["results"]
        if specimen_code(r.get("catalogNumber") or "") == code
    ]
    log.info("q=%s: %d hits total, %d with matching specimen_code", code, res["count"], len(matches))
    for r in matches:
        views = sorted({view_from_url(m.get("identifier") or "") or "?" for m in r.get("media", [])})
        log.info(
            "  catalogNumber=%-22s occurrenceID=%-30s media=%d views=%s",
            r.get("catalogNumber"), r.get("occurrenceID"),
            len(r.get("media", [])), ",".join(views) or "-",
        )
    suffixed = [r for r in matches if (r.get("catalogNumber") or "") != code]
    if len(matches) > 1 and suffixed:
        log.info("=> one-record-per-view HOLDS for this specimen (suffixed catalogNumbers)")
    elif len(matches) == 1:
        n_media = len(matches[0].get("media", []))
        log.info(
            "=> one-record-per-view does NOT hold here: ONE record carries %d media items",
            n_media,
        )
    else:
        log.info("=> inconclusive for this specimen")


def test_image_download(client: GbifClient, cfg: dict, recs: list[dict]) -> bool:
    from PIL import Image

    size_variants = [f"{cfg['image_max_px']}x{cfg['image_max_px']}", "x"]
    candidates = [
        (r["key"], m["identifier"])
        for r in recs
        for m in r.get("media", [])
        if is_antweb_url(m.get("identifier") or "")
    ]
    if not candidates:
        log.info("no antweb.org media among these records")
        return False
    dest = DATA / "_test.jpg"
    for gbif_id, ident in candidates[:6]:
        for size in size_variants:
            url = cache_url(gbif_id, ident, size)
            resp = client.get(url)
            ctype = resp.headers.get("Content-Type", "")
            log.info("GET %s -> HTTP %d, %s, %d bytes", url, resp.status_code, ctype, len(resp.content))
            if resp.status_code == 200 and ctype.startswith("image"):
                dest.write_bytes(resp.content)
                with Image.open(dest) as im:
                    log.info(
                        "saved %s: %d bytes, %dx%d px, format=%s (source: %s)",
                        dest, len(resp.content), im.size[0], im.size[1], im.format, ident,
                    )
                return True
    return False


def main() -> None:
    setup_logging(REPORTS / "01_explore.txt")
    cfg = load_config(ROOT / "config.yaml")
    client = GbifClient()
    DATA.mkdir(exist_ok=True)

    section("1. Dataset check")
    ds = client.get_json(f"dataset/{cfg['dataset_key']}")
    total = occurrence_count(client, datasetKey=cfg["dataset_key"])
    log.info("title  : %s", ds.get("title"))
    log.info("licence: %s", ds.get("license"))
    log.info("doi    : %s", ds.get("doi"))
    log.info("occurrence records (all countries): %d", total)

    section("2. Formicidae backbone match")
    m = client.get_json("species/match", {"name": "Formicidae"})
    log.info(
        "usageKey=%s rank=%s matchType=%s confidence=%s",
        m.get("usageKey"), m.get("rank"), m.get("matchType"), m.get("confidence"),
    )

    section("3. Madagascar counts")
    base = {"datasetKey": cfg["dataset_key"], "country": cfg["country"]}
    ants = {**base, "familyKey": cfg["family_key"]}
    n_mg = occurrence_count(client, **base)
    n_ants = occurrence_count(client, **ants)
    n_img = occurrence_count(client, **ants, mediaType="StillImage")
    log.info("MG records (all taxa)          : %d", n_mg)
    log.info("MG Formicidae records          : %d", n_ants)
    log.info("MG Formicidae with StillImage  : %d", n_img)

    section("4. Genus facet (imaged MG records)")
    rows = build_genera_csv(client, cfg)
    print_genus_table(rows, cfg["min_specimens_per_genus"])

    section("5. Sample imaged MG records")
    recent = sample_records(client, cfg)
    print_sample_records(recent)

    section("6. One-record-per-view check")
    check_one_record_per_view(client, cfg, recent[0].get("catalogNumber") or "")
    older = sample_records(client, cfg, year=2003)
    if older:
        log.info("")
        log.info("same check on an older (2003) imaged specimen:")
        check_one_record_per_view(client, cfg, older[0].get("catalogNumber") or "")

    section("7. Image download test (GBIF cache)")
    ok = test_image_download(client, cfg, recent)
    if not ok and older:
        log.info("recent images not in the GBIF cache; retrying with 2003 records")
        ok = test_image_download(client, cfg, older)
    log.info("image download test: %s", "OK" if ok else "FAILED")

    section("Done")
    log.info("report saved to %s", REPORTS / "01_explore.txt")


if __name__ == "__main__":
    main()

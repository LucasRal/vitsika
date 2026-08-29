#!/usr/bin/env python3
"""Phase 2: harvest imaged Madagascar occurrence metadata, one genus at a time.

Paginates /v1/occurrence/search per genusKey (from data/genera.csv), explodes
media[] into one row per image, keeps only antweb.org identifiers, and writes
data/raw/records.parquet. For each kept record, one extra call to
/occurrence/{gbifID}/verbatim retrieves the caste (verbatim dwc:sex; GBIF's
interpreted sex field drops 'worker'). Resumable: each finished genus is
recorded in data/raw/done_genera.txt and stored as its own parquet part.

Refuses to run the inline strategy if the imaged MG count exceeds 90,000
(offset + limit must stay <= 100,000); in that case switch to the GBIF
Occurrence Download API.
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import (  # noqa: E402
    GbifClient,
    is_antweb_url,
    load_config,
    setup_logging,
    specimen_code,
    view_from_url,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
REPORTS = ROOT / "reports"

MAX_INLINE_RECORDS = 90_000
PAGE_SIZE = 300
OFFSET_CAP = 100_000

COLUMNS = [
    "gbifID", "occurrenceID", "catalogNumber", "specimen_code", "view",
    "genusKey", "genus", "subfamily", "species", "scientificName",
    "verbatimScientificName", "taxonRank", "sex", "caste", "typeStatus",
    "stateProvince", "locality", "decimalLatitude", "decimalLongitude",
    "elevation", "eventDate", "recordedBy", "identifiedBy",
    "image_url", "image_title", "creator", "license", "rightsHolder", "issues",
]

log = logging.getLogger("harvest")


def media_rows(rec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    catalog = rec.get("catalogNumber") or ""
    for m in rec.get("media", []):
        ident = m.get("identifier") or ""
        if not is_antweb_url(ident):
            continue
        rows.append({
            "gbifID": rec.get("key"),
            "occurrenceID": rec.get("occurrenceID"),
            "catalogNumber": catalog,
            "specimen_code": specimen_code(catalog),
            "view": view_from_url(ident) or "",
            "genusKey": rec.get("genusKey"),
            "genus": rec.get("genus"),
            "subfamily": "",  # filled from genera.csv below
            "species": rec.get("species"),
            "scientificName": rec.get("scientificName"),
            "verbatimScientificName": rec.get("verbatimScientificName"),
            "taxonRank": rec.get("taxonRank"),
            "sex": rec.get("sex"),
            "caste": "",  # filled from the verbatim record (see harvest_genus)
            "typeStatus": rec.get("typeStatus"),
            "stateProvince": rec.get("stateProvince"),
            "locality": rec.get("locality"),
            "decimalLatitude": rec.get("decimalLatitude"),
            "decimalLongitude": rec.get("decimalLongitude"),
            "elevation": rec.get("elevation"),
            "eventDate": rec.get("eventDate"),
            "recordedBy": rec.get("recordedBy"),
            "identifiedBy": rec.get("identifiedBy"),
            "image_url": ident,
            "image_title": m.get("title"),
            "creator": m.get("creator"),
            "license": m.get("license"),
            "rightsHolder": m.get("rightsHolder"),
            "issues": ";".join(rec.get("issues", [])),
        })
    return rows


def fetch_verbatim_caste(client: GbifClient, gbif_id: Any) -> str:
    """AntWeb stores the caste in the verbatim dwc:sex field ('worker',
    'queen', 'male', 'alate queen', ...). GBIF's interpretation layer only
    keeps Male/Female/Other, so 'worker' becomes sex=null; the verbatim
    record is the only reliable source."""
    v = client.get_json(f"occurrence/{gbif_id}/verbatim")
    return (v.get("http://rs.tdwg.org/dwc/terms/sex") or "").strip().lower()


def harvest_genus(client: GbifClient, cfg: dict, genus_key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        if offset + PAGE_SIZE > OFFSET_CAP:
            log.warning("genus %s hit the offset cap at %d; results truncated", genus_key, offset)
            break
        res = client.get_json(
            "occurrence/search",
            {
                "datasetKey": cfg["dataset_key"],
                "country": cfg["country"],
                "familyKey": cfg["family_key"],
                "mediaType": "StillImage",
                "genusKey": genus_key,
                "limit": PAGE_SIZE,
                "offset": offset,
            },
        )
        for rec in res["results"]:
            new_rows = media_rows(rec)
            if new_rows:  # only spend a verbatim call on records we keep
                caste = fetch_verbatim_caste(client, rec["key"])
                for row in new_rows:
                    row["caste"] = caste
            rows.extend(new_rows)
        offset += PAGE_SIZE
        if res.get("endOfRecords", True):
            break
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    setup_logging(REPORTS / "02_harvest.log")
    cfg = load_config(ROOT / "config.yaml")
    client = GbifClient()
    RAW.mkdir(parents=True, exist_ok=True)

    n_img = int(client.get_json(
        "occurrence/search",
        {
            "datasetKey": cfg["dataset_key"],
            "country": cfg["country"],
            "familyKey": cfg["family_key"],
            "mediaType": "StillImage",
            "limit": 0,
        },
    )["count"])
    log.info("imaged MG Formicidae records: %d", n_img)
    if n_img > MAX_INLINE_RECORDS:
        log.error(
            "count %d > %d: the search API offset cap would truncate results. "
            "Switch to the GBIF Occurrence Download API instead.",
            n_img, MAX_INLINE_RECORDS,
        )
        sys.exit(1)

    genera_path = DATA / "genera.csv"
    if not genera_path.exists():
        log.error("%s missing; run 01_explore.py first", genera_path)
        sys.exit(1)
    with genera_path.open() as fh:
        genera = list(csv.DictReader(fh))
    subfamily_by_key = {g["genusKey"]: g["subfamily"] for g in genera}

    done_path = RAW / "done_genera.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()

    for i, g in enumerate(genera, 1):
        key = g["genusKey"]
        part_path = RAW / f"genus_{key}.parquet"
        if key in done and part_path.exists():
            log.info("[%d/%d] %s (%s): already harvested, skipping", i, len(genera), g["genus"], key)
            continue
        df = harvest_genus(client, cfg, key)
        if not df.empty:
            df["subfamily"] = subfamily_by_key.get(key, "")
        df.to_parquet(part_path, index=False)
        with done_path.open("a") as fh:
            fh.write(key + "\n")
        log.info(
            "[%d/%d] %s (%s): %d media rows, %d specimens",
            i, len(genera), g["genus"], key, len(df), df["specimen_code"].nunique(),
        )

    parts = sorted(RAW.glob("genus_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    out = RAW / "records.parquet"
    df.to_parquet(out, index=False)
    log.info(
        "%s: %d media rows, %d specimens, %d genera",
        out, len(df), df["specimen_code"].nunique(), df["genusKey"].nunique(),
    )


if __name__ == "__main__":
    main()

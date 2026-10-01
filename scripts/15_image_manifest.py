#!/usr/bin/env python3
"""data/image_manifest.csv: one row per image used in data/dataset.csv.

Offline. For each row: the file's sha256, pixel size and source. Columns:

  specimen_code, genus, split, view, image_path, sha256, width, height, bytes
  image_source          commons / gbif_cache (from data/dataset.csv)
  source_url            where the file was fetched from, where known
  commons_file          Commons file name (commons rows; from 14_commons_provenance.py)
  commons_sha1          SHA-1 Commons reports for that file's original
  commons_upload_timestamp
  match_confidence      exact / visual_match / unconfirmed (commons rows only;
                        see 14_commons_provenance.py for the definitions)
  match_method          how the match was established
  source_note           what is and is not known about the source; for commons
                        rows, whether the file is the original, the harvester's
                        downscale of it, or Commons' own thumbnail of it

GBIF-cache rows: data/download_status.csv records that the cache download
succeeded for that image_url, but not which of the two size variants
04_download.py tried (1024x1024 first, then original 'x') delivered it;
source_url gives the 1024x1024 URL and source_note says so. These rows
were not re-fetched.

The manifest is the checksum list for an archive of data/images: restoring
the baseline means every file's sha256 matches this table.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import cache_url, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "image_manifest.csv"


def file_facts(path: Path) -> dict:
    data = path.read_bytes()
    with Image.open(path) as im:
        width, height = im.size
    return {"sha256": hashlib.sha256(data).hexdigest(), "width": width, "height": height,
            "bytes": len(data)}


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    size = f"{cfg['image_max_px']}x{cfg['image_max_px']}"
    df = pd.read_csv(DATA / "dataset.csv")
    prov_path = DATA / "commons_provenance.csv"
    prov = pd.read_csv(prov_path).set_index("specimen_code") if prov_path.exists() else pd.DataFrame()

    rows = []
    for r in df.itertuples():
        rec = {"specimen_code": r.specimen_code, "genus": r.genus, "split": r.split, "view": r.view,
               "image_path": r.image_path, **file_facts(ROOT / r.image_path),
               "image_source": r.image_source, "source_url": "", "commons_file": "",
               "commons_sha1": "", "commons_upload_timestamp": "", "match_confidence": "",
               "match_method": "", "source_note": ""}
        if r.image_source == "gbif_cache":
            rec["source_url"] = cache_url(r.gbifID, r.image_url, size)
            rec["source_note"] = (f"GBIF image cache of {r.image_url}; download success recorded in "
                                  f"download_status.csv; size variant ({size} or original) not recorded")
        elif r.image_source == "commons":
            if r.specimen_code in prov.index:
                p = prov.loc[r.specimen_code]
                for col in ("commons_file", "commons_sha1", "commons_upload_timestamp", "match_confidence"):
                    rec[col] = "" if pd.isna(p[col]) else p[col]
                rec["match_method"] = "" if pd.isna(p["method"]) else p["method"]
                # the API appends utm_* tracking parameters; the file URL is the path
                rec["source_url"] = "" if pd.isna(p["commons_url"]) else str(p["commons_url"]).split("?")[0]
                rank = p["candidate_rank"]
                method = rec["match_method"]
                if method.startswith("original file"):
                    how = "the Commons original, unchanged"
                elif method.startswith("harvester downscale"):
                    how = ("the Commons original after 06_harvest_commons.py's downscale "
                           "(RGB, max 1024 px, JPEG quality 90)")
                elif method.startswith("Commons server thumbnail"):
                    how = ("Commons' own thumbnail of that original (requested at 1024 px, served at "
                           "the next standard width); not what the current harvester fetches, so most likely an "
                           "earlier version of it, whose docstring still describes 1024 px thumbnails")
                else:
                    how = method or "no match"
                rec["source_note"] = (
                    f"recovered by 14_commons_provenance.py: {how}"
                    + ("" if pd.isna(rank) or int(rank) == 1
                       else f"; matched the rank-{int(rank)} same-view candidate, not the harvester's first choice"))
            else:
                rec["match_confidence"] = "unconfirmed"
                rec["source_note"] = "not processed by 14_commons_provenance.py"
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} rows)")
    print(out["image_source"].value_counts().to_string())
    print(out[out["image_source"] == "commons"]["match_confidence"].value_counts().to_string())


if __name__ == "__main__":
    main()

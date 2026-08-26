# mg-ants — Malagasy ant genus classifier, data collection (POC)

Proof-of-concept data pipeline for a classifier that identifies the **genus**
of a Malagasy ant from a specimen photo. Images come from **AntWeb**
(California Academy of Sciences), accessed through the **GBIF API** — the
antweb.org API itself sits behind Cloudflare and blocks scripts, while GBIF
mirrors the full dataset openly. Later phases (not in this repo yet) will use
BioCLIP 2 embeddings + a linear probe and a Gradio demo.

## Pipeline

All scripts are resumable and safe to re-run. Configuration lives in
`config.yaml`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/01_explore.py        # verify dataset facts, build data/genera.csv
                                              # report: reports/01_explore.txt
.venv/bin/python scripts/02_harvest.py        # per-genus metadata harvest -> data/raw/records.parquet
.venv/bin/python scripts/03_build_dataset.py  # filter + split -> data/dataset.csv, reports/dataset_stats.md
.venv/bin/python scripts/04_download.py       # images -> data/images/<genus>/<specimen>_<view>.jpg
```

Key conventions:

- One image per specimen: profile view first, dorsal fallback; label views dropped.
- Caste filter: workers only (the caste is stored in GBIF's `sex` field).
- Genera with fewer than `min_specimens_per_genus` specimens are dropped.
- 80/20 train/test split, stratified by genus, grouped by specimen code, seed 42.
- Downloads go through the GBIF image cache
  (`api.gbif.org/v1/image/cache/...`), not antweb.org directly.

## Attribution

**Dataset citation:**

> Fisher B L (2026). AntWeb. California Academy of Sciences. Occurrence
> dataset https://doi.org/10.15468/wqmjjt accessed via GBIF.org on 2026-08-26.

**Licensing:** the dataset *metadata* is CC BY 4.0; the *images* are
CC BY-SA (+ GFDL) per AntWeb's media terms (see the per-image `license`
column in `data/dataset.csv`). Every displayed image must show the
photographer (the `creator` field), the specimen code, and the mention
"from www.antweb.org" linking to the specimen page
(`https://www.antweb.org/specimen/<specimen_code>`).

# mg-ants — Malagasy ant genus classifier, data collection (POC)

Proof-of-concept data pipeline for a classifier that identifies the **genus**
of a Malagasy ant from a specimen photo. Metadata comes from **AntWeb**
(California Academy of Sciences) via its **GBIF** mirror; images come from
the GBIF image cache and, mostly, from Wikimedia Commons (see "Image
sourcing" below). Later phases (not in this repo yet) will use BioCLIP 2
embeddings + a linear probe and a Gradio demo.

## Current state

- `data/dataset_full.csv` — the **target manifest**: 4,354 specimens,
  39 genera, one image URL per specimen. What we would train on with full
  image access.
- `data/dataset.csv` — the **POC dataset actually on disk**: 1,236 images,
  27 genera, 986 train / 250 test. Built with
  `03_build_dataset.py --available-only` (threshold and split re-applied to
  the obtained images only). Column `image_source` says where each file came
  from (`commons` or `gbif_cache`).
- 1,297 of 4,354 manifest images were obtainable (29.8%). 12 genera fell
  below the 10-specimen threshold as a result — including *Tanipone* (0
  images) and *Vitsika* (1/32), genera described in 2014 that postdate the
  Commons bulk uploads, and *Carebara* (8/140). Full funnel in
  `reports/dataset_stats.md` § Provenance.
- Getting the missing ~70% requires bulk access from AntWeb/CAS;
  `scripts/05_download_antweb.py` + `dataset_full.csv` are the ready path if
  that (or any non-blocked network) materializes.

## Pipeline

All scripts are resumable and safe to re-run. Configuration lives in
`config.yaml` (genus synonyms in `config/synonyms.yaml`).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/01_explore.py          # verify dataset facts, build data/genera.csv
                                                # report: reports/01_explore.txt
.venv/bin/python scripts/02_harvest.py          # per-genus metadata harvest -> data/raw/records.parquet
.venv/bin/python scripts/03_build_dataset.py    # filter + split -> data/dataset_full.csv (full manifest)
.venv/bin/python scripts/04_download.py         # attempt images via GBIF cache (see below)
.venv/bin/python scripts/06_harvest_commons.py  # fallback: images from Wikimedia Commons
.venv/bin/python scripts/03_build_dataset.py --available-only
                                                # -> data/dataset.csv (POC set), reports/dataset_stats.md
.venv/bin/python scripts/05_embed.py            # Phase B: BioCLIP 2 embeddings -> data/embeddings.npy
                                                # + data/embeddings_index.csv, log: reports/05_embed.log
```

Phase B needs torch (CPU build is enough — ~1.4 img/s on 6 cores) and
open_clip; see the note in `requirements.txt`. The first run downloads the
BioCLIP 2 weights (~1.7 GB) into the Hugging Face cache.

Key conventions:

- One image per specimen: profile view first, dorsal fallback; head/label
  views dropped. Shot 1 preferred (standard whole-ant view; higher shot
  numbers are often dissections or SEM).
- Caste filter: workers only. GBIF's interpreted `sex` field drops "worker";
  the true caste is read from the verbatim record
  (`/v1/occurrence/{gbifID}/verbatim`, key `dwc:sex`).
- Genus synonyms remapped before thresholding (Oligomyrmex → Carebara,
  Pyramica → Strumigenys); original GBIF name kept in `genus_gbif`.
- Genera with fewer than `min_specimens_per_genus` (10) specimens are dropped.
- 80/20 train/test split, stratified by genus, grouped by specimen code, seed 42.
- Images stored as `data/images/<genus>/<specimen>_<view>.jpg`, max 1024 px.
- Embeddings: frozen BioCLIP 2 image tower (`hf-hub:imageomics/bioclip-2`,
  ViT-L/14), L2-normalised float32, 768-d. `data/embeddings.npy` row *i* is
  `data/embeddings_index.csv` row *i*; both are sorted by `specimen_code`
  and the index SHA256 is logged so downstream steps can pin the exact set.
  Images that fail to open are skipped (no row), never zero-filled.

## Image sourcing (why three download scripts)

antweb.org blocks scripted access twice over: Cloudflare challenges for
scripts, and a plain 403 for datacenter IPs (so `05_download_antweb.py`,
which impersonates a browser, works only from a residential network).
It also blocks GBIF's own image fetcher, so the GBIF image cache
(`04_download.py`) is cold — 96.8% of requests 404, uniformly across years
(evidence in `reports/download_stats.md`).

The working fallback is `06_harvest_commons.py`: ~33k AntWeb images were
bulk-uploaded to Wikimedia Commons (Category:Images from AntWeb) around
2008–2014. The script enumerates the category once, matches specimen codes
locally, downloads originals politely (4–6 s pacing, Retry-After honored)
and downscales locally. That yielded 1,160 images; the GBIF cache
contributed 137 more.

## Reports

- `reports/dataset_stats.md` — provenance funnel, per-genus counts and
  splits, caste/view distributions, provinces.
- `reports/contact_sheet.jpg` — one random profile image per kept genus (27).
- `reports/download_stats.md` — GBIF-cache failure evidence (kept as-is).

## Attribution

**Dataset citation:**

> Fisher B L (2026). AntWeb. California Academy of Sciences. Occurrence
> dataset https://doi.org/10.15468/wqmjjt accessed via GBIF.org on 2026-08-26.

**Licensing:** the dataset *metadata* is CC BY 4.0; the *images* are
CC BY-SA (+ GFDL) per AntWeb's media terms (see the per-image `license`
column in `data/dataset.csv`). This applies equally to the copies hosted on
Wikimedia Commons. Every displayed image must show the photographer (the
`creator` field), the specimen code, and the mention "from www.antweb.org"
linking to the specimen page
(`https://www.antweb.org/specimen/<specimen_code>`).

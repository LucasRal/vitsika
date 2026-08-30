# mg-ants: Malagasy ant genus classifier, data collection (POC)

Proof-of-concept data pipeline for a classifier that identifies the **genus**
of a Malagasy ant from a specimen photo. Metadata comes from **AntWeb**
(California Academy of Sciences) via its **GBIF** mirror; images come from
the GBIF image cache and, mostly, from Wikimedia Commons (see "Image
sourcing" below). On top of the data pipeline the repo holds the BioCLIP 2
embeddings + linear-probe classifier (Phases B/C), a FastAPI service
(Phase E) and the "Vitsika" web front end; a one-page summary is in
`reports/onepager.md` and a 6-slide deck in `reports/vitsika_deck.pdf`.

## Current state

- `data/dataset_full.csv`, the **target manifest**: 4,354 specimens,
  39 genera, one image URL per specimen. What we would train on with full
  image access.
- `data/dataset.csv`, the **POC dataset actually on disk**: 1,236 images,
  27 genera, 986 train / 250 test. Built with
  `03_build_dataset.py --available-only` (threshold and split re-applied to
  the obtained images only). Column `image_source` says where each file came
  from (`commons` or `gbif_cache`).
- 1,297 of 4,354 manifest images were obtainable (29.8%). 12 genera fell
  below the 10-specimen threshold as a result, including *Tanipone* (0
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
.venv/bin/python scripts/06_embed.py            # Phase B: BioCLIP 2 embeddings -> data/embeddings.npy
                                                # + data/embeddings_index.csv, log: reports/06_embed.log
.venv/bin/python scripts/07_eval.py             # Phase C: zero-shot / linear probe / kNN evaluation
                                                # -> reports/metrics.json, per_genus.csv, errors.csv,
                                                #    confusion_matrix.png; probe saved to data/probe.pkl
.venv/bin/python scripts/08_umap.py             # UMAP of the embeddings -> data/umap_coords.csv, umap_model.pkl,
                                                #    reports/umap_by_subfamily.png, umap_by_genus.png
.venv/bin/python scripts/09_geo.py              # coverage of the full manifest -> data/geo_summary.csv,
                                                #    reports/map_specimens.png
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
                                                # Phase E: serve the classifier (see below)
.venv/bin/python scripts/10_smoke_api.py        # smoke-test the running API -> reports/10_smoke_api.log
```

Phase B needs torch (CPU build is enough, ~1.4 img/s on 6 cores) and
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
(`04_download.py`) is cold: 96.8% of requests 404, uniformly across years
(evidence in `reports/download_stats.md`).

The working fallback is `06_harvest_commons.py`: ~33k AntWeb images were
bulk-uploaded to Wikimedia Commons (Category:Images from AntWeb) around
2008–2014. The script enumerates the category once, matches specimen codes
locally, downloads originals politely (4–6 s pacing, Retry-After honored)
and downscales locally. That yielded 1,160 images; the GBIF cache
contributed 137 more.

## Evaluation (Phase C)

`07_eval.py` compares three classifiers on the frozen embeddings, using the
80/20 split from `dataset.csv` (986 train / 250 test, 27 genera):

- **zero-shot**: BioCLIP 2 text tower on the genus name, prompts
  `"a photo of {genus}, a genus of ant"` and bare `"{genus}"`;
- **linear probe**: `LogisticRegression(class_weight="balanced", C=1.0)`,
  then temperature-scaled (`api/probe.py`: one scalar T on the logits, fitted
  by NLL on 5-fold out-of-fold train logits, so the ranking is untouched and
  the probabilities the API shows are calibrated); saved to `data/probe.pkl`
  for the demo API (raw probe in `data/probe_uncalibrated.pkl`; before/after
  ECE and the rejected sklearn sigmoid-CV alternative in `metrics.json`);
- **nearest neighbour**: cosine top-1 against the train set (its top-3 is
  the majority vote of the 3 nearest).

Results are in `reports/metrics.json` (top-1, top-3, macro-F1),
`reports/per_genus.csv` (recall/F1 per genus and method),
`reports/confusion_matrix.png` (probe, ordered by subfamily) and
`reports/errors.csv` (probe misclassifications with probabilities).

## Serving (Phase E)

`api/` is a FastAPI app that loads BioCLIP 2, the probe, the embeddings, the
fitted UMAP (`data/umap_model.pkl`, written by `08_umap.py`) and the geo
tables once at startup (~25 s on CPU, half of it numba compiling the UMAP
transform so the first request doesn't pay for it) and serves them:

```bash
.venv/bin/pip install fastapi uvicorn python-multipart      # already in requirements.txt
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
.venv/bin/python scripts/10_smoke_api.py                    # in another shell; --base-url to override
```

| Route | Purpose |
|---|---|
| `POST /analyze` | multipart `file` (image/*, ≤ 10 MB) → probe top-3 `predictions` (genus, subfamily, probability) **and** the 5 most similar train specimens (`specimen_code`, species, cosine `similarity`, `image_url`, `antweb_url`, photographer, license), plus `atlas_position` `{x, y}` (the query projected onto the UMAP with `umap_model.transform`; `null` if that fails) and `model_name` / `probe_version`. Response schema `AnalyzeResponse`. |
| `GET /genera` | the 27 POC genera: subfamily, `n_train`, `n_test`, probe F1 (from `reports/per_genus.csv`), `atlas_median` `{x, y}` for the selector's fly-to. |
| `GET /examples` | held-out **test** specimens for the demo picker (`per_genus=2` default, `0` = all 250): code, genus, species, subfamily, photographer, `image_url`, `antweb_url`. |
| `GET /atlas` | every specimen on the UMAP: `specimen_code`, `x`, `y`, genus, subfamily, species, `image_available`; plus the fit parameters. Built once at startup, served from memory (~150 KB). |
| `GET /geo/{genus}` | `n_specimens`, `n_species`, `n_unidentified`, province counts, elevation min/median/max, year range and `[lat, lon]` points (≤ 1000, seeded subsample) from `dataset_full.csv`; 404 if unknown. |
| `GET /images/{specimen_code}` | the local profile-view jpg; thumbnails for the similar-specimen cards; 404 if absent. |
| `GET /health` | `status`, `model_loaded`, `n_embeddings`, library / probe / embedding-index versions. |

Errors are JSON `{"message": …}`: 400 wrong content type, 413 too large,
422 unreadable image, 404 unknown genus or specimen. CORS origins
(`api_cors_origins`, default `http://localhost:3000` for the Next.js dev
server), the upload cap and the point cap live in `config.yaml`; startup and
request logs go to `reports/api.log` (untracked). Layout: `api/main.py`
(app, lifespan, routes), `api/state.py` (loads everything once),
`api/inference.py` (`embed_image`, `predict_genus`, `find_similar`, pure
functions, no FastAPI), `api/schemas.py` (pydantic models).

```bash
curl -s -F "file=@data/images/Royidris/casent0002219_p.jpg" localhost:8000/analyze | python3 -m json.tool
```

Latency on the 6-core CPU box: ~0.8 s per `/analyze` (almost all of it the
ViT-L/14 forward pass; inference is serialised behind a lock).

## Web front end (`web/`)

"Vitsika", a Next.js 15 site over the API: Identify (upload → genus +
similar specimens), Genera (reliability table), Distribution (Leaflet
map per genus), Atlas (ECharts UMAP with the upload as a star) and
Methods. Setup, env vars and the production/nginx sketch are in
[`README-web.md`](README-web.md); screenshots in `reports/web_screenshots/`.

## Reports

- `reports/dataset_stats.md`: provenance funnel, per-genus counts and
  splits, caste/view distributions, provinces.
- `reports/contact_sheet.jpg`: one random profile image per kept genus (27).
- `reports/download_stats.md`: GBIF-cache failure evidence (kept as-is).
- `reports/metrics.json`, `per_genus.csv`, `confusion_matrix.png`,
  `errors.csv`: Phase C evaluation (see above); `eval_notes.md`,
  `errors_sheet.jpg`, `confusions_pairs.jpg`: manual error analysis.
- `reports/umap_by_subfamily.png`, `umap_by_genus.png`: 2-D UMAP of the
  embeddings (subfamilies form clusters; a few cross-subfamily islands group
  ants by body plan, e.g. the long-legged *Camponotus imitator* /
  *Aphaenogaster* / *Odontomachus coquereli* island).
- `reports/map_specimens.png`, `data/geo_summary.csv`, `geo_by_place.csv`,
  `geo_by_locality.csv`: geographic and
  temporal coverage of the full 4,354-specimen manifest (`09_geo.py`).
  Plot conventions live in `scripts/viz.py`.
- `reports/10_smoke_api.log`: last smoke-test run of the API (Phase E).
- `reports/onepager.md`: one-page summary (what, data chain, results,
  findings, limitations, next steps) with sources; `reports/vitsika_deck.pdf`
  (6 slides) is built from it by `scripts/11_deck.py`.

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

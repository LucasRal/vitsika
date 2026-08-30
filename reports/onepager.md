# Vitsika, in one page

Malagasy ant genus identification from a profile photograph. Proof of
concept, live at https://vitsika.lucas-ralambo.com. Every number below is
taken from a file in this repository; the source is given in brackets.

## What it is

Vitsika takes one lateral photo of a worker ant, embeds it with the frozen
BioCLIP 2 image tower (ViT-L/14, 768-d, no fine-tuning) and proposes the
three most likely genera among the 27 covered with a balanced logistic-
regression probe fitted on those embeddings (probabilities temperature-scaled
on cross-validated training folds) [README.md § Evaluation;
config.yaml]. Alongside the classifier it retrieves the five most similar
training specimens by cosine similarity, places the photo on a fitted UMAP
of all 1,236 specimens, and links every image back to its AntWeb record, so
a person can check the answer against real museum specimens rather than
trust a probability [README.md § Serving; api/inference.py].

## Data chain

| Step | Count | Source |
|---|---|---|
| AntWeb occurrence dataset on GBIF (Fisher B. L., CAS), doi:10.15468/wqmjjt, accessed 2026-08-26 | 920,136 records, all countries | reports/01_explore.txt |
| Madagascar, Formicidae | 141,430 records; 6,995 with a still image | reports/01_explore.txt |
| Harvested per genus (60 genera with imaged records) | 5,858 records, 5,857 specimens, 24,013 media rows | reports/02_harvest.log, 03_build_dataset.log |
| Workers only, profile (or dorsal) view, one image per specimen, >= 10 per genus | **4,354 specimens, 39 genera** (target manifest, `data/dataset_full.csv`) | reports/dataset_stats.md |
| Images actually obtained | **1,297 (29.8 %)**: 1,160 from Wikimedia Commons, 137 from the GBIF image cache | reports/dataset_stats.md |
| Threshold re-applied to obtained images | **1,236 images, 27 genera, 8 subfamilies** (1,120 Commons + 116 cache; 1,232 profile + 4 dorsal) | data/dataset.csv |
| Split (stratified by genus, grouped by specimen, seed 42) | **986 train / 250 test** | reports/metrics.json |

## Results on the 250 held-out specimens (27 genera) [reports/metrics.json]

| Method | Top-1 | Top-3 | Macro-F1 |
|---|---|---|---|
| Zero-shot, "a photo of {genus}, a genus of ant" | 74.4 % | 94.0 % | 0.590 |
| Zero-shot, bare genus name | 32.8 % | 48.8 % | 0.114 |
| **Linear probe (deployed)** | **92.8 %** | **98.4 %** | **0.885** |
| Nearest neighbour (cosine; top-3 = vote of 3) | 91.6 % | 93.2 % | 0.834 |

Of the probe's 18 errors, 12 are genuinely hard look-alikes or species
unseen in training, 3 are poor photographs, 2 odd angles, 1 unexplained;
nearly all sit inside a subfamily block of the confusion matrix
[reports/eval_notes.md; reports/confusion_matrix.png].

## Two data-integration findings

1. **GBIF's image cache is empty for AntWeb.** 4,215 of 4,354 cache requests
   returned 404 (96.8 %), uniformly across collection years and gbifID
   deciles (94-100 %), because antweb.org blocks GBIF's fetcher as it blocks
   scripts (Cloudflare) and datacenter IPs (403). The working source is the
   2008-2014 bulk upload of AntWeb images to Wikimedia Commons
   (Category:Images from AntWeb, 33,081 files), matched by specimen code
   [reports/download_stats.md; reports/01_explore.txt § 7; data/commons_files.txt; README.md § Image sourcing].
2. **Caste is not in GBIF's interpreted fields.** GBIF maps `sex` to
   Male/Female/Other, so every worker comes through as null: 4,540 of 5,857
   specimens (17,518 of 24,013 media rows). The caste has to be read from the
   verbatim record (`/occurrence/{id}/verbatim`, key `dwc:sex`): 4,539
   workers, 673 queens, 640 males, 4 other, 1 "1 head"
   [data/raw/records.parquet; scripts/02_harvest.py; reports/dataset_stats.md § Caste filter].

## Honest limitations

- 70.2 % of the manifest's images are missing (3,057 of 4,354); getting them needs bulk access from AntWeb/CAS [reports/dataset_stats.md].
- 12 of 39 manifest genera are not covered, including *Vitsika* itself (1 of 32 images obtained), *Tanipone* (0 of 30) and *Carebara* (8 of 140) [reports/dataset_stats.md § Provenance].
- Workers only: 673 queens and 640 males were dropped; a queen or male photo still gets a worker-genus answer [reports/dataset_stats.md].
- Studio images only: 1,232 of 1,236 training images are AntWeb pinned-specimen profile shots; field photos are out of distribution [data/dataset.csv].
- Closed world (probabilities are temperature-calibrated on train folds, ECE 0.548 → 0.023, but a photo outside the 27 genera still gets an answer): 13 of 27 genera are flagged low-reliability (fewer than 5 test images or probe F1 < 0.8); 10 genera have <= 3 test images; the split is by genus, not species, so some test species have no training example (359 species in the set) [reports/per_genus.csv; reports/eval_notes.md].
- Maps show collecting effort, not abundance; ~0.8 s per image on a 6-core CPU, single worker [DEPLOY.md; README.md].

## Next steps this code already supports

1. **Complete the image set** with `scripts/05_download_antweb.py` from a residential IP or an AntWeb bulk export: `data/dataset_full.csv` already lists all 4,354 specimens / 39 genera and every script is resumable; re-running `03_build_dataset.py --available-only` re-thresholds and would bring *Vitsika*, *Tanipone*, *Carebara* and 9 more genera into scope [README.md § Current state].
2. **Species-grouped split and an open-set reject**: group the split by species (one flag in `03_build_dataset.py`) to measure cross-species generalisation honestly, and use the nearest-neighbour similarity the API already returns (`find_similar`) as a "none of the 27" threshold; probability calibration is already done (temperature scaling on train folds, `api/probe.py`) [reports/eval_notes.md § Unseen-species effect; api/inference.py].
3. **Multi-view and multi-caste embeddings**: the harvest already holds 6,634 head and 5,127 dorsal media rows plus the queen/male records; `config.yaml` (`views_priority`, `castes`) and `06_embed.py` make view-wise embedding and late fusion a configuration change rather than a new pipeline [reports/dataset_stats.md § View distribution; config.yaml].

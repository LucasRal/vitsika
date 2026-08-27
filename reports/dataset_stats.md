# Dataset stats

## Provenance

- Target manifest (data/dataset_full.csv): **4354** images, **39** genera
- Obtained on disk: **1297** (29.8%)
  - commons: 1160
  - gbif_cache: 137
- Genera kept after >= 10 threshold on obtained specimens: **27** (vs 39 in the full manifest)

### Genera dropped for lack of obtainable images

| genus | obtained | in full manifest |
|---|---|---|
| Paratrechina | 9 | 16 |
| Carebara | 8 | 140 |
| Eutetramorium | 8 | 15 |
| Malagidris | 8 | 21 |
| Metapone | 8 | 13 |
| Proceratium | 5 | 22 |
| Adetomyrma | 4 | 30 |
| Simopone | 4 | 31 |
| Prionopelta | 3 | 25 |
| Stigmatomma | 3 | 33 |
| Vitsika | 1 | 32 |
| Tanipone | 0 | 30 |

- Images (one per specimen): **1236**
- Genera kept (>= 10 specimens): **27**
- Genera dropped: **11**
- Share of rows with coordinates: **99.4%**

## Genus names merging several genusKeys

- Carebara: merged 2 genusKeys [np.int64(1317472), np.int64(1323617)]
- Strumigenys: merged 2 genusKeys [np.int64(1320203), np.int64(1321015)]

## Media items per occurrence record (raw harvest)

- Records: 5858; mean 4.10, max 75
| media items | records |
|---|---|
| 1 | 163 |
| 2 | 840 |
| 3 | 181 |
| 4 | 3352 |
| 5 | 715 |
| 6 | 330 |
| 7 | 128 |
| 8 | 67 |
| 9 | 24 |
| 10 | 11 |
| 11 | 7 |
| 12 | 7 |
| 13 | 8 |
| 14 | 3 |
| 15 | 4 |
| 16 | 2 |
| 17 | 5 |
| 18 | 1 |
| 23 | 1 |
| 26 | 1 |
| 28 | 1 |
| 29 | 1 |
| 31 | 1 |
| 32 | 1 |
| 34 | 1 |
| 52 | 1 |
| 74 | 1 |
| 75 | 1 |

## View distribution before filtering (raw media rows)

| view | media rows |
|---|---|
| p | 7625 |
| h | 6634 |
| d | 5127 |
| l | 4618 |
| ? | 9 |

## Specimen view coverage (raw harvest)

| coverage | specimens |
|---|---|
| has profile shot | 5681 |
| dorsal only (no profile) | 41 |
| neither profile nor dorsal | 135 |

## Caste filter per genus (verbatim caste; specimens with usable views)

| genus | dropped: 1 head | dropped: male | dropped: other | dropped: queen | kept (worker) | total |
|---|---|---|---|---|---|---|
| Hypoponera | 1 | 3 | 0 | 13 | 882 | 899 |
| Camponotus | 0 | 115 | 0 | 84 | 593 | 792 |
| Pheidole | 0 | 36 | 0 | 68 | 685 | 789 |
| Tetramorium | 0 | 20 | 0 | 102 | 373 | 495 |
| Strumigenys | 0 | 18 | 0 | 38 | 331 | 387 |
| Crematogaster | 0 | 28 | 0 | 53 | 199 | 280 |
| Mystrium | 0 | 38 | 0 | 30 | 107 | 175 |
| Carebara | 0 | 6 | 0 | 10 | 140 | 156 |
| Technomyrmex | 0 | 27 | 2 | 18 | 107 | 154 |
| Leptogenys | 0 | 11 | 0 | 2 | 138 | 151 |
| Nesomyrmex | 0 | 14 | 0 | 21 | 87 | 122 |
| Monomorium | 0 | 22 | 0 | 22 | 55 | 99 |
| Tetraponera | 0 | 11 | 0 | 18 | 62 | 91 |
| Bothroponera | 0 | 34 | 0 | 6 | 47 | 87 |
| Adetomyrma | 0 | 53 | 0 | 3 | 30 | 86 |
| Pachycondyla | 0 | 20 | 0 | 17 | 37 | 74 |
| Vitsika | 0 | 14 | 0 | 22 | 32 | 68 |
| Anochetus | 0 | 12 | 0 | 8 | 39 | 59 |
| Tapinoma | 0 | 11 | 1 | 8 | 38 | 58 |
| Nylanderia | 0 | 17 | 0 | 11 | 29 | 57 |
| Aphaenogaster | 0 | 5 | 0 | 10 | 36 | 51 |
| Stigmatomma | 0 | 6 | 0 | 8 | 36 | 50 |
| Cataulacus | 0 | 4 | 0 | 12 | 31 | 47 |
| Prionopelta | 0 | 4 | 0 | 4 | 34 | 42 |
| Royidris | 0 | 5 | 0 | 7 | 29 | 41 |
| Malagidris | 0 | 11 | 0 | 6 | 21 | 38 |
| Syllophopsis | 0 | 12 | 0 | 7 | 18 | 37 |
| Platythyrea | 0 | 9 | 1 | 5 | 21 | 36 |
| Simopone | 0 | 5 | 0 | 0 | 31 | 36 |
| Terataner | 0 | 5 | 0 | 4 | 24 | 33 |
| Tanipone | 0 | 3 | 0 | 0 | 30 | 33 |
| Meranoplus | 0 | 4 | 0 | 6 | 22 | 32 |
| Odontomachus | 0 | 7 | 0 | 4 | 20 | 31 |
| Proceratium | 0 | 2 | 0 | 6 | 22 | 30 |
| Paratrechina | 0 | 7 | 0 | 2 | 16 | 25 |
| Cardiocondyla | 0 | 3 | 0 | 1 | 17 | 21 |
| Eutetramorium | 0 | 1 | 0 | 5 | 15 | 21 |
| Metapone | 0 | 3 | 0 | 5 | 13 | 21 |
| Lioponera | 0 | 4 | 0 | 0 | 15 | 19 |
| Aptinoma | 0 | 2 | 0 | 1 | 12 | 15 |
| Probolomyrmex | 0 | 3 | 0 | 3 | 8 | 14 |
| Ravavy | 0 | 6 | 0 | 2 | 5 | 13 |
| Plagiolepis | 0 | 1 | 0 | 5 | 6 | 12 |
| Pilotrochus | 0 | 2 | 0 | 2 | 6 | 10 |
| Melissotarsus | 0 | 0 | 0 | 2 | 7 | 9 |
| Solenopsis | 0 | 2 | 0 | 2 | 4 | 8 |
| Parasyscia | 0 | 4 | 0 | 2 | 2 | 8 |
| Trichomyrmex | 0 | 1 | 0 | 1 | 5 | 7 |
| Erromyrma | 0 | 2 | 0 | 2 | 2 | 6 |
| Ponera | 0 | 0 | 0 | 2 | 4 | 6 |
| Lepisiota | 0 | 2 | 0 | 1 | 1 | 4 |
| Lividopone | 0 | 1 | 0 | 1 | 2 | 4 |
| Paraparatrechina | 0 | 1 | 0 | 0 | 2 | 3 |
| Brachymyrmex | 0 | 2 | 0 | 0 | 1 | 3 |
| Leptothorax | 0 | 0 | 0 | 0 | 1 | 1 |
| Cerapachys | 0 | 1 | 0 | 0 | 0 | 1 |
| Discothyrea | 0 | 0 | 0 | 0 | 1 | 1 |
| Ooceraea | 0 | 0 | 0 | 0 | 1 | 1 |

## Images per genus per split

| genus | train | test | total |
|---|---|---|---|
| Camponotus | 190 | 48 | 238 |
| Strumigenys | 178 | 45 | 223 |
| Tetramorium | 84 | 21 | 105 |
| Crematogaster | 66 | 16 | 82 |
| Pheidole | 62 | 16 | 78 |
| Mystrium | 35 | 9 | 44 |
| Tetraponera | 34 | 9 | 43 |
| Leptogenys | 30 | 8 | 38 |
| Monomorium | 30 | 7 | 37 |
| Technomyrmex | 28 | 7 | 35 |
| Anochetus | 23 | 6 | 29 |
| Cataulacus | 23 | 6 | 29 |
| Bothroponera | 22 | 5 | 27 |
| Aphaenogaster | 22 | 5 | 27 |
| Hypoponera | 18 | 5 | 23 |
| Nylanderia | 16 | 4 | 20 |
| Platythyrea | 14 | 4 | 18 |
| Nesomyrmex | 14 | 3 | 17 |
| Terataner | 13 | 3 | 16 |
| Tapinoma | 12 | 3 | 15 |
| Syllophopsis | 11 | 3 | 14 |
| Lioponera | 11 | 3 | 14 |
| Meranoplus | 11 | 3 | 14 |
| Odontomachus | 11 | 3 | 14 |
| Cardiocondyla | 10 | 3 | 13 |
| Pachycondyla | 10 | 3 | 13 |
| Royidris | 8 | 2 | 10 |

## Genera dropped (below threshold)

| genus | specimens |
|---|---|
| Paratrechina | 9 |
| Eutetramorium | 8 |
| Carebara | 8 |
| Metapone | 8 |
| Malagidris | 8 |
| Proceratium | 5 |
| Adetomyrma | 4 |
| Simopone | 4 |
| Prionopelta | 3 |
| Stigmatomma | 3 |
| Vitsika | 1 |

## Top 15 stateProvince values

| stateProvince | images |
|---|---|
| Toamasina | 295 |
| Antananarivo | 287 |
| Antsiranana | 264 |
| Fianarantsoa | 167 |
| Toliara | 157 |
| Mahajanga | 52 |
| (none) | 8 |
| Toliary | 3 |
| Majunga | 3 |

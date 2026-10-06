# Segmentation of all 1,236 images with rembg

Every image of `data/dataset.csv` was segmented with rembg (u2net), masks
used as-is, in one run on this Mac. Each mask was checked by eye on the QA
sheets and 60 random masks were rated at full size. This report gives the
results and the list of images to leave out of the masking experiment.

## Conclusion

1. The masks are good enough for the masking experiment. 17 of 1,236 images (1.4 %) are excluded: 1 empty mask, 16 bad masks, no inverted mask.
2. On the 60-image hand check, 51 masks are good (85 %), 9 partial (15 %) and none bad. Partial almost always means lost antennae or leg tips.
3. Most failures come from the source images rather than from rembg. Six of the excluded images are detail close-ups (sculpture, setae, a leg, an SEM view), not whole ants. Seven more detail images got a mask that covers the visible part; they are kept but marked.
4. rembg keeps the mounting card or glue point in many masks. A high card residue is real card or glue in about two thirds of the cases checked; the rest is a pale ant body. In Strumigenys every mask above 15 % keeps the white glue point.
5. The run is reproducible. The 736 masks made earlier on this Mac are identical. The 500 made on the VPS differ by at most 22 pixels per million (lowest IoU 0.9997), with no flag changed.
6. The full run takes 4 minutes on this Mac (239 s wall time for 1,236 images, 8 workers).

## Settings

`scripts/27_segment_all.py`, all values in its constants.

| Setting | Value |
|---|---|
| Model | rembg 2.0.85, u2net, onnxruntime 1.30.0, CPUExecutionProvider |
| Mask | alpha >= 128, every component kept, no card or pin cleaning |
| Ant on grey | mask applied on uniform grey (128, 128, 128) |
| Ant erased | mask dilated by 15 px, filled with the median colour of a 10 px border |
| Scale bar erased | bar box from `reports/confounds/image_stats.csv`, filled the same way |
| Card residue | pixels in the mask with luminance > 150 and chroma < 60, after a 2 px opening |
| Pin residue | thin, dark (luminance < 100, chroma < 30), elongated (>= 3:1) parts touching the frame |
| Flags | empty < 0.5 % of the image, large > 80 %, frame_filling (touches all 4 edges), no_scalebar |

Outputs: five files per image in `data/segmented/<genus>/` (gitignored) and
`manifest.csv`, copied here.

## Runtime

| Run | Machine | Images | Wall time | Per image |
|---|---|---|---|---|
| Full run, 2026-10-06 19:51 to 19:55 | Apple M5 Pro, 15 CPUs, 24 GB | 1,236 | 239 s (3 min 59 s) | 0.19 s |

rembg runs one image at a time in the main process (0.19 s per image, min
0.16, max 0.78). The post-processing (derived images, residues, flags) runs in
8 worker processes; it was the slow part, at about 1.15 s per image when run
sequentially. The masks do not depend on the number of workers. Source:
`27_segment_all.log`, a single session with no pause.

Note on the VPS: the first 500 images of run 1 were made on the VPS. Its log
(`27_segment_all_part1.log`) gives 1.77 s of rembg plus 3.08 s of
post-processing per image, about 2,422 s summed over the 500 images. No wall
time was logged and the run was restarted twice, so it is not a runtime
measurement. A sequential attempt on this Mac was stopped after 300 images to
switch to parallel post-processing; the full run above started afresh, so
nothing from it remains.

## Reproducibility across machines

All run-1 masks were copied to `data/segmented_run1/` before the full run and
compared per image with the new ones (`scripts/29_segmentation_repro.py`,
output `29_segmentation_repro.csv` and `.log`).

| Run-1 masks made on | Images | Identical | Lowest IoU | Largest changed-pixel share | Flags changed |
|---|---|---|---|---|---|
| Mac (same machine) | 736 | 736 | 1.0000 | 0 | 0 |
| VPS | 500 | 339 | 0.9997 | 0.000022 | 0 |

The five lowest IoUs are all VPS masks: Camponotus casent0192117 (0.9997),
Cataulacus casent0274827 (0.9998), Cardiocondyla casent0101129 (0.9999),
Crematogaster casent0101410 and Leptogenys casent0102017 (both above 0.99995).
The first two are fragment masks of detail images, where a few pixels weigh
more. The differences are floating-point noise between CPUs; for the masking
experiment, all embeddings should still be computed on one machine.

## Flags and residues

| Flag | Images | Specimen codes |
|---|---|---|
| empty | 1 | Syllophopsis casent0291293 |
| small (< 5 %, from `28_segmentation_qa.py`) | 7 | the empty one, Camponotus casent0192117, casent0101202, Cataulacus casent0274827, Pachycondyla casent0389498, Mystrium casent0317582, Meranoplus casent0486683 |
| large | 0 | |
| frame_filling | 1 | Camponotus casent0104641, a good mask that includes the pin |
| no_scalebar | 36 | see `manifest.csv` |

No mask covers more than 75.1 % of its image (Camponotus casent0121525, a
gaster close-up), and the only frame-filling mask is good, so no mask is
inverted.

Card residue, as a share of the mask:

| Card residue | Images |
|---|---|
| under 1 % | 210 |
| 1 to 5 % | 540 |
| 5 to 10 % | 271 |
| 10 to 20 % | 161 |
| 20 to 30 % | 37 |
| 30 to 50 % | 15 |
| 50 % and more | 2 |
| **above 5 %** | **486** |

Median 3.8 %; 215 images above 10 %, 54 above 20 %. Strumigenys has the most
images above 5 % (144 of 223), then Camponotus (81 of 238) and Tetramorium (43
of 105). Every Tapinoma (15 of 15) is above 5 %, all from pale bodies or card.

Pin residue is above 5 % on 6 images: Aphaenogaster casent0107566 and
casent0178196, Camponotus casent0101383, casent0101413, casent0101430 and
casent0104640. Pins kept in the mask were seen on the sheets for 0101413 and
0101430; Aphaenogaster 0107566 is a bad mask whose long hind leg reaches the
frame. The other three were not checked at full size.

## Is high card residue real card or a pale ant?

All 54 images above 20 % were checked on the sheets or at full size.

| What the residue is | Images | Examples |
|---|---|---|
| Real card or glue kept in the mask | 36 | all 11 Strumigenys (casent0005591 39 %, casent0005510 35 %, casent0005498 31 %), Pheidole casent0101829 40 % and casent0104595 38 %, Crematogaster casent0102938 37 %, Cardiocondyla casent0101218 32 %, Leptogenys casent0101727 31 %, Nesomyrmex casent0101687 30 % |
| Pale ant body, no card | 16 | Tapinoma casent0453961 61 % and casent0008659 59 %, Royidris casent0002257 37 %, Technomyrmex casent0485542 29 %, Monomorium casent0076204 26 %, the cleared Crematogaster casent0101771 20 % |
| Other | 2 | Camponotus casent0209055 (white background between legs and highlights), Syllophopsis casent0291293 (the empty mask) |

Strumigenys are mounted on a white glue point that rembg keeps as part of
the ant. On its 12 sheets, every mask above 15 % (22 images) keeps the glue
point; below 10 % the residue is mostly pale legs, hairs and highlights. On
the 60-image hand check, card or glue is kept on 31 masks. The measure also
misses tan card: Hypoponera casent0101016 keeps a large card block and scores
2 %. So "ant on grey" often means "ant and its mount on grey", and the card
residue is a lower bound on how often that happens.

## Hand check

60 images drawn by `28_segmentation_qa.py` (seed 42, `qa/handcheck_1-3.jpg`),
each rated at full size from the original, the ant on grey and the mask
outline. Ratings and notes are in `handcheck.csv`.

| Rating | Meaning | Images | Share |
|---|---|---|---|
| good | whole ant kept | 51 | 85 % |
| partial | ant parts missing | 9 | 15 % |
| bad | ant mostly missing, or a head, mesosoma or gaster lost | 0 | 0 % |

The 9 partial masks lose raised antennae (Camponotus casent0101118,
casent0101441, casent0101517, casent0101893, Platythyrea casent0497789), legs
(Camponotus casent0101180, Tetramorium casent0101279), or both (Tetramorium
casent0101141). Odontomachus casent0101015 loses all six legs and
both antennae. With 0 bad of 60, the bad rate is below about 5 % (95 %
upper bound); the sheet review found 1.4 %.

## Failure examples

From the review of all 73 genus sheets and full-size checks of 40 suspects
(`exclude.csv`, column `source` = sheet).

| Failure | Specimen codes |
|---|---|
| Detail image, fragment or empty mask | Syllophopsis casent0291293 (two legs, empty), Camponotus casent0192117 (head-mesosoma joint), Cataulacus casent0274827 and Meranoplus casent0486683 (sculpture), Pachycondyla casent0389498 (SEM), Mystrium casent0317582 (seta tip) |
| Close-up of a large ant, fragment mask | Camponotus casent0101202 |
| Long-legged ant on a pin, body broken up | Aphaenogaster casent0101069 (mesosoma lost), casent0101070 (only gaster), casent0101072 (head lost), casent0107566 (head lost), casent0101087 (gaster lost) |
| Ant on card, gaster lost | Odontomachus casent0102019, casent0102021 |
| Cropped close-up, body part lost | Camponotus casent0145284 (head), Crematogaster casent0423124 (gaster), Pheidole casent0235036 (head and mesosoma) |
| Pale legs lost | Pheidole casent0101745 (partial) |
| Glue blob around a flattened specimen | Cardiocondyla casent0101128 (partial) |

Aphaenogaster is the genus most affected: 5 of 27 images excluded. Detail
images with a mask that covers the visible part are kept: Aphaenogaster
casent0822320 (leg), Camponotus casent0121525 (gaster), Crematogaster
casent0054136 (body segment) and casent0193045 (petiole), Pachycondyla
casent0300904 (gaster), Pheidole casent0040756 (head), Syllophopsis
casent0274442 (gaster).

## Images to exclude

`exclude.csv` has 52 rows, one per image that is excluded, partial or a
detail image, with genus, split, image path, mask share and a note.

| Column | Images | Use |
|---|---|---|
| `exclude` = 1 | 17 (12 train, 5 test) | leave out of the masking experiment: 1 empty (`exclude_reason` = empty mask) and 16 bad |
| `partial` = 1 | 28 (22 train, 6 test) | keep; drop in a sensitivity check |
| `detail_image` not empty | 13 | the part shown; 6 are among the excluded, 7 kept |

Excluded per genus: Aphaenogaster 5 of 27, Camponotus 3 of 238, Odontomachus
2 of 14, and one each in Cataulacus, Crematogaster, Meranoplus, Mystrium,
Pachycondyla, Pheidole and Syllophopsis.

The partial list is not complete. It holds the 9 partial masks of the hand
check and the 19 confirmed among the sheet suspects. The sheets were read at
tile size, where a lost antenna or leg tip is easy to miss. Going by the hand
check, about 15 % of all masks (some 185 images) are partial, so a
sensitivity check on this column tests the worst partial masks, not all of
them.

## Files

| File | Content |
|---|---|
| `manifest.csv` | one row per image: mask share, components, border edges, residues, scale-bar box, flags, timings |
| `27_segment_all.log` | the full run |
| `27_segment_all_part1.log` | the VPS part of run 1 |
| `28_segmentation_qa.log`, `qa/` | 73 genus sheets, `worst40.jpg`/`.csv`, the hand-check sheets |
| `29_segmentation_repro.csv`, `.log` | run 1 against the full run, per image |
| `handcheck.csv` | the 60 hand-check images with rating and note |
| `exclude.csv` | images to exclude, partial and detail images |

Reproduce: `.venv/bin/python scripts/27_segment_all.py`, then
`scripts/28_segmentation_qa.py` and `scripts/29_segmentation_repro.py`. No
network is needed; the u2net model is cached in `~/.rembg/models/`.

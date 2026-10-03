# Segmentation trial: SAM 2 against rembg on five images

A dry run before segmenting the whole dataset. Five images from
`data/dataset.csv`, five genera, drawn with seed 7 (`selection.csv`):
Camponotus is forced as the large ant and one of Strumigenys / Syllophopsis
as the small one (the seed picked Syllophopsis); the other three genera are
random. Nothing in `data/` was touched. Script: `scripts/24_segmentation_trial.py`.

| # | Genus | Specimen | Split | Setup |
|---|---|---|---|---|
| 1 | Camponotus | casent0102441 | train | large black ant, card point under the body, dark pin top right |
| 2 | Syllophopsis | casent0044971 | train | small pale ant, wide out-of-focus pin behind it |
| 3 | Odontomachus | casent0005967 | test | slender dark ant, thin pin, white card point at the petiole |
| 4 | Terataner | casent0101703 | train | black ant lying on a card that fills the frame, glue around it |
| 5 | Pheidole | casent0101624 | train | small pale ant, large card point filling the right third |

## Tools

- **SAM 2**: `sam2` 1.1.0 (pip), checkpoint `sam2.1_hiera_tiny.pt` (the
  smallest, 39 M parameters, 156 MB, sha256
  `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`),
  CPU, `multimask_output=False`. One image embedding per image, then one
  prediction per prompt.
- **rembg**: 2.0.85 with its default `u2net` model through onnxruntime 1.30.0
  on CPU, no prompt. Its alpha above 0.5 counts as ant.

Three SAM 2 prompt versions per image:

- **A, automatic box** as briefed: the bounding box of the pixels that are
  darker than 75 % of the background luminance *or* more saturated than 40
  (the second test is needed because the pale ants are not dark: the
  dark-pixel rule alone boxes the pin instead of the Syllophopsis), after
  removing specks and excluding the bottom-right corner and the detected
  scale bar with its label.
- **B, box + points placed by hand** after looking at A: negative points on
  the pin and card point, plus one positive point on the mesosoma. The
  positive point is not in the brief; it was added because with a
  near-full-frame box SAM 2 returned the background instead of the ant on
  three images, and negatives cannot undo that. The negatives-only version
  (B0) was run and recorded as well.
- **C, automatic prompt from rembg**: box of the rembg matte plus a positive
  point at its thickest part. Not in the brief; added as the obvious
  automatic alternative once A had failed.

Each contact sheet `<k>_<genus>_<specimen>.jpg` has one row per version:
original | SAM 2 mask overlay (box orange, positives green, negatives red) |
SAM 2 ant only on grey | rembg ant only on grey; row B ends with the rembg
alpha matte and row C with a map of where SAM 2 (C) and rembg disagree.
Binary masks are in `masks/`.

## Runtime per image

Wall-clock seconds on this machine's 6-core CPU (`runtime.csv`; the machine
also runs the API, so timings move by up to 2x between runs).

| # | Genus | SAM 2 image embedding | SAM 2 per prompt | rembg |
|---|---|---|---|---|
| 1 | Camponotus | 2.33 | 0.04 to 0.40 | 0.55 |
| 2 | Syllophopsis | 2.00 | 0.06 to 0.08 | 0.54 |
| 3 | Odontomachus | 1.67 | 0.05 to 0.09 | 0.51 |
| 4 | Terataner | 1.92 | 0.04 to 0.16 | 0.55 |
| 5 | Pheidole | 1.84 | 0.04 to 0.06 | 1.59 |
| mean | | 1.95 | 0.15 | 0.75 |

Across the three runs made while writing this, the embedding took 1.7 to
3.8 s and rembg 0.4 to 1.6 s. Model load: SAM 2 1.3 s, rembg 0.9 s, once.
For all 1,236 images: about 45 minutes for SAM 2, about 15 for rembg, and
the two together under an hour, since C reuses the one embedding.

## Result per image

Judged by eye on the contact sheets. "Intact" means the part is in the
mask and not broken into speckles.

| # | Genus | Version | Antennae | Legs | Mandibles | Leaked in |
|---|---|---|---|---|---|---|
| 1 | Camponotus | A box | intact | intact | intact | card point in patches, pin top as speckles |
| | | B box + points | intact | mid and hind legs broken into speckles | intact | card mostly removed, some speckles |
| | | rembg | intact | intact | intact | the whole pin above the thorax, to the top edge |
| | | C rembg prompt | intact | intact | intact | card point in patches, pin top partly |
| 2 | Syllophopsis | A box | **mask is the background**, ant removed | | | |
| | | B box + points | gaster only, 7 % of the image | | | |
| | | rembg | intact | all six intact to the tarsi | intact | small glue spot under the thorax; pin removed |
| | | C rembg prompt | intact | all six intact | intact | nothing; pin removed. Cleanest mask of the trial |
| 3 | Odontomachus | A box | **mask is the background** | | | |
| | | B box + points | 82 % of the image, wrong | | | |
| | | rembg | intact | intact, thin | intact | part of the white card point; pin removed |
| | | C rembg prompt | intact | intact | intact | the white card point; pin removed |
| 4 | Terataner | A box | intact | intact | intact | a large part of the card (the card is the whole background here) |
| | | B box + points | intact | intact | intact | card patches top right; the gaster is partly missing |
| | | rembg | intact | intact | intact | the translucent glue under the ant; scale bar removed |
| | | C rembg prompt | intact | intact | intact | a little glue under the gaster; most glue removed. Best mask for this image |
| 5 | Pheidole | A box | **mask is the background** | | | |
| | | B box + points | mesosoma only, plus background patches | | | |
| | | rembg | intact | intact | intact | the whole card point, about half the mask |
| | | C rembg prompt | intact | intact | intact | the whole card point, as rembg |

Scale bar: A keeps the bar and label on 3 of 5 images (75 to 86 % of the
bar region in the mask); rembg and C leave it out on all five (0 to 7 %,
the Camponotus region is confounded by a leg tip resting on it).

Negatives only (B0, as briefed, no positive point): Camponotus loses the
mid and hind legs (score 0.21), Terataner flips to the card, and the three
images that A had inverted stay inverted (46 to 66 % of the frame). Not
shown on the sheets; masks in `masks/*_B0_*.png`, numbers in `runtime.csv`.

## What this says

1. **The automatic dark-pixel box does not work on these photos.** The pin
   and the card point reach the image edge in every image, so the box is
   the full frame (0 to 95 % or more of width and height on all five) and
   carries no information. Given a full-frame box, SAM 2 tiny returns the
   background on 3 of 5 images and the ant plus card on the other two.
2. **Hand-placed points do not rescue it.** With the same full-frame box,
   negatives alone or with one positive give erratic masks: a single body
   part, the background, or a speckled ant. The tiny model needs a tight
   box, not more points.
3. **rembg alone is the strongest automatic result**: the ant is complete,
   antennae, legs and mandibles included, on all five images, in under a
   second, with no prompt. Its failures are the attached objects: the pin
   in Camponotus, the card point in Pheidole (whole) and Odontomachus
   (part), the glue in Terataner.
4. **SAM 2 prompted from the rembg matte (C) matches rembg** (IoU 0.86 to
   0.96) and improves the edges and the glue in Terataner, but it inherits
   rembg's box, so it keeps the same card point in Pheidole and
   Odontomachus. It costs 2 s more per image.
5. **The card point is the open problem**, not the ant. It is pale, large,
   touches the ant, and neither model separates it when it is in the
   mask. The pin is removed by rembg in 4 of 5 images.

Recommendation for the full run: do not use the dark-pixel box. Run rembg
on every image (15 minutes), take its matte as the mask or as the prompt
for SAM 2, and add a step for the card point and pin: for example an
automatic negative point on any large pale low-saturation blob inside the
rembg mask, or a check of the mask's share of near-white pixels to flag
images for review. Whether to spend the extra 2 s per image on SAM 2 C
depends on whether edge quality matters for the downstream use; on these
five it changed little.

## Limitations

- Five images, one view (profile), all by one photographer and all
  Commons originals; none of the 116 square GBIF-cache crops, where the ant
  fills the frame and the box problem may be different.
- The judgements are by eye on 640 px panels, not against hand-drawn
  ground truth. Thin tarsi and antennal tips can look intact when a few
  pixels are missing.
- Only the smallest SAM 2 checkpoint and the default rembg model were
  tried. rembg's `isnet-general-use` or `birefnet` models and SAM 2 small
  or base may behave differently, at a cost in time and downloads.
- The point prompts of version B were placed by hand on these five images
  and are not an automatic method.
- Timings are from a shared machine and vary by up to 2x between runs.

## Reproduce

```bash
# one-off installs (network): SAM 2 built against the venv's CPU torch, rembg with onnxruntime
SAM2_BUILD_CUDA=0 .venv/bin/pip install --no-build-isolation -r requirements-segmentation.txt
curl -L -o ../mg-ants-cache/sam2/sam2.1_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
# rembg downloads u2net.onnx (176 MB) to ~/.rembg/models/ on first use
.venv/bin/python scripts/24_segmentation_trial.py        # about 1 minute
```

The checkpoint path can be overridden with `SAM2_CHECKPOINT`. The masks and
sheets are deterministic; only the timings change between runs.

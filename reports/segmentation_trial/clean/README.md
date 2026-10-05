# Second step: card point and pin removal from the rembg masks

Follow-up to the trial in `..`: rembg is the primary segmenter, this step
cleans its mask. Same five images, starting from `../masks/<specimen>_rembg_alpha.png`
(alpha > 0.5). Script: `scripts/25_clean_mask.py`. Nothing in `data/` was
touched. Not run on the full dataset.

Contact sheets `<k>_<genus>_<specimen>.jpg`: original | rembg mask | removed
regions (card yellow, pin red, dropped pieces magenta; pale blobs that were
kept outlined green) | final ant only on grey 128. Masks in `masks/`,
per-candidate features in `candidates.csv`, tests within 10 % of a threshold
in `margins.csv`, per-image counts in `summary.csv`.

Second pass (after the first review): the SAM 2 refinement was dropped (it
was never kept) and a shaded-card growth step was added after a card is
accepted. The 30-image measurement run that followed is in `../run30/`.

## Rules and thresholds

Lengths in pixels at 1024 px width, scaled by width / 1024.

**Card point.** Candidate pixels: inside the mask, luminance > 150, chroma
(max RGB minus min RGB) < 95; opened by 2 px; connected components. Two
tiers by the component's median chroma, because the white card of the
Odontomachus image (chroma 14) and the warm-lit cream card of the Pheidole
image (chroma 75) look different, and the cream one overlaps the colour of
pale yellow ant parts (chroma 65 to 90):

| Test | White tier, chroma < 50 | Cream tier, chroma 50 to 95 |
|---|---|---|
| Area, share of the image | >= 0.1 % | >= 1 % |
| Largest inscribed disc radius (thick, not spindly) | >= 8 px | >= 18 px |
| Solidity, area over convex hull area | >= 0.45 | >= 0.60 |
| Straight edge: most outline pixels on one line (3 px bins, 0.5 degree steps) | >= 50 px | >= max(70 px, 0.6 x sqrt(area)) |
| Share of the outline within 4 px of the background | >= 0.15 | >= 0.15 |

The straight-edge test is what keeps pale ant parts: a gaster or femur has
a curved outline, a card has cut edges. The background test keeps pale
bands on a gaster, which are enclosed by ant pixels. Accepted components
are grown by 3 px into neighbouring pale mask pixels, then into the shaded
card: connected mask pixels whose hue is within 15 degrees of the card's
mean hue (any hue when either is nearly grey, chroma < 15), whose chroma is
at most the card's median + 20 and whose luminance is >= 110, never into the
body core (the opening used by the pin rule). Pale blobs that fail are left
in place and counted ("glue" in the summary; most of them are ant
highlights, hairs and coxae, not glue).

**Pin.** Body core = mask opened with a disc of radius 40 px (head,
mesosoma, gaster). Thin parts = mask minus core, opened by 3 px to drop
hairs. A thin component is a pin when: length >= 100 px, width (area /
length) <= 80 px, length / width >= 3, straight (>= 90 % of its pixels
within 0.75 width of the fitted axis), even width (coefficient of variation
of the width along the axis <= 0.25, ends excluded), touches the image
border, median luminance <= 130 and median chroma <= 25. Rule for the body:
only thin-part pixels are removed, so where the pin passes behind or
through the body the core is never touched. The Camponotus pin is 57 px
wide in the rembg mask (blur halo included), which is why the core radius
is 40 and not smaller.

**Keep.** After removal, the largest connected component plus every
component reachable from it through gaps <= 6 px (repeated dilation).
Everything else is dropped and counted.

## Result per image

| # | Genus | Card point | Pin | Ant pixels wrongly removed | Pixels removed |
|---|---|---|---|---|---|
| 1 | Camponotus | **not removed**: rembg had already cut most of it; the slivers left between the legs (4,551 px, white tier) fail the background test (0.05) because they are enclosed by legs and shaded card. The growth step cannot reach them: it only grows from an accepted card, and there is none | **fully**: the 11,473 px band above the thorax to the top edge; the part behind the body is untouched by rule | none seen; antennae, legs, mandibles intact | 3.4 % of the mask |
| 2 | Syllophopsis | nothing to remove (rembg mask was clean); nothing removed | already absent from the rembg mask | none; 31 pale blobs (coxae, femora, highlights) all correctly kept | 0 |
| 3 | Odontomachus | **fully**: the white triangle at the petiole, 2,324 px (1,708 pale, 616 added by growth) | already absent | none; the pale bands on the gaster (2,880 px) are kept, by a thin margin (background share 0.13 against the 0.15 threshold); growth did not enter the gaster (it is body core); all legs intact | 1.3 % |
| 4 | Terataner | no card point; the glue under the ant (11,006 px) fails the straight-edge test by 2 px (68 against 70) and stays, which the brief allows | none in frame | none | 0 |
| 5 | Pheidole | **fully**: 66,511 px removed (56,422 pale, 10,089 added by growth); the shaded strip along the ant's underside is now gone | already absent | none seen; the hind leg lying on the card is intact to the tarsus (chroma 97 to 133, above the growth cap of 75 + 20); 2,286 px of card fragments dropped as disconnected pieces | 31 % |

So the shaded-card growth fixes the Pheidole strip, leaves the Odontomachus
gaster bands and every leg intact, and does nothing for the Camponotus
slivers, which have no accepted seed. Rules take 7 to 12 s per image, mostly
the Hough edge test on 20 to 30 candidates.

## Thin margins on these five

From `margins.csv` (tests within 10 % of their threshold; "decisive" means
the outcome would flip): the Terataner glue, straight edge 68 against 70
(3 %), and an Odontomachus white blob of 3,248 px that fails only the radius
test, 7.3 against 8 (9 %). Nothing accepted is within 10 % of a threshold:
the Odontomachus card's nearest test is the straight edge (70 against 50),
the Pheidole card's is the background share (0.17 against 0.15, 13 %), the
Camponotus pin's is the straightness (1.00 against 0.90).

## Reproduce

```bash
.venv/bin/python scripts/25_clean_mask.py             # about 1 minute, deterministic
```

Needs rembg's packages from `../README.md` (SAM 2 is no longer used here).

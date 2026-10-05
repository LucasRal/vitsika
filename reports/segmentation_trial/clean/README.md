# Second step: card point and pin removal from the rembg masks

Follow-up to the trial in `..`: rembg is the primary segmenter, this step
cleans its mask. Same five images, starting from `../masks/<specimen>_rembg_alpha.png`
(alpha > 0.5). Script: `scripts/25_clean_mask.py`. Nothing in `data/` was
touched. Not run on the full dataset.

Contact sheets `<k>_<genus>_<specimen>.jpg`: original | rembg mask | removed
regions (card yellow, pin red, dropped pieces magenta; pale blobs that were
kept outlined green) | final ant only on grey 128 | SAM 2 refinement and its
verdict. Masks in `masks/`, per-candidate features in `candidates.csv`,
per-image counts in `summary.csv`.

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
are grown by 3 px into neighbouring pale mask pixels. Pale blobs that fail
are left in place and counted ("glue" in the summary; most of them are ant
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

**SAM 2 refinement.** sam2.1 tiny, box of the clean mask padded 2 %, up to
3 positive points at the thickest spots of the clean mask (>= 150 px
apart), one negative point at the thickest spot of each removed region.
Kept only if IoU with the clean mask > 0.85 and it has fewer connected
components (>= 30 px).

## Result per image

| # | Genus | Card point | Pin | Ant pixels wrongly removed | Pixels removed |
|---|---|---|---|---|---|
| 1 | Camponotus | **not removed**: rembg had already cut most of it; the slivers left between the legs (4,551 px, white tier) fail the background test (0.05) because they are enclosed by legs and shaded card | **fully**: the 11,473 px band above the thorax to the top edge; the part behind the body is untouched by rule | none seen; antennae, legs, mandibles intact | 3.4 % of the mask |
| 2 | Syllophopsis | nothing to remove (rembg mask was clean); nothing removed | already absent from the rembg mask | none; 31 pale blobs (coxae, femora, highlights) all correctly kept | 0 |
| 3 | Odontomachus | **fully**: the white triangle at the petiole, 1,931 px | already absent | none; the pale bands on the gaster (2,880 px) were taken on an earlier pass and are now kept, by a thin margin (background share 0.13 against the 0.15 threshold) | 1.0 % |
| 4 | Terataner | no card point; the glue under the ant (11,006 px) fails the straight-edge test by 2 px (68 against 70) and stays, which the brief allows | none in frame | none | 0 |
| 5 | Pheidole | **mostly**: 63,135 px removed; a strip of shaded card next to the ant's underside stays because its luminance is below 150 | already absent | none seen; the hind leg lying on the card is intact; 2,300 px of card fragments dropped as disconnected pieces | 29 % |

SAM 2 refinement: never kept. IoU with the clean mask 0.70 (Pheidole), 0.87, 0.91, 0.93, 0.96; it always
had the same number of components or more. With the literal rule "fewer
components" it can never be kept when the clean mask is one piece, which it
is on 4 of 5 images. The rules alone took 5 to 9 s per image, mostly the
Hough edge test on 20 to 30 candidates; SAM 2 added 2 to 4 s.

## Ready for a 30-image check?

Not yet. Two things to adjust first, then run seed 11 as a measurement run,
not a validation:

1. **Shaded card.** The pale test is absolute (luminance > 150), so the
   darker part of a card next to the ant escapes it: the Pheidole strip and
   the Camponotus slivers. Grow each accepted card into connected mask
   pixels of similar hue down to luminance about 110, stopping at the body
   core. This is the main cause of partial removal.
2. **Thin margins.** The background-share threshold sits between the
   Odontomachus gaster bands (0.13) and the Pheidole card (0.17); the cream
   straight-edge threshold sits 2 px from the Terataner glue. Both were set
   on these five images and will move on 30. Record the features for every
   candidate (already in `candidates.csv`) and pick thresholds from the 30
   before trusting them.

What the 30-image run would test that these five cannot: the cream tier
needs >= 1 % of the image, so small cream cards will be missed; the pin rule
has met one pin, and its border and colour tests are untested on dark-legged
ants whose legs reach the frame edge; GBIF-cache crops (square, ant filling
the frame) were not in the five. The failure mode so far is "not removed"
rather than "ant removed", which is the safer side.

If the SAM 2 step is kept at all, change its rule to "no more components"
and look at whether it ever beats the rules; on these five it did not.

## Reproduce

```bash
.venv/bin/python scripts/25_clean_mask.py             # about 1 minute; SAM2_SKIP=1 to skip the refinement
```

Needs the SAM 2 checkpoint and packages from `../README.md`. Deterministic.

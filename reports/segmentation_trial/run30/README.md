# Measurement run: rembg + card/pin cleaning on 30 images

Thirty images from `data/dataset.csv`, seed 11, all 27 genera, 4 GBIF-cache
crops and 7 images by photographers other than April Nobile
(`selection.csv`: 3 GBIF crops and 2 other-photographer images were forced
from distinct genera, then one random image per remaining genus, then 3
extra at random). rembg (u2net, CPU, 0.69 s per image) followed by
`scripts/25_clean_mask.py` with the thresholds of the five-image pass in
`../clean/`, including the shaded-card growth. Script:
`scripts/26_segmentation_run30.py`. Nothing in `data/` was touched. No
threshold was re-tuned on these 30.

Files: `<k>_<genus>_<specimen>.jpg` contact sheets (original | rembg mask |
removed regions, card yellow, pin red, dropped pieces magenta, kept pale
blobs outlined green | final ant only on grey 128); `summary_sheet.jpg`, the
30 final results as thumbnails; `candidates.csv`, every card and pin
candidate (678 rows: 472 card candidates, 313 cream tier and 159 white; 206
pin candidates) with area, inscribed radius, solidity, straight-edge
length, background share, median chroma and luminance, elongation, the pin
features, the tests failed, and an empty `true_label` column to fill by
hand (card / pin / glue / ant_part); `margins.csv`, every test within 10 %
of its threshold; `summary.csv`, pixels removed per image and which frame
edges the final mask touches; `masks/`, the rembg alpha and the card, pin
and clean masks.

## Result per image

Judged by eye on the sheets and on full-resolution crops of every removed
region. "Card in mask" is whether rembg had left any card in its mask; a
card outside the mask needs no removal. "Ant removed" is any ant pixel
taken by the cleaning step, which the brief makes a blocker.

| # | Genus (source, photographer) | Card in mask | Card removed | Pin removed | Ant removed | Small cream card below the 1 % floor | Ant reaches the frame edge |
|---|---|---|---|---|---|---|---|
| 1 | Anochetus | white blob under the coxae | fully, 16,956 px | no pin in mask | **yes, blocker**: the pale base of the fore leg lying over the card (femur base and trochanter) was in the pale component; the leg is now a separate piece | no | no |
| 2 | Aphaenogaster | no (the card is the whole background) | n/a | **not removed**: the pale grey pin above the thorax (40,547 px) fails width 101 > 80, straightness 0.89 and chroma 48 | **yes, blocker**: three leg tips (1,364 px) that rembg had already cut off on the card were dropped by the keep rule | no | top (the pin) |
| 3 | Bothroponera | no | n/a | no pin in mask | no | no | no |
| 4 | Camponotus | yes, under the body | fully, 17,056 px | **fully** for the band above the thorax (6,871 px) | **yes, blocker**: the whole fore leg below the head (25,248 px with the pin behind it) was accepted as a second pin: dark, straight 0.92, even width, touching the left edge | no | left (the removed leg) |
| 5 | Camponotus (GBIF crop) | yes, the card fills the bottom of the crop | **not removed**: the 120,001 px cream candidate fails solidity 0.455 and background share 0.08 (cut by the frame, bordered by the ant) | **partially**: the top part (18,406 px); the stub between thorax and card, inside the body-core radius, stays | no | no (large) | left, right (the ant fills the crop) |
| 6 | Cardiocondyla | yes, large white card | fully, 84,112 px | no pin in mask | **yes, blocker**: the mid and hind legs lying over the card are pale (luminance > 150) and were part of the pale component | no | no |
| 7 | Cataulacus | yes, a shaded strip under the ant | **not removed**: the strip is below luminance 150; the pale pieces are all under 1,100 px | yes, the strip | no | no |
| 8 | Cataulacus (Nick Olgeirson) | yes, large cream card | fully, 53,730 px; the dropped 821 px are scale-bar fragments | no pin in mask | no | no | no |
| 9 | Crematogaster (GBIF crop, Bonnie Blaimer) | no | n/a | n/a | no | no | left, right |
| 10 | Hypoponera | yes, cream card under the ant | **not removed**: 7,661 px (1.3 %), fails solidity only, 0.564 against 0.60 | no pin in mask | no | no, just above the floor | no |
| 11 | Leptogenys | yes, a small white card at the coxa | **not removed**: 3,061 px white tier, fails the straight edge only, 43 against 46 | no pin in mask | no | small white card (0.4 %) | no |
| 12 | Lioponera | yes, cream card under the ant | **not removed**: two pieces, 11,474 px (fails radius and solidity) and 6,171 px (0.8 %, fails area and radius) | pin below the card is outside the mask | no | yes | no |
| 13 | Meranoplus (Nick Olgeirson) | yes, white card under the ant | **not removed**: 11,825 px, fails solidity only, 0.591 against 0.60 | no pin in mask | no | no | left (head) |
| 14 | Monomorium | yes, translucent card under the thorax | fully, but 4 regions accepted | no pin in mask | **yes, blocker, the worst case**: the fore leg (1,646 px pale + 2,100 grown), the mid leg (1,476 + 8,778 grown) and the gaster tip (1,560 px) were accepted as white cards; the hind leg, cut from the body, was dropped (most of the 19,551 px dropped) | no | no |
| 15 | Mystrium | no (pin behind the body is outside the mask) | n/a | n/a | no | no | no |
| 16 | Nesomyrmex (Erin Prado) | yes, round white card under the thorax | **not removed**: 21,620 px, fails solidity only, 0.432 against 0.45 | no pin in mask | no | no | bottom (leg) |
| 17 | Nylanderia | yes, large card under the ant | fully, 95,338 px; dropped 1,373 px are scale-bar fragments | no pin in mask | **yes, blocker, minor**: the growth climbed a few hundred pixels into the pale underside of the hind coxa; legs intact | no | no |
| 18 | Odontomachus | no | n/a | pin above the head is outside the mask | no | no | no |
| 19 | Pachycondyla (GBIF crop, Wade Lee) | no | n/a | n/a | **yes, blocker**: the gaster apex with its pale hairs (1,162 + 355 px) was accepted as a white card | no | no |
| 20 | Pheidole (Nick Olgeirson) | yes, shaded strip under head and mesosoma | **not removed**: the pale pieces are 2,629 and 3,184 px (0.4 %) | pin is outside the mask | no | yes | no |
| 21 | Platythyrea | yes, a strip of the card the ant lies on | **not removed**: 6,487 px (0.8 %), fails area and radius | no pin in mask | no | yes | no |
| 22 | Royidris | no | n/a | **not removed**: the dark pin head under the mesosoma sits inside the body core, so it is never a thin-part candidate | no | no | no |
| 23 | Strumigenys (GBIF crop) | the translucent card between the front legs is mostly outside the mask | nothing to remove | n/a | **yes, blocker, trivial**: three hair tips (401 px) that rembg had left disconnected above the head and mesosoma were dropped by the keep rule | no | bottom, left, right |
| 24 | Syllophopsis | yes, translucent card under the thorax | fully, 13,344 px | pin below the gaster is outside the mask | **yes, blocker, minor**: pale pieces of the coxae and trochanters touching the card were in the pale component (a few hundred pixels) | no | no |
| 25 | Tapinoma | no | n/a | **not removed**: the dark pin head under the thorax is inside the body core | **yes, blocker**: a 760 + 206 px pale patch on the upper edge of the gaster was accepted as a white card, exactly at the area floor (760 against 759) and the radius floor (8.0 against 8) | no | no |
| 26 | Technomyrmex | yes, cream card under the gaster | fully, 24,136 px; gaster underside intact | no pin in mask | no | no | no |
| 27 | Technomyrmex | no (white card behind the head is outside the mask) | n/a | n/a | no | no | no |
| 28 | Terataner | no (the card is the whole background) | n/a | n/a | no; the 205 px dropped is a pale glue speck beside the head, disconnected in the rembg mask | no | no |
| 29 | Tetramorium | yes, white card between the legs | **not removed**: 3,615 px, fails the background test (0.0: enclosed by legs and body) | no pin in mask | no | no (white tier) | no |
| 30 | Tetraponera (Erin Prado) | no (pin behind the body is outside the mask) | n/a | n/a | no | no | no |

Totals. Card in the rembg mask on 18 images: removed fully on 8 (1, 4, 6,
8, 14, 17, 24, 26), of which 5 took ant pixels with it; not removed on 10
(5, 7, 10, 11, 12, 13, 16, 20, 21, 29). Pin in the mask on 5 images: fully
removed on 1 (4), partially on 1 (5), not removed on 3 (2, 22, 25). Ant
pixels removed on **10 images (blockers: 1, 2, 4, 6, 14, 17, 19, 23, 24, 25)**,
seven of them legs, coxae or gaster parts and three of them pieces rembg had
already cut off (leg tips on 2, hair tips on 23).
Small cream cards below the 1 % floor on 4 images (7, 12, 20, 21), plus a
small white one (11). The final mask reaches a frame edge on 7 images (2,
4, 5, 9, 13, 16, 23); on 5 of these it is legs or the ant filling a GBIF
crop, on 2 the removed leg (4) or the pin (2). Removal share of the rembg
mask: 0 on 11 images, under 1 % on 10, 4.8 to 33 % on the 9 where a card or
pin was taken.

On the 8 images where rembg had left neither card nor pin in its mask (3, 9,
15, 18, 23, 27, 28, 30) the rules removed nothing but disconnected specks,
at most 427 px.

## What the blockers have in common

- **Pale ant parts connected to a card are in the pale component.** The
  card candidate is "mask pixels above luminance 150", and a pale leg or
  coxa lying on the card is above 150 too, so it is removed as part of the
  card (1, 6, 24) or as a card of its own (14). The shaded-card growth
  added little of this damage (Monomorium mid leg, Nylanderia coxa); the
  seed did most of it.
- **The white tier's small area floor (0.1 %) accepts pale patches on the
  ant** when they have a short straight run and touch the background: the
  gaster apex of Pachycondyla (19), a gaster patch on Tapinoma (25), the
  gaster tip and two legs of Monomorium (14). All are 760 to 1,650 px.
- **The pin rule has no shape test that separates a dark leg from a pin**
  when the leg is straight, even and reaches the frame (4). Its straightness
  0.92 is 2.6 % above the threshold.
- **The keep rule drops ant pieces that rembg had already cut off** (2:
  leg tips on a card; 23: hair tips; 6 px bridging does not reach them).

And the misses: cards under the ant that are shaded (luminance below 150:
7, 20), or cut by the frame (5), or small (12, 21), or enclosed by legs (29),
or just under a shape threshold (10, 13, 16, 11).

## Candidates within 10 % of a threshold

`margins.csv` lists 416 (candidate, test) rows within 10 % of a threshold.
Most (396) are on candidates that fail two or more tests, where one near
miss changes nothing; they are mostly small pin candidates near the
straightness threshold. The 20 **decisive** rows, where the outcome would
flip if the threshold moved past the value, are these. "Right" means the
current outcome is the one wanted, so the margin is a risk; "wrong" means a
move of under 10 % would fix the case.

| # | Genus | Candidate | Test | Value | Threshold | Distance | Current outcome |
|---|---|---|---|---|---|---|---|
| 25 | Tapinoma | gaster patch, 760 px, white | area | 760 px | 759 px | 0.2 % | accepted, wrong |
| 25 | Tapinoma | same | radius | 8.00 | 8 | 0 % | accepted, wrong |
| 25 | Tapinoma | same | straight edge | 54 | 50 | 8 % | accepted, wrong |
| 14 | Monomorium | fore leg, 1,646 px, white | solidity | 0.455 | 0.45 | 1.2 % | accepted, wrong |
| 14 | Monomorium | mid leg, 1,476 px, white | straight edge | 53 | 50 | 6 % | accepted, wrong |
| 14 | Monomorium | same | solidity | 0.489 | 0.45 | 8.8 % | accepted, wrong |
| 19 | Pachycondyla | gaster apex, 1,162 px, white | straight edge | 52 | 50 | 4 % | accepted, wrong |
| 4 | Camponotus | fore leg + pin, 25,248 px | straightness | 0.923 | 0.90 | 2.6 % | accepted as pin, wrong |
| 13 | Meranoplus | card, 11,825 px, cream | solidity | 0.591 | 0.60 | 1.5 % | rejected, wrong |
| 16 | Nesomyrmex | card, 21,620 px, white | solidity | 0.432 | 0.45 | 4 % | rejected, wrong |
| 10 | Hypoponera | card, 7,661 px, cream | solidity | 0.564 | 0.60 | 6 % | rejected, wrong |
| 11 | Leptogenys | card, 3,061 px, white | straight edge | 43 | 46.3 | 7 % | rejected, wrong |
| 22 | Royidris | gaster, 28,988 px, white | solidity | 0.436 | 0.45 | 3 % | rejected, right |
| 26 | Technomyrmex | card, 21,607 px, cream | solidity | 0.645 | 0.60 | 7 % | accepted, right |
| 17 | Nylanderia | card, 91,279 px, white | background share | 0.163 | 0.15 | 9 % | accepted, right |
| 5 | Camponotus (GBIF) | pale blob, 1,055 px, white | straight edge | 47 | 50 | 6 % | rejected, right |
| 5 | Camponotus (GBIF) | cream blob, 9,576 px | area | 0.91 % | 1 % | 9 % | rejected (not checked by eye) |
| 2 | Aphaenogaster | pale blob, 1,127 px, white | radius | 7.07 | 7.5 | 6 % | rejected, right |
| 20 | Pheidole | pale blob, 773 px, white | radius | 7.28 | 8 | 9 % | rejected, right |
| 25 | Tapinoma | pale blob, 1,580 px, white | radius | 7.21 | 8 | 10 % | rejected, right |

Accepted candidates not in that list have a wider margin: the nearest test
is at 11 % (the two true pins, straightness 1.00 against 0.90), 18 % (the
Monomorium gaster tip, straight edge 59 against 50), 20 % (Syllophopsis
card, solidity), 30 % and more for the large cards of Anochetus,
Camponotus, Cardiocondyla and Cataulacus.

Two observations for the labelling pass, not acted on: the solidity
thresholds sit in the middle of the cards (0.43 to 0.59 rejected, 0.54 to
0.87 accepted) and of the pale ant parts (Royidris gaster 0.436, Tapinoma
body 0.528), so solidity alone will not separate them on this sample; and
the white tier's area floor is where three of the four wrong acceptances
live (760 to 1,650 px, against a floor of 759 px), while the smallest true
card in the sample is 3,061 px (Leptogenys).

## Runtime

rembg 0.69 s per image (0.45 to 2.00). Rules 4.3 to 18.6 s per image, mean
8.9 s, almost all in the straight-edge (Hough) test over 10 to 43 card
candidates per image. For 1,236 images: about 15 minutes of rembg and
about 3 hours of rules at this speed; the Hough loop over 360 angles is the
obvious thing to vectorise if that matters.

## Reproduce

```bash
.venv/bin/python scripts/26_segmentation_run30.py     # about 5 minutes; rembg model cached in ~/.rembg/models/
```

Deterministic apart from the timings. Needs rembg from
`../README.md`.

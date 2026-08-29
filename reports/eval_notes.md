# Phase C: error analysis notes

Manual review of the 18 linear-probe misclassifications (`errors.csv`,
250 test images, 27 genera). Contact sheets: `errors_sheet.jpg` (all 18,
sorted by probe confidence) and `confusions_pairs.jpg` (the Syllophopsis /
Tetramorium and Royidris / Monomorium cases next to train examples).

Sanity checks passed first:

- No specimen code appears in both splits
  (`groupby('specimen_code').split.nunique().max() == 1`); one image per
  specimen in `dataset.csv`.
- None of the 18 errors is an SEM, a dissection or a non-profile view.

## Errors by category

| category | n | specimens |
|---|---|---|
| genuinely hard (look-alike genera, photo fine) | 12 | casent0101118, casent0101119 (*C. imitator*), casent0048930, casent0443365, casent0010867, casent0101583, casent0101943, casent0101616, casent0102037, casent0467063, casent0101053, casent0173591 |
| bad photo | 3 | casent0317759 (gbif_cache: extreme close crop, head cut off, no scale bar), casent0101146 (blurry, washed out, card fills the frame; probe 0.09 vs 0.09), casent0101139 (soft, low contrast, card background) |
| odd angle / mounting | 2 | casent0128362 (*Royidris*, gbif_cache: head tilted towards the camera, ¾ view), casent0102035 (*Tetraponera* lying flat on a card) |
| suspicious | 1 | casent0102418 (*Camponotus reaumuri* → *Anochetus* at 0.16): an ordinary big-headed reddish *Camponotus*; nothing resembles *Anochetus*. Second guess is *Camponotus* (0.11), so this is low-confidence noise rather than a labelling problem, but worth a second look. |

Only ~3 of 18 errors are attributable to image quality, and 2 of those are
`gbif_cache` images (137 of 1,236 in the set); the Commons images are
uniformly good AntWeb profile shots.

## The *Camponotus imitator* pair

`casent0101118` and `casent0101119` (consecutive codes = same collecting
series) are both *Camponotus imitator* paralectotypes from Taolagnaro,
photographed by April Nobile, predicted *Aphaenogaster* at 0.42 / 0.38.
They are normal workers, not minims, and the photos are fine. Next to a
train *Aphaenogaster swammerdami* the resemblance is striking: slender
body, very long legs and antennae, small head, reddish body with a dark
gaster, unlike the robust *C. foersteri* / *C. maculatus* body plan the
probe mostly learns *Camponotus* from. The epithet *imitator* fits.

The probe had **five train examples of *C. imitator*** (including the
lectotype) and still missed both test specimens, so this is a genuine hard
case, not a coverage gap.

## Unseen-species effect

The split is stratified by **genus** and grouped by specimen, not by
species. In small genera a test image can therefore belong to a species
with no train example at all:

- *Syllophopsis infusca* (both test *Syllophopsis* errors → *Tetramorium*):
  **0 train examples**. Train *Syllophopsis* are fisheri, ferodens,
  hildebrandti, modesta, adiastolon, gongromos: mostly pale, tiny species.
  *S. infusca* is dark, compact and big-gastered, and looks far more like
  the dark *Tetramorium tosii* in train than like *S. ferodens*.
- *Royidris singularis* (→ *Monomorium*): **0 train examples**. Train
  *Royidris* (robertsoni, notorthotenes) are pale yellow; *R. singularis*
  is brown, hairy, with a dark gaster, closest in gestalt to
  *Monomorium nigricans* in train. Its photo is also the ¾-angle
  gbif_cache one.

Implication: per-genus recall for genera with ≤ 3 test images (11 of 27)
partly measures cross-species generalisation from very few species, and
should be read as such. A species-aware split, or simply more images per
genus, would separate the two effects.

## The confusions recapitulate taxonomy

Nearly every off-diagonal cell in `confusion_matrix.png` sits inside a
subfamily block, and the recurring pairs are exactly the ones a taxonomist
would expect:

- *Syllophopsis* and *Royidris* are Monomorium-group genera split off in
  the 2010s revisions (Bolton & Fisher); the probe confuses them with
  *Tetramorium* / *Monomorium*: small compact myrmicines.
- *Bothroponera* was a subgenus of *Pachycondyla* until 2014;
  *Pachycondyla perroti* → *Bothroponera* (0.38) is that boundary.
- *Technomyrmex* ↔ *Tapinoma* (both dolichoderines, 0.30 vs 0.28; a
  near tie), *Aphaenogaster* ↔ *Pheidole*, *Nesomyrmex* ↔ *Crematogaster*
  (heart-shaped gaster), *Tetraponera* ↔ *Lioponera* (both elongate,
  cylindrical).

Cross-subfamily errors are rare and each has a reason: the *C. imitator*
mimicry case above, the long-legged *Pheidole grallatrix* → *Odontomachus*,
and the two bad-photo *Tetramorium*.

## Addendum after 08_umap.py

The UMAP map (`umap_by_genus.png`, probe errors marked ×) resolves two of
the cases above:

- A small island of 51 specimens at UMAP ≈ (3.5, 9.3) contains all 7
  *Camponotus imitator*, 25 *Aphaenogaster* (*swammerdami*, *gonacantha*),
  14 *Odontomachus* (mostly *coquereli*), *Pheidole grallatrix* and a few
  *Strumigenys agra*: long-legged, slender ants from **three
  subfamilies**. The embedding groups them by body plan; the *C. imitator*
  and *P. grallatrix* errors are exactly the members of this island whose
  genus is a minority there.
- The "suspicious" *Camponotus reaumuri* → *Anochetus* case is not
  suspicious: the 3 *C. reaumuri* in the set sit inside the *Anochetus*
  island at ≈ (5.5, 9.6) (27 *Anochetus*, 2 *Tetraponera grandidieri*),
  far from the main *Camponotus* cluster.

See `08_umap.log` for the 10-nearest-neighbour genus composition around
every error.

## Verdict

No evidence of leakage or labelling bugs. The linear probe's 92.8 % top-1 /
0.885 macro-F1 is dominated by genuinely look-alike taxa and by unseen
species in small genera, with a small residue of poor images.

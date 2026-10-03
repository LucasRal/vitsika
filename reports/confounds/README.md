# Photo conditions as a confound for genus

Question: is genus linked to how the photo was taken (photographer, background
tone, image source, framing), so that a classifier could learn the photo setup
instead of the ant? All 1,236 images of `data/dataset.csv` (27 genera), the
frozen `poc-baseline` data. Nothing in `data/` or the baseline `reports/` was
changed; everything here is new and was produced offline.

## Conclusion

1. Genus is clearly linked to photo conditions: every measured condition except image width differs between genera far beyond chance.
2. But without the ant, those conditions predict genus weakly (macro-F1 0.16 against 0.04 for chance), while the image embedding reaches 0.85 under the same protocol.
3. The model does not appear to rely on them: Strumigenys, the genus with the strongest setup signature, is recognised just as well in atypical photos, and no other genus photographed in that setup is pulled towards it.
4. The image embedding does encode the background (it predicts background brightness with R² 0.66), and accuracy drops a little on backgrounds atypical for the genus (0.945 to 0.896), so some background reliance cannot be ruled out.
5. Biggest risks: the Strumigenys imaging campaign (thin-line scale bar, 2000-2007 rights range, 100 % one photographer), GBIF-cache crops (accuracy 0.80 against 0.93), and background colour and type status, which differ by genus.

## What was measured

`scripts/20_image_stats.py` writes one row per image to `image_stats.csv`:

- **Background**: median RGB of the border region (the outer 5 % of width and
  height on all four sides), its mean luminance (0-255) and a warm/cool index,
  R minus B of the border median.
- **Geometry**: width, height, aspect ratio, file size.
- **Dark-pixel share**: pixels darker than half the border's median luminance
  (ant, pin and scale bar), plus an absolute variant (luminance below 60).
- **Scale bar**: style, position and length in pixels. See "Limitations".
- **Metadata**: genus, subfamily, photographer (`creator`), image source,
  `typeStatus`, the year range in `rightsHolder` as two numbers, split,
  sha256, source URL, and the three-way source from `data/image_manifest.csv`.

| Image source | Images | Typical size |
|---|---|---|
| Commons original, downscaled by the harvester | 1,037 | 1024 x 705 |
| Commons thumbnail, the 83 files at 1280 px | 83 | 1280 x 740 |
| GBIF cache | 116 | 1024 x 1024, all square, zoomed crops |

## Main numbers

`association_summary.md` has the full table. Each measure is compared with
its value under 1,000 random shuffles of the genus labels; all are far above
that level (permutation p < 0.001) except image width (p = 0.52).

| Variable | Association with genus | Level with no link |
|---|---|---|
| Scale bar style | V = 0.55 | 0.03 |
| Type status | V = 0.46 | 0.02 |
| Background warm/cool, 3 bands | V = 0.31 | 0.02 |
| Background brightness, 4 bands | V = 0.25 | 0.02 |
| Rights year range | V = 0.24 | 0.03 |
| Photographer | V = 0.22 | 0.02 |
| Image source | V = 0.17 | 0.03 |
| Background R, G, B, brightness, R-B | eta² = 0.11 to 0.15 | 0.02 |
| Height, aspect ratio, file size | eta² = 0.11 to 0.12 | 0.02 |
| Width | eta² = 0.02 | 0.02 |

V is bias-corrected Cramér's V; eta² is the ANOVA eta squared. The
dark-pixel share (eta² 0.34) and scale bar length (0.25) are left out of
this table: both mostly measure the ant (its colour and size).

Patterns behind the numbers (tables in `tables/`):

- **Photographer.** April Nobile took 1,051 of the 1,236 images. 11 genera come
  ≥ 90 % from her, 8 of them 100 % (`single_photographer_genera.csv`).
  The other photographers concentrate in a few genera: Pheidole (27 of 78 by
  Michele Esposito), Nesomyrmex, Cataulacus, Terataner, Pachycondyla.
- **Image source.** GBIF-cache crops make up 46 % of Pachycondyla, 35 % of
  Nesomyrmex, 30 % of Royidris and 23 % of Pheidole, but 0 % of 8 genera.
- **Background.** Median brightness runs from 144 (Anochetus) to 207
  (Meranoplus). Leptogenys, Hypoponera and Lioponera are mostly warm
  (71 to 84 % of images with R-B ≥ 10), most others neutral.
- **Strumigenys.** 171 of its 223 images use a thin-line scale bar that only
  28 of the 1,013 other images have. 162 of those 171 are from one specimen
  code series (casent0005xxx). Its rights range is 2000-2007 in 90 % of images,
  against 3 to 64 % for other genera. 85 % are type specimens.
- **Type specimens.** The share ranges from 14 % (Odontomachus) to 85 %
  (Strumigenys).

Figures in `figures/`:

- `genus_x_photographer_heatmap.png`: share of each genus's images per photographer.
- `background_strips.png`: 8 random images per genus, sorted by background brightness.
- `brightness_vs_warmcool.png`: background brightness against warm/cool, one panel per genus.
- `scalebar_validation_*.jpg`: the scale-bar check described under Limitations.

The brightness scatter uses one small panel per genus because 27 colours in
one plot cannot be told apart.

## The key test: genus without the ant

`scripts/22_metadata_only_classifier.py`, full output in
`metadata_only_classifier.md`. 5-fold stratified cross-validation, seed 42,
balanced class weights. Features: background RGB, brightness, warm/cool,
width, height, aspect ratio, image source, photographer, rights end year.

| Features | Logistic regression macro-F1 | Random forest macro-F1 | Forest accuracy |
|---|---|---|---|
| All non-ant features | 0.120 | 0.161 | 0.245 |
| Without photographer | 0.088 | 0.144 | 0.239 |
| Background colour only | 0.061 | 0.116 | 0.167 |
| Chance: always the majority genus | 0.012 | 0.012 | 0.193 |
| Chance: random with class priors, 99th percentile | 0.049 | 0.049 | 0.122 |
| For comparison: BioCLIP 2 embedding | 0.852 | | 0.921 (logistic) |

Every non-ant feature set beats chance (permutation p < 0.005 for macro-F1).
Accuracy barely beats always guessing Camponotus (0.193). Balanced class
weights trade accuracy on Camponotus for rare genera, so macro-F1 is the
fair yardstick. The signal is concentrated: Strumigenys reaches recall 0.70
to 0.77 from non-ant features alone. Bothroponera, Lioponera, Anochetus and
Pheidole reach 0.32 to 0.48 with the forest. The other genera stay below 0.3
with the forest, Camponotus at 0.03; the logistic regression ranks a few
differently, Syllophopsis at 0.43 (`metadata_only_per_genus.csv`).

**Interpretation.** Photo conditions carry real but weak genus information.
That is enough for a shortcut on a few genera, but far from explaining the
92 % accuracy. Two further checks bear on whether the model uses it:

| Check, out-of-fold BioCLIP 2 predictions | Result |
|---|---|
| Strumigenys in its typical thin-line setup | 168 of 171 correct |
| Strumigenys in other setups | 52 of 52 correct |
| Other genera in the Strumigenys setup, predicted as Strumigenys | 0 of 28 |
| Commons originals / GBIF-cache crops | 0.932 / 0.802 correct |
| Commons, April Nobile / other photographers | 0.932 / 0.942 correct |
| Background typical / atypical for the genus | 0.945 / 0.896 correct |

The strongest confound is not used as a shortcut. Photographer makes no
difference. The GBIF-cache gap fits the crops showing less of the ant at
least as well as it fits a setup effect. The background gap is small. Its
95 % intervals barely separate (0.924-0.960 against 0.870-0.918), and they
overlap within Commons images alone.

## Limitations

- **Border is not pure background.** Legs, antennae, pins and card edges
  reach into the border, more so for large ants and in the GBIF-cache crops,
  so background colour is partly the ant. The median limits but does not
  remove this.
- **The dark-pixel share depends on the ant's colour.** Pale ants such as
  Strumigenys (median 0.00) barely register, so it is not a reliable size
  proxy and is left out of the classifier.
- **Scale bar mm labels were not read.** They exist only as pixels, the file
  names and metadata do not carry them, and no OCR is available offline.
  `scalebar_mm` is empty. The labels seen range from 0.1 to 5 mm, so pixel
  length alone does not give magnification.
- **Scale bar detection was checked by eye.**
  - Black bars and thin lines: 40 of 40 random detections of each style sat
    on a real bar (`scalebar_validation_black_bar.jpg`, `..._thin_line.jpg`).
    Another 80 earlier black-bar detections were also all correct.
  - The 36 images with no detected bar: about 22 have none in frame, about 8
    show only a fragment at a corner, about 5 faint Strumigenys thin lines are
    probably missed, and 1 black bar placed mid-image on the card was missed.
  - Bars cut by the image edge are marked `truncated` with no length.
  - Lengths are reported only for status `found`.
- **Rights years have no variation in the start.** Every rightsHolder starts
  at 2000, so only the end year varies. It reflects when AntWeb's rights
  statement was written, which may track the imaging campaign; it is not the
  photo date.
- **The tests show association and susceptibility, not mechanism.** The
  failure analysis uses the cross-validated embedding classifier, not the
  deployed probe on its test split. A direct test would mask the ant and
  measure how much prediction survives. The RISE maps in `reports/rise/`
  look at this per image.
- **Scope of the data.** All images are AntWeb photographs taken under
  similar museum conditions. A user's phone photo is far outside this
  distribution, and these results say nothing about it.

## Reproduce

Offline, from the repository root:

```bash
.venv/bin/python scripts/20_image_stats.py            # image_stats.csv, about 45 s
.venv/bin/python scripts/21_confound_tables.py        # tables/, association_summary.*, about 2 min
.venv/bin/python scripts/22_metadata_only_classifier.py   # metadata_only_*, embedding_*, about 2.5 min
.venv/bin/python scripts/23_confound_figures.py       # figures/, about 20 s
```

Seeds are fixed (42) throughout.

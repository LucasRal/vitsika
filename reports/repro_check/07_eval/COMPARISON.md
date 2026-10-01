# Step 5: re-running 07_eval.py from cached embeddings

Command, run offline from the cached `data/embeddings.npy`:

```
HF_HUB_OFFLINE=1 .venv/bin/python scripts/07_eval.py --output-dir reports/repro_check/07_eval
```

Baseline: `reports/metrics.json`, `reports/per_genus.csv` and the other files of commit 961e00c (tag `poc-baseline`). Re-run: this folder. Values below are read from both files; "identical" means exact equality of the stored values, not rounding.

## Headline metrics

| Method | Top-1 baseline | Top-1 re-run | Top-3 baseline | Top-3 re-run | Macro-F1 baseline | Macro-F1 re-run | Identical |
|---|---|---|---|---|---|---|---|
| zero_shot_template | 0.7440 | 0.7440 | 0.9400 | 0.9400 | 0.5901 | 0.5901 | yes |
| zero_shot_plain | 0.3280 | 0.3280 | 0.4880 | 0.4880 | 0.1139 | 0.1139 | yes |
| linear_probe | 0.9280 | 0.9280 | 0.9840 | 0.9840 | 0.8846 | 0.8846 | yes |
| nearest_neighbour | 0.9160 | 0.9160 | 0.9320 | 0.9320 | 0.8340 | 0.8340 | yes |
| linear_probe_uncalibrated | 0.9280 | 0.9280 | 0.9840 | 0.9840 | 0.8846 | 0.8846 | yes |

## Everything else

| Check | Result |
|---|---|
| metrics.json, all 46 shared fields (counts, probe settings, temperature, ECE, sigmoid alternative) | identical |
| metrics.json, new field | `embed_model_revision` only (added in Step 2) |
| per_genus.csv, 27 genera x 11 columns | byte-identical |
| errors.csv, 18 misclassified test images | byte-identical |
| confusion_matrix.png | byte-identical |
| Calibrated probe (`data/probe.pkl`): temperature 0.22432117692838643, LogisticRegression coef_ and intercept_, classes_ | identical (numpy array_equal) |
| Uncalibrated probe (`data/probe_uncalibrated.pkl`): coef_ and intercept_ | identical (numpy array_equal) |
| 07_eval.log | identical apart from timings, output paths and one Hugging Face warning absent offline |

The re-run's probe pickles are kept here for inspection but are gitignored, like `data/probe*.pkl`.

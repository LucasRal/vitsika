#!/usr/bin/env python3
"""Item 3, steps 3 and 4: genus scores of the four image versions.

A. Deployed probe, unchanged (data/probe.pkl, read only): predicts the
   test-split rows of each version. Accuracy and macro-F1 (labels = the
   probe's 27 genera, as in scripts/07_eval.py), with 95 % percentile
   bootstrap intervals over test images (1,000 resamples, seed 42). In a
   resample, a genus with no true and no predicted image is left out of the
   macro average rather than counted as F1 = 0.
B. Retrained per version: the protocol of scripts/22_metadata_only_classifier.py
   for its bioclip_embedding row (LogisticRegression C=1, max_iter 5000,
   class_weight="balanced", 5-fold StratifiedKFold, shuffle, seed 42),
   trained and tested on the same version over all rows. Out-of-fold
   accuracy and macro-F1.

Each is run on all rows ("all") and again without the partial and detail
images ("clean", sensitivity). The chance levels of reports/confounds/README.md
(1,236 images, 5-fold CV) are added as reference rows.

Reads:  data/masking/embeddings_<version>.npy, data/masking/embeddings_index.csv,
        data/probe.pkl, reports/metrics.json (POC test metrics, for a check)
Writes: reports/masking/scores.csv, reports/masking/per_genus.csv (recall
        per genus, per version, for A and B), reports/masking/31_score_variants.log
Offline.
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # api.probe, to unpickle the deployed probe
sys.path.insert(0, str(ROOT / "scripts"))
from gbif_client import setup_logging  # noqa: E402

MASK = ROOT / "data" / "masking"
REPORT = ROOT / "reports" / "masking"
VERSIONS = ["original", "ant_grey", "ant_erased", "scalebar_erased"]
SEED = 42
N_BOOT = 1000

# reports/confounds/README.md, table "Features x macro-F1"
CHANCE = [
    {"version": "chance: always the majority genus", "macro_f1": 0.012, "accuracy": 0.193},
    {"version": "chance: random with class priors, 99th percentile", "macro_f1": 0.049, "accuracy": 0.122},
]

log = logging.getLogger("score_variants")


def bootstrap(y: np.ndarray, pred: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(y)
    acc, f1 = np.empty(N_BOOT), np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        acc[b] = (y[i] == pred[i]).mean()
        f1[b] = f1_score(y[i], pred[i], average="macro", zero_division=np.nan)
    q = lambda a: np.quantile(a, [0.025, 0.975])  # noqa: E731
    (al, ah), (fl, fh) = q(acc), q(f1)
    return {"accuracy_lo": al, "accuracy_hi": ah, "macro_f1_lo": fl, "macro_f1_hi": fh}


def per_genus(y, pred, genera, **keys) -> list[dict]:
    rec = recall_score(y, pred, labels=genera, average=None, zero_division=np.nan)
    n = pd.Series(y).value_counts()
    return [{**keys, "genus": g, "n": int(n.get(g, 0)), "recall": r} for g, r in zip(genera, rec)]


def main() -> None:
    setup_logging(REPORT / "31_score_variants.log")
    index = pd.read_csv(MASK / "embeddings_index.csv")
    embs = {v: np.load(MASK / f"embeddings_{v}.npy") for v in VERSIONS}
    assert all(len(e) == len(index) for e in embs.values())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probe = joblib.load(ROOT / "data" / "probe.pkl")
    for w in caught:
        log.info("note while loading the probe: %s", str(w.message).split(".")[0])
    genera = np.asarray(probe.classes_)
    log.info("deployed probe: %r, %d genera", probe, len(genera))

    clean = (index["partial"] == 0) & (index["detail_image"] == 0)
    subsets = {"all": np.ones(len(index), bool), "clean": clean.to_numpy()}
    log.info("rows: %d all, %d clean (without %d partial and %d detail images)",
             len(index), clean.sum(), index["partial"].sum(), index["detail_image"].sum())

    scores, genus_rows = [], []
    for subset, keep in subsets.items():
        test = keep & (index["split"] == "test").to_numpy()
        y_test = index.loc[test, "genus"].to_numpy()
        y_all = index.loc[keep, "genus"].to_numpy()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        log.info("[%s] A: %d test images; B: %d images, %d genera, smallest genus %d",
                 subset, test.sum(), keep.sum(), len(set(y_all)), pd.Series(y_all).value_counts().min())
        for v in VERSIONS:
            # A: deployed probe on the test split
            pred = probe.predict(embs[v][test])
            row = {"version": v, "subset": subset, "scheme": "A_deployed_probe",
                   "n": int(test.sum()),
                   "accuracy": accuracy_score(y_test, pred),
                   "macro_f1": f1_score(y_test, pred, labels=genera, average="macro", zero_division=0),
                   **bootstrap(y_test, pred, np.random.default_rng(SEED))}
            scores.append(row)
            genus_rows += per_genus(y_test, pred, genera, version=v, subset=subset,
                                    scheme="A_deployed_probe")
            log.info("[%s] A %-16s acc %.3f [%.3f, %.3f]  macro-F1 %.3f [%.3f, %.3f]", subset, v,
                     row["accuracy"], row["accuracy_lo"], row["accuracy_hi"],
                     row["macro_f1"], row["macro_f1_lo"], row["macro_f1_hi"])

            # B: retrained, 5-fold CV on the same version
            pred = cross_val_predict(LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"),
                                     embs[v][keep], y_all, cv=cv, n_jobs=5)
            row = {"version": v, "subset": subset, "scheme": "B_retrained_cv", "n": int(keep.sum()),
                   "accuracy": accuracy_score(y_all, pred),
                   "macro_f1": f1_score(y_all, pred, average="macro")}
            scores.append(row)
            genus_rows += per_genus(y_all, pred, genera, version=v, subset=subset,
                                    scheme="B_retrained_cv")
            log.info("[%s] B %-16s acc %.3f  macro-F1 %.3f", subset, v, row["accuracy"], row["macro_f1"])

    # check: A on the originals should be close to the POC test metrics
    poc = json.loads((ROOT / "reports" / "metrics.json").read_text())["methods"]["linear_probe"]
    a0 = next(r for r in scores if r["version"] == "original" and r["subset"] == "all"
              and r["scheme"] == "A_deployed_probe")
    log.info("check: A original (%d test images) acc %.3f macro-F1 %.3f; POC test (250 images) "
             "top-1 %.3f macro-F1 %.3f", a0["n"], a0["accuracy"], a0["macro_f1"],
             poc["top1"], poc["macro_f1"])

    out = pd.DataFrame(scores + [{**c, "subset": "", "scheme": "chance (reports/confounds/README.md)",
                                  "n": 1236} for c in CHANCE])
    cols = ["scheme", "subset", "version", "n", "accuracy", "accuracy_lo", "accuracy_hi",
            "macro_f1", "macro_f1_lo", "macro_f1_hi"]
    out[cols].round(4).to_csv(REPORT / "scores.csv", index=False)
    pg = pd.DataFrame(genus_rows)
    pg["recall"] = pg["recall"].round(4)
    pg.to_csv(REPORT / "per_genus.csv", index=False)
    log.info("wrote reports/masking/scores.csv (%d rows) and per_genus.csv (%d rows)", len(out), len(pg))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Can genus be predicted without looking at the ant?

Reads reports/confounds/image_stats.csv and trains classifiers on non-ant
features only, with 5-fold stratified cross-validation (shuffle, seed 42)
over all 1,236 images. Two models, both class_weight="balanced" like the
deployed probe:

  logreg   one-hot categoricals + standardised numerics, LogisticRegression
           (C=1, lbfgs, max_iter 5000)
  forest   RandomForestClassifier(200 trees, min_samples_leaf=2, seed 42)

Feature sets (the first three are the requested ones):

  all_non_ant       background RGB, brightness, warm/cool, width, height,
                    aspect ratio, image source, photographer, rights end year
  no_photographer   the same without photographer
  background_only   background RGB, brightness, warm/cool
  photographer_only photographer alone (context)
  all_plus_scalebar all_non_ant + scale bar style (context; see 20_image_stats.py)
  bioclip_embedding the frozen BioCLIP 2 embeddings, same folds, logreg only
                    (reference: what the image model reaches under this protocol)

Chance levels on the same folds: always predicting the majority genus, and
guessing at random with the class priors (expected accuracy sum p_k^2,
expected macro-F1 1/K; also simulated 200 times). Each feature set's
logreg score is also tested against 200 permutations of the genus labels
(sklearn permutation_test_score).

A second part asks whether the embedding itself encodes the photo
conditions: 5-fold CV R^2 of a ridge regression from the embedding to each
background measure, and CV accuracy of a logistic regression from the
embedding to scale bar style and image source.

A third part checks whether the embedding classifier's out-of-fold errors
follow atypical photo conditions (embedding_accuracy_by_condition.csv).

Output: reports/confounds/metadata_only_classifier.md and .csv,
metadata_only_per_genus.csv (per-genus recall of the all_non_ant models),
embedding_encodes_conditions.csv and embedding_accuracy_by_condition.csv.
Offline.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, f1_score, r2_score, recall_score
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                     permutation_test_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "reports" / "confounds"
SEED = 42
N_PERM = 200
N_RANDOM = 200

BACKGROUND = ["border_r", "border_g", "border_b", "border_brightness", "warm_cool"]
GEOMETRY = ["width", "height", "aspect"]
FEATURE_SETS = {
    "all_non_ant": (BACKGROUND + GEOMETRY + ["rights_year_end"], ["source3", "creator"]),
    "no_photographer": (BACKGROUND + GEOMETRY + ["rights_year_end"], ["source3"]),
    "background_only": (BACKGROUND, []),
    "photographer_only": ([], ["creator"]),
    "all_plus_scalebar": (BACKGROUND + GEOMETRY + ["rights_year_end"],
                          ["source3", "creator", "scalebar_style"]),
}
REQUESTED = ["all_non_ant", "no_photographer", "background_only"]

log = logging.getLogger("metadata_only")


def logreg(num: list[str], cat: list[str]):
    steps = []
    if num:
        steps.append(("num", StandardScaler(), num))
    if cat:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    return make_pipeline(ColumnTransformer(steps),
                         LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))


def forest(num: list[str], cat: list[str]):
    steps = []
    if num:
        steps.append(("num", "passthrough", num))
    if cat:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    return make_pipeline(ColumnTransformer(steps),
                         RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                class_weight="balanced", random_state=SEED, n_jobs=-1))


def scores(y, pred) -> dict:
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}


def chance_levels(y: np.ndarray, cv) -> dict:
    out = {}
    maj = cross_val_predict(DummyClassifier(strategy="most_frequent"), np.zeros((len(y), 1)), y, cv=cv)
    out["majority"] = scores(y, maj)
    p = pd.Series(y).value_counts(normalize=True).to_numpy()
    sims = np.array([list(scores(y, cross_val_predict(
        DummyClassifier(strategy="stratified", random_state=i), np.zeros((len(y), 1)), y, cv=cv)).values())
        for i in range(N_RANDOM)])
    out["random_priors"] = {"accuracy": float((p ** 2).sum()), "macro_f1": 1 / len(p),
                            "sim_accuracy_mean": sims[:, 0].mean(), "sim_accuracy_p99": np.quantile(sims[:, 0], .99),
                            "sim_macro_f1_mean": sims[:, 1].mean(), "sim_macro_f1_p99": np.quantile(sims[:, 1], .99)}
    return out


def load_embeddings(stats: pd.DataFrame) -> np.ndarray:
    emb = np.load(ROOT / "data" / "embeddings.npy")
    idx = pd.read_csv(ROOT / "data" / "embeddings_index.csv")
    pos = pd.Series(range(len(idx)), index=idx["specimen_code"])
    return emb[pos.loc[stats["specimen_code"]].to_numpy()]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def failure_analysis(s: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Out-of-fold correctness of the embedding classifier, by photo condition."""
    s = s.assign(pred=pred, ok=pred == s["genus"].to_numpy())
    commons = s["source3"] != "gbif_cache"
    # how unusual the background is for the image's own genus: |x - genus median| / genus sd
    dev = s.groupby("genus")["border_brightness"].transform(lambda v: (v - v.median()).abs() / v.std())
    half = dev > dev.median()
    strum = s["genus"] == "Strumigenys"
    thin = s["scalebar_style"] == "thin_line"
    groups = [
        ("Strumigenys, thin-line scale bar (its typical setup)", strum & thin),
        ("Strumigenys, other scale bar or none", strum & ~thin),
        ("Other genera, thin-line scale bar", ~strum & thin),
        ("Other genera, other scale bar or none", ~strum & ~thin),
        ("Commons original", s["source3"] == "commons_original"),
        ("Commons thumbnail (1280 px)", s["source3"] == "commons_thumbnail"),
        ("GBIF cache (square crops)", s["source3"] == "gbif_cache"),
        ("Commons images, April Nobile", commons & (s["creator"] == "April Nobile")),
        ("Commons images, other photographers", commons & (s["creator"] != "April Nobile")),
        ("Background brightness typical for the genus (closer half)", ~half),
        ("Background brightness atypical for the genus (further half)", half),
        ("Commons images, typical background (closer half)", commons & ~half),
        ("Commons images, atypical background (further half)", commons & half),
    ]
    rows = []
    for name, m in groups:
        n, k = int(m.sum()), int(s.loc[m, "ok"].sum())
        lo, hi = wilson(k, n)
        rows.append({"group": name, "n": n, "correct": k, "rate": k / n if n else np.nan,
                     "ci_low": lo, "ci_high": hi})
    other_thin = ~strum & thin
    rows.append({"group": "Other genera with thin-line bar predicted as Strumigenys", "n": int(other_thin.sum()),
                 "correct": int((s.loc[other_thin, "pred"] == "Strumigenys").sum()),
                 "rate": float((s.loc[other_thin, "pred"] == "Strumigenys").mean()),
                 "ci_low": wilson(int((s.loc[other_thin, "pred"] == "Strumigenys").sum()), int(other_thin.sum()))[0],
                 "ci_high": wilson(int((s.loc[other_thin, "pred"] == "Strumigenys").sum()), int(other_thin.sum()))[1]})
    return pd.DataFrame(rows)


def fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    setup_logging(CONF / "22_metadata_only_classifier.log")
    t0 = time.perf_counter()
    s = pd.read_csv(CONF / "image_stats.csv")
    y = s["genus"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    chance = chance_levels(y, cv)
    log.info("chance: %s", chance)

    rows, per_genus = [], {}
    for name, (num, cat) in FEATURE_SETS.items():
        x = s[num + cat]
        for model_name, make in (("logreg", logreg), ("forest", forest)):
            pred = cross_val_predict(make(num, cat), x, y, cv=cv)
            r = {"feature_set": name, "model": model_name, **scores(y, pred),
                 "n_features": len(num) + len(cat)}
            if model_name == "logreg":
                _, perm, p = permutation_test_score(make(num, cat), x, y, cv=cv, n_permutations=N_PERM,
                                                    random_state=SEED, n_jobs=-1, scoring="f1_macro")
                r.update({"perm_macro_f1_mean": perm.mean(), "perm_macro_f1_p99": np.quantile(perm, .99),
                          "perm_p": p})
            if name == "all_non_ant":
                per_genus[model_name] = recall_score(y, pred, average=None, labels=np.unique(y))
            rows.append(r)
            log.info("%-18s %-6s acc %.3f macro-F1 %.3f", name, model_name, r["accuracy"], r["macro_f1"])

    emb = load_embeddings(s)
    pred = cross_val_predict(LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"),
                             emb, y, cv=cv)
    bio_pred = pred
    rows.append({"feature_set": "bioclip_embedding", "model": "logreg", **scores(y, pred),
                 "n_features": emb.shape[1]})
    per_genus["bioclip_logreg"] = recall_score(y, pred, average=None, labels=np.unique(y))
    log.info("bioclip embedding logreg acc %.3f macro-F1 %.3f", rows[-1]["accuracy"], rows[-1]["macro_f1"])
    res = pd.DataFrame(rows)
    res.to_csv(CONF / "metadata_only_classifier.csv", index=False)

    genera = np.unique(y)
    pg = pd.DataFrame({"genus": genera, "n": pd.Series(y).value_counts().reindex(genera).to_numpy(),
                       "recall_non_ant_logreg": per_genus["logreg"],
                       "recall_non_ant_forest": per_genus["forest"],
                       "recall_bioclip_logreg": per_genus["bioclip_logreg"]}).round(3)
    pg = pg.sort_values("recall_non_ant_forest", ascending=False)
    pg.to_csv(CONF / "metadata_only_per_genus.csv", index=False)

    # does the embedding encode the photo conditions?
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    enc = []
    for col in BACKGROUND + ["aspect"]:
        p = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), emb, s[col].to_numpy(float), cv=kf)
        enc.append({"target": col, "measure": "CV R^2 (ridge)", "value": r2_score(s[col], p),
                    "baseline": 0.0})
    for col in ("scalebar_style", "source3"):
        t = s[col].to_numpy()
        p = cross_val_predict(LogisticRegression(C=1.0, max_iter=5000), emb, t, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
        enc.append({"target": col, "measure": "CV accuracy (logreg)", "value": accuracy_score(t, p),
                    "baseline": pd.Series(t).value_counts(normalize=True).iloc[0],
                    "macro_recall": recall_score(t, p, average="macro")})
    enc = pd.DataFrame(enc)
    enc.to_csv(CONF / "embedding_encodes_conditions.csv", index=False)

    fail = failure_analysis(s, bio_pred)
    fail.to_csv(CONF / "embedding_accuracy_by_condition.csv", index=False)

    # markdown
    maj, rnd = chance["majority"], chance["random_priors"]
    L = ["# Can genus be predicted without looking at the ant?", "",
         "Generated by `scripts/22_metadata_only_classifier.py`. All "
         f"{len(s)} images, 27 genera, 5-fold stratified cross-validation (shuffled, seed {SEED}); "
         "pooled out-of-fold predictions. Both models use balanced class weights, like the deployed "
         "probe. No pixel of the ant is used: only the background border colour, image geometry, "
         "image source, photographer and the rightsHolder end year (the start year is 2000 for every "
         "image, so it carries nothing).", "",
         "## Chance levels", "",
         "| Baseline | Accuracy | Macro-F1 |", "|---|---|---|",
         f"| Always the majority genus (Camponotus, {np.mean(y == 'Camponotus'):.1%} of images) | {fmt(maj['accuracy'])} | {fmt(maj['macro_f1'])} |",
         f"| Random guess with class priors, expected | {fmt(rnd['accuracy'])} | {fmt(rnd['macro_f1'])} |",
         f"| Random guess with class priors, 99th percentile of {N_RANDOM} runs | {fmt(rnd['sim_accuracy_p99'])} | {fmt(rnd['sim_macro_f1_p99'])} |",
         "", "## Results", "",
         "| Features | Model | Accuracy | Macro-F1 | Permutation null macro-F1, mean / 99th pct | p |",
         "|---|---|---|---|---|---|"]
    for r in res.itertuples():
        perm = (f"{fmt(r.perm_macro_f1_mean)} / {fmt(r.perm_macro_f1_p99)}"
                if r.model == "logreg" and r.feature_set != "bioclip_embedding" else "")
        p = ("" if pd.isna(getattr(r, "perm_p", np.nan)) else
             (f"{r.perm_p:.3f}" if r.perm_p >= 1 / N_PERM else f"<{1 / N_PERM:.3f}"))
        tag = "" if r.feature_set in REQUESTED else " (context)"
        L.append(f"| {r.feature_set}{tag} | {r.model} | {fmt(r.accuracy)} | {fmt(r.macro_f1)} | {perm} | {p} |")
    L += ["", "The permutation test refits the logistic regression on shuffled genus labels "
          f"{N_PERM} times and scores macro-F1; p is the share of shuffles that score at least as well "
          f"(smallest possible value 1/{N_PERM + 1}). Accuracy is not the right yardstick here: the "
          "balanced class weights trade accuracy on Camponotus, the majority genus, for recall on rare "
          "genera, so compare macro-F1 with the chance levels above.", "",
          "## Genera most predictable from non-ant features", "",
          "Per-genus recall of the all_non_ant models (out-of-fold), next to the BioCLIP 2 embedding "
          "under the same protocol. Full table: `metadata_only_per_genus.csv`.", "",
          "| Genus | n | Recall, non-ant logreg | Recall, non-ant forest | Recall, BioCLIP 2 |",
          "|---|---|---|---|---|"]
    for r in pg.head(10).itertuples():
        L.append(f"| {r.genus} | {r.n} | {fmt(r.recall_non_ant_logreg)} | {fmt(r.recall_non_ant_forest)} | {fmt(r.recall_bioclip_logreg)} |")
    L += ["", "## Does the image embedding itself encode the photo conditions?", "",
          "Cross-validated prediction of each photo condition from the frozen BioCLIP 2 embedding "
          "(5 folds, seed 42). A high value means the classifier's input carries that condition, so "
          "the probe could use it; it does not show that it does.", "",
          "| Condition | Measure | Value | Baseline |", "|---|---|---|---|"]
    for r in enc.itertuples():
        base = "0 (predicting the mean)" if r.measure.startswith("CV R") else f"{r.baseline:.3f} (majority class)"
        L.append(f"| {r.target} | {r.measure} | {fmt(r.value)} | {base} |")
    L += ["", "## Does the image classifier fail when the photo conditions are atypical?", "",
          "If the classifier relied on photo conditions, images taken under conditions unusual for "
          "their genus would be misclassified more often, and images of other genera taken under a "
          "genus's typical conditions would be pulled towards it. Out-of-fold predictions of the "
          "BioCLIP 2 logistic regression above; 95 % Wilson intervals.", "",
          "| Group | Images | Correct | Share correct (95 % CI) |", "|---|---|---|---|"]
    for r in fail.itertuples():
        L.append(f"| {r.group} | {r.n} | {r.correct} | {r.rate:.3f} ({r.ci_low:.3f}-{r.ci_high:.3f}) |")
    L.append("")
    (CONF / "metadata_only_classifier.md").write_text("\n".join(L))
    log.info("wrote %s in %.0fs", CONF / "metadata_only_classifier.md", time.perf_counter() - t0)


if __name__ == "__main__":
    main()

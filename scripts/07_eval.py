#!/usr/bin/env python3
"""Phase C: evaluate genus classification on the frozen BioCLIP 2 embeddings.

Reads data/embeddings.npy + data/embeddings_index.csv (from 06_embed.py) and
uses the index's split column (train / test). Three methods:

- A, zero-shot: BioCLIP 2 text tower on the genus name, two prompts
  ("a photo of {genus}, a genus of ant" and the bare "{genus}").
- B, linear probe: sklearn LogisticRegression (balanced, C from config.yaml)
  on the train embeddings, then temperature-scaled (api/probe.py): one scalar
  T on the logits, fitted by NLL on 5-fold out-of-fold TRAIN logits, so the
  probabilities the API shows are calibrated while the ranking is untouched.
  The calibrated model is saved to data/probe.pkl (joblib) for the demo API,
  the raw probe to data/probe_uncalibrated.pkl. metrics.json reports mean
  max-probability and 10-bin ECE before/after, plus the same numbers for
  sklearn's CalibratedClassifierCV(sigmoid, cv=5), which was tried first and
  rejected because its per-class sigmoids re-rank test images.
- C, nearest neighbour: cosine top-1 against the train embeddings; its
  "top-3" is the majority vote of the 3 nearest (ties -> nearest wins).

Outputs: reports/metrics.json (top-1, top-3, macro-F1 per method),
reports/per_genus.csv (n_test, recall, F1 per genus and method),
reports/confusion_matrix.png (probe; rows = true, cols = predicted, ordered
by subfamily then genus), reports/errors.csv (probe misclassifications).
Deterministic: no sampling anywhere; lbfgs is deterministic for a fixed
input. Log: reports/07_eval.log.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gbif_client import load_config, quiet_logging, setup_logging  # noqa: E402
from api.probe import TemperatureScaledProbe, fit_temperature  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

EMB_PATH = DATA / "embeddings.npy"
INDEX_PATH = DATA / "embeddings_index.csv"
PROBE_PATH = DATA / "probe.pkl"
PROBE_RAW_PATH = DATA / "probe_uncalibrated.pkl"
CALIBRATION_FOLDS = 5
ECE_BINS = 10

TEMPLATES = {
    "zero_shot_template": "a photo of {genus}, a genus of ant",
    "zero_shot_plain": "{genus}",
}
METHODS = [*TEMPLATES, "linear_probe", "nearest_neighbour"]

log = logging.getLogger("eval")


# --------------------------------------------------------------------------- data
def load_embeddings() -> tuple[np.ndarray, pd.DataFrame]:
    embs = np.load(EMB_PATH)
    index = pd.read_csv(INDEX_PATH)
    assert len(embs) == len(index), f"{len(embs)} embeddings vs {len(index)} index rows"
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), (
        f"embeddings not L2-normalised (norms in [{norms.min():.4f}, {norms.max():.4f}])")
    assert index["specimen_code"].is_unique
    log.info("embeddings: %s %s, %d genera, norms in [%.5f, %.5f]",
             embs.shape, embs.dtype, index["genus"].nunique(), norms.min(), norms.max())
    return embs.astype(np.float32), index


def topk_from_scores(scores: np.ndarray, classes: np.ndarray, k: int = 3) -> np.ndarray:
    """(n, k) array of class names, best first."""
    order = np.argsort(-scores, axis=1)[:, :k]
    return classes[order]


# ------------------------------------------------------------------ method A: zero-shot
def zero_shot_scores(model_name: str, genera: np.ndarray, x_test: np.ndarray
                     ) -> dict[str, np.ndarray]:
    import open_clip
    import torch

    t0 = time.perf_counter()
    with quiet_logging():  # open_clip + huggingface_hub chatter
        model, _, _ = open_clip.create_model_and_transforms(model_name)
        tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    log.info("text model %s loaded in %.1fs", model_name, time.perf_counter() - t0)

    out = {}
    for name, template in TEMPLATES.items():
        prompts = [template.format(genus=g) for g in genera]
        with torch.no_grad():
            t = model.encode_text(tokenizer(prompts))
        t = (t / t.norm(dim=-1, keepdim=True)).to(torch.float32).numpy()
        out[name] = x_test @ t.T
        log.info("zero-shot %-20s text embeddings %s, e.g. %r", name, t.shape, prompts[0])
    return out


# --------------------------------------------------------------- method B: linear probe
def fit_probe(x_train: np.ndarray, y_train: np.ndarray, C: float, max_iter: int
              ) -> LogisticRegression:
    t0 = time.perf_counter()
    clf = LogisticRegression(max_iter=max_iter, class_weight="balanced", C=C)
    clf.fit(x_train, y_train)
    log.info("linear probe: %d classes, C=%g, %d iterations, train acc %.3f, fit %.1fs",
             len(clf.classes_), C, int(np.max(clf.n_iter_)),
             clf.score(x_train, y_train), time.perf_counter() - t0)
    return clf


def fit_temperature_scaled(raw: LogisticRegression, x_train: np.ndarray, y_train: np.ndarray,
                           C: float, max_iter: int) -> TemperatureScaledProbe:
    """Temperature scaling fitted on out-of-fold TRAIN logits (5 folds): the
    fold models never see the rows they score, and the test split never
    enters. The deployed model is the full-train probe with that T."""
    t0 = time.perf_counter()
    cv = StratifiedKFold(n_splits=CALIBRATION_FOLDS, shuffle=True, random_state=0)
    oof = cross_val_predict(LogisticRegression(max_iter=max_iter, class_weight="balanced", C=C),
                            x_train, y_train, cv=cv, method="decision_function")
    y_idx = np.searchsorted(raw.classes_, y_train)
    T = fit_temperature(np.asarray(oof), y_idx)
    log.info("temperature scaling: T = %.3f from %d-fold out-of-fold logits, %.1fs",
             T, CALIBRATION_FOLDS, time.perf_counter() - t0)
    return TemperatureScaledProbe(raw, T)


def fit_sigmoid_cv(x_train: np.ndarray, y_train: np.ndarray, C: float, max_iter: int
                   ) -> CalibratedClassifierCV:
    """sklearn's Platt calibration (per-class sigmoids, 5 folds). Kept only as a
    comparison in metrics.json: it re-ranks test images, so it is not deployed."""
    cal = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=max_iter, class_weight="balanced", C=C),
        method="sigmoid", cv=CALIBRATION_FOLDS)
    cal.fit(x_train, y_train)
    return cal


def expected_calibration_error(proba: np.ndarray, y_true: np.ndarray, classes: np.ndarray,
                               n_bins: int = ECE_BINS) -> float:
    """ECE on the top-1 probability: equal-width bins on [0, 1], weighted mean of
    |accuracy - confidence| per bin."""
    conf = proba.max(axis=1)
    hit = (classes[proba.argmax(axis=1)] == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(ece)


# ----------------------------------------------------------- method C: nearest neighbour
def nearest_neighbour(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Return (top-1 genus, majority-of-3 genus) per test row."""
    sims = x_test @ x_train.T
    nn3 = np.argsort(-sims, axis=1)[:, :3]
    top1 = y_train[nn3[:, 0]]
    vote = np.empty_like(top1)
    for i, row in enumerate(y_train[nn3]):
        counts = Counter(row)
        best = max(counts.values())
        # first label in nearest-first order that reaches the max count -> tie = nearest
        vote[i] = next(g for g in row if counts[g] == best)
    return top1, vote


# --------------------------------------------------------------------------- metrics
def summarise(name: str, y_true: np.ndarray, top1: np.ndarray, top3_hit: np.ndarray,
              genera: np.ndarray) -> dict:
    m = {
        "top1": float((top1 == y_true).mean()),
        "top3": float(top3_hit.mean()),
        "macro_f1": float(f1_score(y_true, top1, labels=genera, average="macro",
                                   zero_division=0)),
    }
    log.info("%-20s top-1 %.3f  top-3 %.3f  macro-F1 %.3f", name, m["top1"], m["top3"],
             m["macro_f1"])
    return m


def per_genus_table(index_test: pd.DataFrame, preds: dict[str, np.ndarray],
                    genera: np.ndarray, subfamily_of: dict[str, str]) -> pd.DataFrame:
    y_true = index_test["genus"].to_numpy()
    table = pd.DataFrame({
        "genus": genera,
        "subfamily": [subfamily_of[g] for g in genera],
        "n_test": [int((y_true == g).sum()) for g in genera],
    })
    for name, top1 in preds.items():
        _, rec, f1, _ = precision_recall_fscore_support(
            y_true, top1, labels=genera, zero_division=0)
        table[f"recall_{name}"] = rec.round(3)
        table[f"f1_{name}"] = f1.round(3)
    return table


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, order: list[str],
                   subfamily_of: dict[str, str], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)  # font cache chatter
    import matplotlib.pyplot as plt

    pos = {g: i for i, g in enumerate(order)}
    n = len(order)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[pos[t], pos[p]] += 1

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import viz  # shared palette: subfamily bands use the same slot order as the web app

    fig, ax = plt.subplots(figsize=(15, 13.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(order, rotation=90, fontsize=13, fontstyle="italic")
    ax.set_yticklabels(order, fontsize=13, fontstyle="italic")
    ax.tick_params(length=0, pad=22)
    ax.set_xlabel("predicted genus", fontsize=14, labelpad=10)
    ax.set_ylabel("true genus", fontsize=14, labelpad=10)
    ax.set_title(f"Linear probe confusion matrix: {len(y_true)} test images, "
                 f"{n} genera, grouped by subfamily", fontsize=16, loc="left", pad=16)
    # subfamily blocks: hairlines between them and a coloured band with the name
    # along the left and bottom edges (same colour slots as the web atlas).
    sub_counts: dict[str, int] = {}
    for t in y_true:
        sub_counts[subfamily_of[t]] = sub_counts.get(subfamily_of[t], 0) + 1
    sub_order = sorted(sub_counts, key=lambda k: -sub_counts[k])
    colour = {k: v[0] for k, v in viz.styles(sub_order).items()}
    from matplotlib.patches import Rectangle
    blocks: list[tuple[str, int, int]] = []  # (subfamily, start, end) in row index
    start = 0
    for i in range(1, n + 1):
        if i == n or subfamily_of[order[i]] != subfamily_of[order[start]]:
            blocks.append((subfamily_of[order[start]], start, i))
            start = i
    for i in range(1, n):
        if subfamily_of[order[i]] != subfamily_of[order[i - 1]]:
            ax.axhline(i - 0.5, color="#888888", lw=0.8)
            ax.axvline(i - 0.5, color="#888888", lw=0.8)
    from matplotlib.patches import Patch
    for sub, a, b in blocks:
        c = colour.get(sub, viz.OTHER_COLOR)
        ax.add_patch(Rectangle((-0.9, a - 0.5), 0.3, b - a, color=c, clip_on=False, lw=0))
        ax.add_patch(Rectangle((a - 0.5, n - 0.4), b - a, 0.3, color=c, clip_on=False, lw=0))
    handles = [Patch(color=colour.get(sub, viz.OTHER_COLOR), label=f"{sub} ({b - a} genera)" if b - a > 1 else f"{sub} (1 genus)")
               for sub, a, b in blocks]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=11, frameon=False,
              title="subfamily (band colour)", title_fontsize=11, handlelength=1.2, borderaxespad=0)
    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                        fontweight="bold" if i == j else "normal",
                        color="white" if cm[i, j] > thresh else "#222222")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.55, anchor=(0.0, 1.0))
    cb.set_label("count", fontsize=12)
    cb.ax.tick_params(labelsize=11)
    # bbox_inches="tight" so the clip_on=False bands and the legend outside the
    # axes are included instead of cropping the y tick labels.
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    log.info("confusion matrix written to %s", out)


def errors_table(index_test: pd.DataFrame, proba: np.ndarray, classes: np.ndarray
                 ) -> pd.DataFrame:
    order = np.argsort(-proba, axis=1)
    pred, second = classes[order[:, 0]], classes[order[:, 1]]
    p1 = proba[np.arange(len(proba)), order[:, 0]]
    p2 = proba[np.arange(len(proba)), order[:, 1]]
    source = pd.read_csv(DATA / "dataset.csv", usecols=["specimen_code", "image_source"])
    df = pd.DataFrame({
        "specimen_code": index_test["specimen_code"].to_numpy(),
        "genus_true": index_test["genus"].to_numpy(),
        "genus_pred": pred, "prob": p1.round(3),
        "second_guess": second, "second_prob": p2.round(3),
        "image_path": index_test["image_path"].to_numpy(),
    }).merge(source, on="specimen_code", how="left")
    return (df[df["genus_true"] != df["genus_pred"]]
            .sort_values("prob", ascending=False).reset_index(drop=True))


# --------------------------------------------------------------------------- main
def main() -> None:
    setup_logging(REPORTS / "07_eval.log")
    cfg = load_config(ROOT / "config.yaml")
    t_run = time.perf_counter()

    if not (EMB_PATH.exists() and INDEX_PATH.exists()):
        log.error("embeddings missing; run 06_embed.py first")
        sys.exit(1)
    embs, index = load_embeddings()
    is_train = (index["split"] == "train").to_numpy()
    is_test = (index["split"] == "test").to_numpy()
    assert is_train.sum() + is_test.sum() == len(index), "unexpected split values"
    x_train, y_train = embs[is_train], index.loc[is_train, "genus"].to_numpy()
    x_test, y_test = embs[is_test], index.loc[is_test, "genus"].to_numpy()
    index_test = index[is_test].reset_index(drop=True)
    genera = np.array(sorted(index["genus"].unique()))
    subfamily_of = index.drop_duplicates("genus").set_index("genus")["subfamily"].to_dict()
    missing = set(genera) - set(y_train)
    assert not missing, f"genera absent from train: {missing}"
    log.info("split: %d train / %d test, %d genera", len(x_train), len(x_test), len(genera))

    metrics: dict[str, dict] = {}
    top1_preds: dict[str, np.ndarray] = {}

    # A: zero-shot
    for name, scores in zero_shot_scores(cfg["embed_model"], genera, x_test).items():
        top3 = topk_from_scores(scores, genera)
        top1_preds[name] = top3[:, 0]
        metrics[name] = summarise(name, y_test, top3[:, 0], (top3 == y_test[:, None]).any(1),
                                  genera)

    # B: linear probe, raw ...
    raw = fit_probe(x_train, y_train, float(cfg["probe_C"]), int(cfg["probe_max_iter"]))
    proba_raw = raw.predict_proba(x_test)
    top3_raw = topk_from_scores(proba_raw, raw.classes_)
    metrics["linear_probe_uncalibrated"] = summarise(
        "linear_probe_uncal", y_test, top3_raw[:, 0], (top3_raw == y_test[:, None]).any(1), genera)
    joblib.dump(raw, PROBE_RAW_PATH)
    # ... and temperature-scaled (the deployed model: "linear_probe" everywhere below)
    clf = fit_temperature_scaled(raw, x_train, y_train, float(cfg["probe_C"]), int(cfg["probe_max_iter"]))
    assert list(clf.classes_) == list(raw.classes_)
    proba = clf.predict_proba(x_test)
    assert (proba.argmax(1) == proba_raw.argmax(1)).all(), "temperature scaling must not re-rank"
    top3 = topk_from_scores(proba, clf.classes_)
    top1_preds["linear_probe"] = top3[:, 0]
    metrics["linear_probe"] = summarise("linear_probe", y_test, top3[:, 0],
                                        (top3 == y_test[:, None]).any(1), genera)
    joblib.dump(clf, PROBE_PATH)
    log.info("calibrated probe saved to %s, raw probe to %s (%d classes)",
             PROBE_PATH, PROBE_RAW_PATH, len(clf.classes_))
    sig = fit_sigmoid_cv(x_train, y_train, float(cfg["probe_C"]), int(cfg["probe_max_iter"]))
    proba_sig = sig.predict_proba(x_test)
    top3_sig = topk_from_scores(proba_sig, sig.classes_)
    sig_metrics = summarise("sigmoid_cv (not used)", y_test, top3_sig[:, 0],
                            (top3_sig == y_test[:, None]).any(1), genera)
    calibration = {
        "method": "temperature scaling (api/probe.py)", "temperature": clf.temperature,
        "cv_folds": CALIBRATION_FOLDS, "fitted_on": "train only (out-of-fold logits)",
        "ece_bins": ECE_BINS,
        "sigmoid_cv_alternative": {
            "note": "sklearn CalibratedClassifierCV(sigmoid, cv=5) on train; rejected: per-class sigmoids re-rank",
            **sig_metrics,
            "top1_changed_images": int((top3_raw[:, 0] != top3_sig[:, 0]).sum()),
            "mean_max_prob": float(proba_sig.max(axis=1).mean()),
            "ece": expected_calibration_error(proba_sig, y_test, sig.classes_),
        },
        "top1_changed_images": int((top3_raw[:, 0] != top3[:, 0]).sum()),
        "top3_set_changed_images": int((np.sort(top3_raw, axis=1) != np.sort(top3, axis=1)).any(1).sum()),
        "mean_max_prob_before": float(proba_raw.max(axis=1).mean()),
        "mean_max_prob_after": float(proba.max(axis=1).mean()),
        "ece_before": expected_calibration_error(proba_raw, y_test, raw.classes_),
        "ece_after": expected_calibration_error(proba, y_test, clf.classes_),
        "share_top1_below_0.5_before": float((proba_raw.max(axis=1) < 0.5).mean()),
        "share_top1_below_0.5_after": float((proba.max(axis=1) < 0.5).mean()),
    }
    log.info("calibration: top-1 changed on %d / %d test images, top-3 set on %d; "
             "mean max-prob %.3f -> %.3f; ECE(%d bins) %.3f -> %.3f; top-1 < 0.5: %.1f%% -> %.1f%%",
             calibration["top1_changed_images"], len(x_test), calibration["top3_set_changed_images"],
             calibration["mean_max_prob_before"], calibration["mean_max_prob_after"], ECE_BINS,
             calibration["ece_before"], calibration["ece_after"],
             100 * calibration["share_top1_below_0.5_before"], 100 * calibration["share_top1_below_0.5_after"])

    # C: nearest neighbour
    nn1, nn_vote = nearest_neighbour(x_train, y_train, x_test)
    top1_preds["nearest_neighbour"] = nn1
    metrics["nearest_neighbour"] = summarise("nearest_neighbour", y_test, nn1,
                                             nn_vote == y_test, genera)

    # reports
    out = {
        "n_train": int(len(x_train)), "n_test": int(len(x_test)), "n_genera": int(len(genera)),
        "embed_model": cfg["embed_model"],
        "probe": {"C": float(cfg["probe_C"]), "max_iter": int(cfg["probe_max_iter"]),
                  "class_weight": "balanced", "calibration": "temperature scaling, T fitted on 5-fold out-of-fold train logits"},
        "calibration": calibration,
        "zero_shot_templates": TEMPLATES,
        "nearest_neighbour_top3": "majority vote of 3 nearest train embeddings",
        "methods": {m: metrics[m] for m in [*METHODS, "linear_probe_uncalibrated"]},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    log.info("metrics written to %s", REPORTS / "metrics.json")

    per_genus = per_genus_table(index_test, top1_preds, genera, subfamily_of)
    per_genus.to_csv(REPORTS / "per_genus.csv", index=False)
    log.info("per-genus table written to %s", REPORTS / "per_genus.csv")

    order = sorted(genera, key=lambda g: (subfamily_of[g], g))
    plot_confusion(y_test, top1_preds["linear_probe"], order, subfamily_of,
                   REPORTS / "confusion_matrix.png")

    errors = errors_table(index_test, proba, clf.classes_)
    errors.to_csv(REPORTS / "errors.csv", index=False)
    log.info("probe errors: %d / %d written to %s", len(errors), len(x_test),
             REPORTS / "errors.csv")
    pairs = Counter(zip(errors["genus_true"], errors["genus_pred"]))
    for (t, p), c in pairs.most_common(5):
        log.info("  %-14s -> %-14s x%d", t, p, c)

    log.info("done in %.0fs", time.perf_counter() - t_run)


if __name__ == "__main__":
    main()

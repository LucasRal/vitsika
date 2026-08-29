#!/usr/bin/env python3
"""Phase C — evaluate genus classification on the frozen BioCLIP 2 embeddings.

Reads data/embeddings.npy + data/embeddings_index.csv (from 06_embed.py) and
uses the index's split column (train / test). Three methods:

- A, zero-shot: BioCLIP 2 text tower on the genus name, two prompts
  ("a photo of {genus}, a genus of ant" and the bare "{genus}").
- B, linear probe: sklearn LogisticRegression (balanced, C from config.yaml)
  on the train embeddings; the fitted estimator is saved to data/probe.pkl
  (joblib) for the demo API.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import load_config, quiet_logging, setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

EMB_PATH = DATA / "embeddings.npy"
INDEX_PATH = DATA / "embeddings_index.csv"
PROBE_PATH = DATA / "probe.pkl"

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

    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    labels = [f"{g}  ({subfamily_of[g][:5]})" for g in order]
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted genus"); ax.set_ylabel("true genus")
    ax.set_title(f"Linear probe confusion matrix — {len(y_true)} test images, "
                 f"{n} genera (ordered by subfamily)", fontsize=11)
    # subfamily boundaries
    for i in range(1, n):
        if subfamily_of[order[i]] != subfamily_of[order[i - 1]]:
            ax.axhline(i - 0.5, color="#888888", lw=0.8)
            ax.axvline(i - 0.5, color="#888888", lw=0.8)
    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > thresh else "#222222")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="count")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
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
        log.error("embeddings missing — run 06_embed.py first")
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

    # A — zero-shot
    for name, scores in zero_shot_scores(cfg["embed_model"], genera, x_test).items():
        top3 = topk_from_scores(scores, genera)
        top1_preds[name] = top3[:, 0]
        metrics[name] = summarise(name, y_test, top3[:, 0], (top3 == y_test[:, None]).any(1),
                                  genera)

    # B — linear probe
    clf = fit_probe(x_train, y_train, float(cfg["probe_C"]), int(cfg["probe_max_iter"]))
    proba = clf.predict_proba(x_test)
    top3 = topk_from_scores(proba, clf.classes_)
    top1_preds["linear_probe"] = top3[:, 0]
    metrics["linear_probe"] = summarise("linear_probe", y_test, top3[:, 0],
                                        (top3 == y_test[:, None]).any(1), genera)
    joblib.dump(clf, PROBE_PATH)
    log.info("probe saved to %s (%d classes, coef %s)", PROBE_PATH, len(clf.classes_),
             clf.coef_.shape)

    # C — nearest neighbour
    nn1, nn_vote = nearest_neighbour(x_train, y_train, x_test)
    top1_preds["nearest_neighbour"] = nn1
    metrics["nearest_neighbour"] = summarise("nearest_neighbour", y_test, nn1,
                                             nn_vote == y_test, genera)

    # reports
    out = {
        "n_train": int(len(x_train)), "n_test": int(len(x_test)), "n_genera": int(len(genera)),
        "embed_model": cfg["embed_model"],
        "probe": {"C": float(cfg["probe_C"]), "max_iter": int(cfg["probe_max_iter"]),
                  "class_weight": "balanced"},
        "zero_shot_templates": TEMPLATES,
        "nearest_neighbour_top3": "majority vote of 3 nearest train embeddings",
        "methods": {m: metrics[m] for m in METHODS},
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

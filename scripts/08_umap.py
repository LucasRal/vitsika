#!/usr/bin/env python3
"""Phase C — 2-D UMAP of the BioCLIP 2 embeddings, for eyeballing structure.

Fits UMAP (cosine metric, parameters in config.yaml, random_state = seed) on
all rows of data/embeddings.npy and writes data/umap_coords.csv
(specimen_code, x, y — same order as embeddings_index.csv) and the fitted
reducer to data/umap_model.pkl (joblib) so the API can place new query
embeddings on the same map with reducer.transform(). When a coords file
already exists the re-fit is asserted identical to it (the fit is
deterministic: random_state = seed forces single-threaded layout).

Figures: reports/umap_by_subfamily.png (every subfamily coloured) and
reports/umap_by_genus.png (the 8 largest genera coloured, the rest grey;
the linear-probe test errors from reports/errors.csv are marked with a
black x when that file exists).

The log lists claims that can be checked against the figures: silhouette
of subfamilies / genera in the 2-D space vs the 768-d space, and for the
genus pairs the probe confused in Phase C, how far apart their UMAP
centroids sit relative to the within-genus spread. Log: reports/08_umap.log.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import load_config, setup_logging  # noqa: E402
import viz  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

COORDS_PATH = DATA / "umap_coords.csv"
MODEL_PATH = DATA / "umap_model.pkl"
N_GENERA_COLOURED = 8
# genus pairs the linear probe confused (reports/eval_notes.md)
CONFUSED_PAIRS = [("Syllophopsis", "Tetramorium"), ("Royidris", "Monomorium"),
                  ("Camponotus", "Aphaenogaster"), ("Pachycondyla", "Bothroponera"),
                  ("Technomyrmex", "Tapinoma"), ("Nesomyrmex", "Crematogaster")]

log = logging.getLogger("umap")


def fit_umap(embs: np.ndarray, cfg: dict) -> tuple[np.ndarray, "umap.UMAP"]:
    import warnings

    logging.getLogger("numba").setLevel(logging.WARNING)
    import umap  # slow import (numba JIT)

    # random_state forces n_jobs=1; umap warns about it every run
    warnings.filterwarnings("ignore", message="n_jobs value .* overridden")

    t0 = time.perf_counter()
    reducer = umap.UMAP(n_neighbors=int(cfg["umap_n_neighbors"]),
                        min_dist=float(cfg["umap_min_dist"]), metric="cosine",
                        random_state=int(cfg["seed"]))
    coords = reducer.fit_transform(embs)
    log.info("UMAP n_neighbors=%d min_dist=%g metric=cosine seed=%d on %s: %.1fs",
             reducer.n_neighbors, reducer.min_dist, reducer.random_state, embs.shape,
             time.perf_counter() - t0)
    return coords.astype(np.float32), reducer


def assert_unchanged(index: pd.DataFrame, coords: np.ndarray) -> None:
    """A re-run must reproduce the committed coordinates (to the 4 decimals
    the CSV carries); otherwise the atlas served by the API would drift."""
    old = pd.read_csv(COORDS_PATH)
    assert list(old["specimen_code"]) == list(index["specimen_code"]), \
        "specimen order differs from the existing umap_coords.csv"
    new = np.round(coords.astype(np.float64), 4)
    diff = np.abs(old[["x", "y"]].to_numpy() - new).max()
    assert diff <= 1e-4, f"UMAP re-fit differs from {COORDS_PATH.name}: max |delta| {diff:.4g}"
    log.info("re-fit reproduces the existing %s (max |delta| %.1e)", COORDS_PATH.name, diff)


def plot(index: pd.DataFrame, coords: np.ndarray, by: str, order: list[str],
         other_label: str | None, errors: pd.DataFrame | None, out: Path, title: str) -> None:
    plt = viz.pyplot()
    fig, ax = plt.subplots(figsize=(11, 8))
    counts = index[by].value_counts().to_dict()
    viz.scatter_by_category(ax, coords[:, 0], coords[:, 1], index[by], order, counts,
                            other_label=other_label)
    if errors is not None and not errors.empty:
        m = index["specimen_code"].isin(errors["specimen_code"]).to_numpy()
        ax.scatter(coords[m, 0], coords[m, 1], s=46, marker="x", c=viz.INK, linewidths=0.9,
                   label=f"probe test error ({int(m.sum())})", zorder=3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=1.4,
                  labelcolor=viz.INK)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("figure written to %s", out)


def log_claims(index: pd.DataFrame, embs: np.ndarray, coords: np.ndarray,
               errors: pd.DataFrame | None) -> None:
    for level in ("subfamily", "genus"):
        labels = index[level].to_numpy()
        s_hi = silhouette_score(embs, labels, metric="cosine")
        s_2d = silhouette_score(coords, labels)
        log.info("silhouette by %-9s 768-d cosine %.3f | UMAP 2-D %.3f  (%d classes)",
                 level, s_hi, s_2d, len(set(labels)))

    cent = {g: coords[(index["genus"] == g).to_numpy()].mean(0) for g in index["genus"].unique()}
    spread = {g: float(np.linalg.norm(coords[(index["genus"] == g).to_numpy()] - cent[g], axis=1)
                       .mean()) for g in cent}
    log.info("median within-genus spread (mean distance to centroid): %.2f UMAP units",
             float(np.median(list(spread.values()))))
    log.info("probe-confused pairs — centroid distance vs the two genera's spread:")
    for a, b in CONFUSED_PAIRS:
        if a not in cent or b not in cent:
            continue
        d = float(np.linalg.norm(cent[a] - cent[b]))
        verdict = ("overlapping" if d < max(spread[a], spread[b])
                   else "adjacent" if d < 2 * max(spread[a], spread[b]) else "separated")
        log.info("  %-13s / %-13s  centroids %.2f apart, spreads %.2f / %.2f -> %s",
                 a, b, d, spread[a], spread[b], verdict)

    # each probe test error: which genera surround it in the 2-D map?
    if errors is not None and not errors.empty:
        log.info("UMAP neighbourhood of each probe test error (10 nearest specimens):")
        codes = index["specimen_code"].to_numpy()
        genus = index["genus"].to_numpy()
        for e in errors.itertuples():
            i = int(np.flatnonzero(codes == e.specimen_code)[0])
            d = np.linalg.norm(coords - coords[i], axis=1)
            d[i] = np.inf
            nb = genus[np.argsort(d)[:10]]
            top = ", ".join(f"{g} {n}" for g, n in
                            sorted(dict(zip(*np.unique(nb, return_counts=True))).items(),
                                   key=lambda kv: -kv[1])[:3])
            log.info("  %-14s true %-13s pred %-13s neighbours: %s",
                     e.specimen_code, e.genus_true, e.genus_pred, top)

    # for each genus: nearest other-genus centroid in UMAP space
    names = list(cent)
    log.info("nearest neighbouring genus centroid in the 2-D map:")
    for g in sorted(names):
        others = [(float(np.linalg.norm(cent[g] - cent[h])), h) for h in names if h != g]
        d, h = min(others)
        log.info("  %-14s -> %-14s %.2f", g, h, d)


def main() -> None:
    setup_logging(REPORTS / "08_umap.log")
    cfg = load_config(ROOT / "config.yaml")

    if not (DATA / "embeddings.npy").exists():
        log.error("embeddings missing — run 06_embed.py first")
        sys.exit(1)
    embs = np.load(DATA / "embeddings.npy")
    index = pd.read_csv(DATA / "embeddings_index.csv")
    assert len(embs) == len(index)
    log.info("embeddings %s, %d genera, %d subfamilies", embs.shape,
             index["genus"].nunique(), index["subfamily"].nunique())

    coords, reducer = fit_umap(embs, cfg)
    if COORDS_PATH.exists():
        assert_unchanged(index, coords)
    joblib.dump(reducer, MODEL_PATH)
    log.info("reducer written to %s (%.1f MB)", MODEL_PATH, MODEL_PATH.stat().st_size / 1e6)
    pd.DataFrame({"specimen_code": index["specimen_code"], "x": coords[:, 0],
                  "y": coords[:, 1]}).to_csv(COORDS_PATH, index=False, float_format="%.4f")
    log.info("coordinates written to %s", COORDS_PATH)

    errors_path = REPORTS / "errors.csv"
    errors = pd.read_csv(errors_path) if errors_path.exists() else None

    subfamilies = index["subfamily"].value_counts().index.tolist()  # slot order = size
    plot(index, coords, "subfamily", subfamilies, None, None,
         REPORTS / "umap_by_subfamily.png",
         f"BioCLIP 2 embeddings, UMAP — {len(index)} specimens by subfamily")
    genera = index["genus"].value_counts().index[:N_GENERA_COLOURED].tolist()
    plot(index, coords, "genus", genera, "other genera", errors,
         REPORTS / "umap_by_genus.png",
         f"BioCLIP 2 embeddings, UMAP — {N_GENERA_COLOURED} largest genera "
         f"(of {index['genus'].nunique()})")

    log_claims(index, embs, coords, errors)
    log.info("done")


if __name__ == "__main__":
    main()

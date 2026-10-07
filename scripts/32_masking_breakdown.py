#!/usr/bin/env python3
"""Item 3, step 5: per-genus and per-group breakdown of the masking scores.

Per-image predictions are recomputed with exactly the settings of
scripts/31_score_variants.py (A: deployed probe on the test split;
B: LogisticRegression C=1, max_iter 5000, balanced, 5-fold StratifiedKFold,
shuffle, seed 42, all 1,219 rows) and checked against reports/masking/scores.csv.

1. Ant-only drop per genus: B recall on original minus ant_grey. For the 5
   genera that drop most: mask area, partial ratings (exclude.csv,
   handcheck.csv), ant_grey recall of partial and small-mask images against
   the rest, ant_erased recall (B) and the metadata-only recall
   (reports/confounds/metadata_only_per_genus.csv), and where their ant_grey
   errors go.
2. Strumigenys: recall on every version, A and B.
3. Accuracy by group, A and B, original / ant_grey / ant_erased: image source
   (Commons vs GBIF-cache crops) and background brightness typical vs
   atypical for the genus (the definition of scripts/22: |brightness - genus
   median| / genus sd above its median over all 1,236 images, from
   reports/confounds/image_stats.csv).

Reads:  data/masking/embeddings_<version>.npy, data/masking/embeddings_index.csv,
        data/probe.pkl (read only), data/segmented/manifest.csv,
        reports/masking/scores.csv, reports/masking/per_genus.csv,
        reports/segmentation/exclude.csv, reports/segmentation/handcheck.csv,
        reports/confounds/image_stats.csv, reports/confounds/metadata_only_per_genus.csv
Writes: reports/masking/breakdown.csv, reports/masking/figures/per_genus_recall.png,
        reports/masking/32_masking_breakdown.log
Offline.
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # api.probe, to unpickle the deployed probe
sys.path.insert(0, str(ROOT / "scripts"))
from gbif_client import setup_logging  # noqa: E402
from viz import INK, MUTED, PALETTE, pyplot  # noqa: E402

MASK = ROOT / "data" / "masking"
REPORT = ROOT / "reports" / "masking"
SEGREP = ROOT / "reports" / "segmentation"
CONF = ROOT / "reports" / "confounds"
VERSIONS = ["original", "ant_grey", "ant_erased", "scalebar_erased"]
FIG_VERSIONS = ["original", "ant_grey", "ant_erased"]
SEED = 42
TOP = 5

log = logging.getLogger("masking_breakdown")


def predictions(index: pd.DataFrame) -> pd.DataFrame:
    """One row per image and version: pred_A (test rows only) and pred_B."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn minor-version unpickle note
        probe = joblib.load(ROOT / "data" / "probe.pkl")
    test = (index["split"] == "test").to_numpy()
    y = index["genus"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    out = []
    for v in VERSIONS:
        emb = np.load(MASK / f"embeddings_{v}.npy")
        pa = np.full(len(index), None, dtype=object)
        pa[test] = probe.predict(emb[test])
        pb = cross_val_predict(LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"),
                               emb, y, cv=cv, n_jobs=5)
        out.append(index.assign(version=v, pred_A=pa, pred_B=pb))
    return pd.concat(out, ignore_index=True)


def check_against_scores(p: pd.DataFrame) -> None:
    s = pd.read_csv(REPORT / "scores.csv")
    s = s[s["subset"] == "all"].set_index(["scheme", "version"])["accuracy"]
    for v in VERSIONS:
        q = p[p["version"] == v]
        a = (q["pred_A"] == q["genus"])[q["split"] == "test"].mean()
        b = (q["pred_B"] == q["genus"]).mean()
        assert abs(a - s[("A_deployed_probe", v)]) < 1e-3 and abs(b - s[("B_retrained_cv", v)]) < 1e-3, v
    log.info("per-image predictions reproduce reports/masking/scores.csv (accuracy, all rows)")


def genus_drop(p: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    pg = pd.read_csv(REPORT / "per_genus.csv")
    pg = pg[pg["subset"] == "all"]
    wide = pg.pivot_table(index=["scheme", "genus"], columns="version", values="recall")
    n = pg[pg["version"] == "original"].set_index(["scheme", "genus"])["n"]
    b, a = wide.loc["B_retrained_cv"], wide.loc["A_deployed_probe"]
    d = pd.DataFrame({
        "n_B": n.loc["B_retrained_cv"], "B_original": b["original"], "B_ant_grey": b["ant_grey"],
        "B_ant_erased": b["ant_erased"], "B_drop": b["original"] - b["ant_grey"],
        "n_A": n.loc["A_deployed_probe"], "A_original": a["original"], "A_ant_grey": a["ant_grey"],
        "A_drop": a["original"] - a["ant_grey"],
    })

    man = pd.read_csv(ROOT / "data" / "segmented" / "manifest.csv")
    man = man[man["specimen_code"].isin(index["specimen_code"])]
    exc = pd.read_csv(SEGREP / "exclude.csv")
    hc = pd.read_csv(SEGREP / "handcheck.csv")
    meta = pd.read_csv(CONF / "metadata_only_per_genus.csv")

    d["mask_area_median"] = man.groupby("genus")["mask_area_fraction"].median()
    d["n_components_30px_gt1"] = man.assign(m=man["n_components_30px"] > 1).groupby("genus")["m"].mean()
    d["partial_n"] = index.groupby("genus")["partial"].sum()
    d["handcheck_n"] = hc.groupby("genus").size()
    d["handcheck_partial_or_bad"] = hc[hc["rating"] != "good"].groupby("genus").size()
    d = d.fillna({"handcheck_n": 0, "handcheck_partial_or_bad": 0, "partial_n": 0})
    d["metadata_only_recall_forest"] = meta.set_index("genus")["recall_non_ant_forest"]

    # within genus, B on ant_grey: are the errors on partial or small-mask images?
    g = p[p["version"] == "ant_grey"].merge(man[["specimen_code", "mask_area_fraction"]], on="specimen_code")
    g["ok"] = g["pred_B"] == g["genus"]
    g["small_mask"] = g["mask_area_fraction"] < g.groupby("genus")["mask_area_fraction"].transform("median")
    d["ant_grey_B_recall_small_mask"] = g[g["small_mask"]].groupby("genus")["ok"].mean()
    d["ant_grey_B_recall_large_mask"] = g[~g["small_mask"]].groupby("genus")["ok"].mean()
    d["ant_grey_B_recall_partial"] = g[g["partial"] == 1].groupby("genus")["ok"].mean()
    d["ant_grey_B_errors_to"] = g[~g["ok"]].groupby("genus")["pred_B"].agg(
        lambda s: "; ".join(f"{k} {v}" for k, v in s.value_counts().head(3).items()))
    d = d.sort_values("B_drop", ascending=False)
    d.index.name = "genus"
    return d


def groups(p: pd.DataFrame, index: pd.DataFrame) -> list[dict]:
    st = pd.read_csv(CONF / "image_stats.csv")
    dev = st.groupby("genus")["border_brightness"].transform(lambda v: (v - v.median()).abs() / v.std())
    st["atypical_bg"] = dev > dev.median()
    p = p.merge(st[["specimen_code", "atypical_bg"]], on="specimen_code", how="left")
    assert p["atypical_bg"].notna().all()
    defs = {
        "Commons": p["image_source"] == "commons",
        "GBIF-cache crops": p["image_source"] == "gbif_cache",
        "Background typical for the genus": ~p["atypical_bg"],
        "Background atypical for the genus": p["atypical_bg"],
    }
    rows = []
    for name, m in defs.items():
        for v in FIG_VERSIONS:
            q = p[m & (p["version"] == v)]
            t = q[q["split"] == "test"]
            rows.append({"section": "3_group_accuracy", "subject": name, "version": v,
                         "A_n": len(t), "A_value": (t["pred_A"] == t["genus"]).mean(),
                         "B_n": len(q), "B_value": (q["pred_B"] == q["genus"]).mean()})
    return rows


def figure(d: pd.DataFrame, path: Path) -> None:
    plt = pyplot()
    order = d.sort_values("B_original").index  # bottom to top
    fig, axes = plt.subplots(1, 2, figsize=(9, 7.5), sharey=True)
    y = np.arange(len(order))
    for ax, scheme, title in [(axes[0], "B", "B: retrained, 5-fold CV (1,219 images)"),
                              (axes[1], "A", "A: deployed probe, test split (245 images)")]:
        for i, (v, mk) in enumerate(zip(FIG_VERSIONS, ["o", "s", "^"])):
            col = f"{scheme}_{v}" if f"{scheme}_{v}" in d else None
            if col is None:
                continue
            ax.scatter(d.loc[order, col], y + (i - 1) * 0.22, s=22, marker=mk, c=PALETTE[i],
                       edgecolors="white", linewidths=0.3, label=v, zorder=2)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("recall")
        ax.grid(True, axis="x", linewidth=0.5)
    n = d.loc[order, "n_B"].astype(int)
    axes[0].set_yticks(y, [f"{g} ({k})" for g, k in zip(order, n)], color=INK)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, labelcolor=INK)
    fig.text(0.01, 0.005, "Per-genus recall by image version. Genus (images); A has 2 to 47 test images "
             "per genus, so its recalls are coarse.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    setup_logging(REPORT / "32_masking_breakdown.log")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    index = pd.read_csv(MASK / "embeddings_index.csv")
    p = predictions(index)
    check_against_scores(p)

    d = genus_drop(p, index)
    pg = pd.read_csv(REPORT / "per_genus.csv")
    a_erased = pg[(pg["subset"] == "all") & (pg["scheme"] == "A_deployed_probe")
                  & (pg["version"] == "ant_erased")].set_index("genus")["recall"]
    d["A_ant_erased"] = a_erased
    log.info("1. B recall drop original -> ant_grey, all genera:\n%s",
             d[["n_B", "B_original", "B_ant_grey", "B_drop", "B_ant_erased", "A_drop",
                "mask_area_median", "partial_n"]].round(3).to_string())
    top = d.head(TOP)
    log.info("1. top %d:\n%s", TOP, top.drop(columns=["n_A"]).round(3).to_string())
    log.info("all genera: median mask area %.3f; partial images %d of %d",
             d["mask_area_median"].median(), int(index["partial"].sum()), len(index))

    strum = pg[(pg["genus"] == "Strumigenys") & (pg["subset"] == "all")]
    log.info("2. Strumigenys recall:\n%s",
             strum.pivot_table(index="version", columns="scheme", values="recall").round(3).to_string())
    sp = p[p["version"] == "ant_erased"]
    to_s = sp[(sp["genus"] != "Strumigenys") & (sp["pred_B"] == "Strumigenys")]
    log.info("2. ant_erased, B: %d non-Strumigenys images predicted as Strumigenys", len(to_s))

    grp = groups(p, index)
    g = pd.DataFrame(grp)
    log.info("3. accuracy by group:\n%s", g.round(3).to_string(index=False))

    rows = []
    for genus, r in d.iterrows():
        rows.append({"section": "1_genus_drop", "subject": genus, "top5": genus in top.index,
                     **{k: r[k] for k in d.columns}})
    for v, q in strum.groupby("version", sort=False):
        a, b = (q[q["scheme"] == s].iloc[0] for s in ("A_deployed_probe", "B_retrained_cv"))
        rows.append({"section": "2_strumigenys", "subject": "Strumigenys", "version": v,
                     "A_n": a["n"], "A_value": a["recall"], "B_n": b["n"], "B_value": b["recall"]})
    rows += grp
    out = pd.DataFrame(rows)
    lead = ["section", "subject", "version", "A_n", "A_value", "B_n", "B_value", "top5"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(4)
    out.to_csv(REPORT / "breakdown.csv", index=False)
    figure(d, REPORT / "figures" / "per_genus_recall.png")
    log.info("wrote reports/masking/breakdown.csv (%d rows) and figures/per_genus_recall.png", len(out))


if __name__ == "__main__":
    main()

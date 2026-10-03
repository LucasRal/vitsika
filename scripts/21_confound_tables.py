#!/usr/bin/env python3
"""Genus x photo-condition tables and one association summary.

Reads reports/confounds/image_stats.csv (scripts/20_image_stats.py) and
writes to reports/confounds/tables/:

  genus_x_<variable>_counts.csv / _rowpct.csv   one pair per categorical variable
  single_photographer_genera.csv                genera with >= 90 % of images
                                                from one photographer
  genus_type_fraction.csv                       type-specimen share per genus
  genus_numeric_medians.csv                     per-genus medians of the numbers

and to reports/confounds/association_summary.csv / .md: for every variable
its association with genus. Categorical: Cramer's V (raw and bias-corrected,
Bergsma 2013). Numeric: eta squared from a one-way ANOVA. Because 27 genera
and sparse tables inflate both measures, each is compared with its value
under 1,000 random permutations of the genus labels (seed 42): the table
gives the permutation mean, the 99th percentile and a permutation p-value.

Offline. Bins: background brightness in 4 quartile bands; warm/cool from
R - B of the border median: cool < 0 <= neutral < 10 <= warm.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "reports" / "confounds"
TABLES = CONF / "tables"
N_PERM = 1000
SEED = 42
SINGLE_SOURCE = 0.90
WARM_COOL_EDGES = (0, 10)

log = logging.getLogger("confound_tables")

CATEGORICAL = {  # column -> description
    "creator": "photographer (creator)",
    "source3": "image source: commons original / commons thumbnail / gbif cache",
    "brightness_band": "background brightness, quartile bands",
    "warm_cool_band": "background warm/cool (R - B of border median)",
    "rights_years": "year range in rightsHolder",
    "type_status": "typeStatus (non-types as 'not a type')",
    "is_type": "type specimen yes/no",
    "scalebar_style": "scale bar style: black bar / thin line / none detected",
}
NUMERIC = {
    "border_brightness": "background brightness (mean luminance of the 5 % border)",
    "warm_cool": "background R - B",
    "border_r": "background median R",
    "border_g": "background median G",
    "border_b": "background median B",
    "width": "image width (px)",
    "height": "image height (px)",
    "aspect": "aspect ratio",
    "file_bytes": "file size (bytes)",
    "dark_frac": "dark-pixel share (ant + pin size proxy; partly the ant itself)",
    "rights_year_end": "end year of the rightsHolder range",
    "scalebar_frac_width": "scale bar length / image width (status found only; also tracks ant size)",
}


def load() -> pd.DataFrame:
    s = pd.read_csv(CONF / "image_stats.csv")
    q = s["border_brightness"].quantile([0, .25, .5, .75, 1]).round(0).astype(int).tolist()
    labels = [f"B{i + 1} {q[i]}-{q[i + 1]}" for i in range(4)]
    s["brightness_band"] = pd.qcut(s["border_brightness"], 4, labels=labels).astype(str)
    lo, hi = WARM_COOL_EDGES
    s["warm_cool_band"] = np.select([s["warm_cool"] < lo, s["warm_cool"] < hi],
                                    [f"cool (<{lo})", f"neutral ({lo}-{hi - 1})"], f"warm (>={hi})")
    s["rights_years"] = (s["rights_year_start"].astype("Int64").astype(str) + "-"
                         + s["rights_year_end"].astype("Int64").astype(str))
    s["type_status"] = s["typeStatus"].fillna("not a type")
    s["is_type"] = s["is_type"].map({True: "type", False: "not a type"})
    return s


def genus_order(s: pd.DataFrame) -> list[str]:
    return s.drop_duplicates("genus").sort_values(["subfamily", "genus"])["genus"].tolist()


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """(raw V, bias-corrected V) for two categorical series."""
    t = pd.crosstab(x, y).to_numpy()
    n = t.sum()
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    r, k = t.shape
    phi2 = chi2 / n
    raw = np.sqrt(phi2 / max(1, min(r, k) - 1))
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    corr = np.sqrt(phi2c / max(1e-12, min(rc, kc) - 1))
    return float(raw), float(corr)


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    grand = values.mean()
    ss_tot = ((values - grand) ** 2).sum()
    ss_b = sum(len(v) * (v.mean() - grand) ** 2
               for v in (values[groups == g] for g in np.unique(groups)))
    return float(ss_b / ss_tot) if ss_tot > 0 else np.nan


def permutation(stat, genus: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.array([stat(rng.permutation(genus)) for _ in range(N_PERM)])


def crosstab_files(s: pd.DataFrame, col: str, order: list[str]) -> pd.DataFrame:
    counts = pd.crosstab(s["genus"], s[col]).reindex(order).fillna(0).astype(int)
    counts = counts[counts.sum().sort_values(ascending=False).index]
    counts["total"] = counts.sum(axis=1)
    pct = (counts.drop(columns="total").div(counts["total"], axis=0) * 100).round(1)
    pct.insert(0, "n", counts["total"])
    counts.to_csv(TABLES / f"genus_x_{col}_counts.csv")
    pct.to_csv(TABLES / f"genus_x_{col}_rowpct.csv")
    return pct


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def f(v):
        return floatfmt.format(v) if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    setup_logging(CONF / "21_confound_tables.log")
    s = load()
    order = genus_order(s)
    rng = np.random.default_rng(SEED)
    genus = s["genus"].to_numpy()

    for col in CATEGORICAL:
        crosstab_files(s, col, order)

    # genera dominated by one photographer
    pct = pd.read_csv(TABLES / "genus_x_creator_rowpct.csv", index_col=0)
    top = pct.drop(columns="n").idxmax(axis=1)
    single = pd.DataFrame({"n": pct["n"], "top_photographer": top,
                           "top_share_pct": pct.drop(columns="n").max(axis=1)})
    single = single[single["top_share_pct"] >= SINGLE_SOURCE * 100].sort_values(
        "top_share_pct", ascending=False)
    single.to_csv(TABLES / "single_photographer_genera.csv")
    log.info("genera with >= %.0f %% from one photographer: %d of %d (%s)",
             SINGLE_SOURCE * 100, len(single), len(order), single["top_photographer"].value_counts().to_dict())

    # type-specimen fraction
    t = s.groupby("genus").agg(n=("genus", "size"), n_type=("typeStatus", lambda x: x.notna().sum()))
    t["type_frac"] = (t["n_type"] / t["n"]).round(3)
    t = t.join(pd.crosstab(s["genus"], s["type_status"])).reindex(order)
    t.to_csv(TABLES / "genus_type_fraction.csv")

    med = s.groupby("genus")[list(NUMERIC)].median().reindex(order).round(4)
    med.insert(0, "n", s.groupby("genus").size().reindex(order))
    med.to_csv(TABLES / "genus_numeric_medians.csv")

    rows = []
    for col, desc in CATEGORICAL.items():
        x = s[col].astype(str)
        raw, corr = cramers_v(x, s["genus"])
        null = permutation(lambda g: cramers_v(x, pd.Series(g))[1], genus, rng)
        rows.append({"variable": col, "type": "categorical", "measure": "Cramer's V (bias-corrected)",
                     "value": corr, "raw_value": raw, "null_mean": null.mean(),
                     "null_p99": np.quantile(null, .99),
                     "perm_p": (1 + (null >= corr).sum()) / (N_PERM + 1),
                     "n": len(x), "levels": x.nunique(), "description": desc})
        log.info("%-18s V=%.3f (raw %.3f) null mean %.3f", col, corr, raw, null.mean())
    for col, desc in NUMERIC.items():
        m = s[col].notna()
        v, g = s.loc[m, col].to_numpy(float), genus[m]
        eta = eta_squared(v, g)
        null = permutation(lambda gg: eta_squared(v, gg), g, rng)
        f_p = stats.f_oneway(*[v[g == k] for k in np.unique(g)]).pvalue
        rows.append({"variable": col, "type": "numeric", "measure": "eta squared (ANOVA)",
                     "value": eta, "raw_value": np.nan, "null_mean": null.mean(),
                     "null_p99": np.quantile(null, .99),
                     "perm_p": (1 + (null >= eta).sum()) / (N_PERM + 1), "anova_p": f_p,
                     "n": int(m.sum()), "levels": np.nan, "description": desc})
        log.info("%-18s eta2=%.3f null mean %.3f (n=%d)", col, eta, null.mean(), m.sum())
    summ = pd.DataFrame(rows)
    # V and eta (sqrt of eta squared) are both on a 0-1 correlation-like scale
    summ["effect"] = np.where(summ["type"] == "numeric", np.sqrt(summ["value"]), summ["value"])
    summ = summ.sort_values("effect", ascending=False)
    summ.to_csv(CONF / "association_summary.csv", index=False)

    view = summ[["variable", "measure", "value", "effect", "null_mean", "null_p99", "perm_p", "n", "description"]].copy()
    view["value"] = view["value"].round(3)
    view["effect"] = view["effect"].round(3)
    view["null_mean"] = view["null_mean"].round(3)
    view["null_p99"] = view["null_p99"].round(3)
    view["perm_p"] = view["perm_p"].map(lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001")
    md = ["# Association of photo conditions with genus", "",
          "Generated by `scripts/21_confound_tables.py` from `image_stats.csv` "
          f"({len(s)} images, {len(order)} genera). Cramer's V is bias-corrected (Bergsma 2013); "
          "eta squared is between-genus variance over total variance. 'null mean' and 'null p99' "
          f"are the measure under {N_PERM} random permutations of the genus labels (seed {SEED}), "
          "the level reached with no link at all; perm p is the share of permutations at or above "
          "the observed value. 0 = no association, 1 = genus fully determines the variable. "
          "'effect' puts both on one scale for ranking: V itself, or eta (the square root of "
          "eta squared, the correlation ratio).", "",
          md_table(view, "{:.3f}"), "",
          f"Genera with >= {SINGLE_SOURCE:.0%} of images from one photographer: "
          f"{len(single)} of {len(order)}; see `tables/single_photographer_genera.csv`.", ""]
    (CONF / "association_summary.md").write_text("\n".join(md))
    log.info("wrote %s and %d tables in %s", CONF / "association_summary.md",
             len(list(TABLES.glob('*.csv'))), TABLES)


if __name__ == "__main__":
    main()

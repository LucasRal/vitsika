#!/usr/bin/env python3
"""Phase 3 — build the classification dataset from harvested metadata.

From data/raw/records.parquet: drop label views, keep configured castes
(default: worker), keep one image per specimen (profile first, dorsal
fallback), drop genera below min_specimens_per_genus, then do a stratified
80/20 split by genus NAME (grouped by specimen_code) with a fixed seed.
Genera whose name merges several GBIF genusKeys are logged.

Writes data/dataset.csv and reports/dataset_stats.md. Deterministic and safe
to re-run (outputs are rewritten).
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import load_config, setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
REPORTS = ROOT / "reports"

log = logging.getLogger("build_dataset")


def log_merged_genus_keys(df: pd.DataFrame) -> list[str]:
    per_name = df.groupby("genus")["genusKey"].nunique()
    merged = per_name[per_name > 1]
    notes = []
    for genus in merged.index:
        keys = sorted(df.loc[df["genus"] == genus, "genusKey"].unique())
        note = f"{genus}: merged {len(keys)} genusKeys {keys}"
        log.info("genus name merge — %s", note)
        notes.append(note)
    return notes


def caste_drop_table(spec: pd.DataFrame, castes: list[str]) -> pd.DataFrame:
    """Per genus: specimens kept vs dropped for null sex vs queen/male/other."""
    sex = spec["sex"].fillna("").str.lower()
    category = pd.Series("kept (" + "/".join(castes) + ")", index=spec.index)
    category[~sex.isin(castes)] = "dropped: " + sex[~sex.isin(castes)]
    category[sex == ""] = "dropped: sex null"
    table = pd.crosstab(spec["genus"], category)
    table["total"] = table.sum(axis=1)
    return table.sort_values("total", ascending=False)


def pick_one_image_per_specimen(df: pd.DataFrame, views_priority: list[str]) -> pd.DataFrame:
    rank = {v: i for i, v in enumerate(views_priority)}
    df = df[df["view"].isin(rank)].copy()
    df["view_rank"] = df["view"].map(rank)
    df = df.sort_values(["specimen_code", "view_rank", "image_url"])
    return df.drop_duplicates("specimen_code", keep="first").drop(columns=["view_rank"])


def split_by_specimen(df: pd.DataFrame, test_fraction: float, seed: int) -> pd.Series:
    rng = random.Random(seed)
    split_map: dict[str, str] = {}
    for _, group in df.groupby("genus", sort=True):
        codes = sorted(group["specimen_code"].unique())
        rng.shuffle(codes)
        n_test = max(1, round(len(codes) * test_fraction))
        for code in codes[:n_test]:
            split_map[code] = "test"
        for code in codes[n_test:]:
            split_map[code] = "train"
    return df["specimen_code"].map(split_map)


def md_table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return lines


def write_stats(
    df: pd.DataFrame,
    raw: pd.DataFrame,
    kept_counts: pd.Series,
    dropped_counts: pd.Series,
    caste_table: pd.DataFrame,
    merge_notes: list[str],
    cfg: dict,
) -> None:
    lines = ["# Dataset stats", ""]
    lines.append(f"- Images (one per specimen): **{len(df)}**")
    lines.append(f"- Genera kept (>= {cfg['min_specimens_per_genus']} specimens): **{len(kept_counts)}**")
    lines.append(f"- Genera dropped: **{len(dropped_counts)}**")
    lines.append(f"- Share of rows with coordinates: **{df['decimalLatitude'].notna().mean():.1%}**")
    lines.append("")

    if merge_notes:
        lines.append("## Genus names merging several genusKeys")
        lines.append("")
        lines += [f"- {n}" for n in merge_notes]
        lines.append("")

    lines.append("## Media items per occurrence record (raw harvest)")
    lines.append("")
    per_record = raw.groupby("gbifID").size()
    lines.append(f"- Records: {per_record.size}; mean {per_record.mean():.2f}, max {per_record.max()}")
    dist = per_record.value_counts().sort_index()
    lines += md_table(["media items", "records"], [[k, v] for k, v in dist.items()])
    lines.append("")

    lines.append("## View distribution before filtering (raw media rows)")
    lines.append("")
    views = raw["view"].replace("", "?").value_counts()
    lines += md_table(["view", "media rows"], [[k, v] for k, v in views.items()])
    lines.append("")

    lines.append("## Specimen view coverage (raw harvest)")
    lines.append("")
    views_per_spec = raw.groupby("specimen_code")["view"].agg(set)
    n_p = int(views_per_spec.map(lambda s: "p" in s).sum())
    n_d_only = int(views_per_spec.map(lambda s: "p" not in s and "d" in s).sum())
    n_neither = int(views_per_spec.map(lambda s: not ({"p", "d"} & s)).sum())
    lines += md_table(
        ["coverage", "specimens"],
        [["has profile shot", n_p],
         ["dorsal only (no profile)", n_d_only],
         ["neither profile nor dorsal", n_neither]],
    )
    lines.append("")

    lines.append("## Caste filter per genus (specimens with usable views)")
    lines.append("")
    lines += md_table(
        ["genus"] + list(caste_table.columns),
        [[genus] + list(row) for genus, row in caste_table.iterrows()],
    )
    lines.append("")

    lines.append("## Images per genus per split")
    lines.append("")
    pivot = df.pivot_table(index="genus", columns="split", values="specimen_code",
                           aggfunc="count", fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    lines += md_table(
        ["genus", "train", "test", "total"],
        [[g, r.get("train", 0), r.get("test", 0), r["total"]] for g, r in pivot.iterrows()],
    )
    lines.append("")

    lines.append("## Genera dropped (below threshold)")
    lines.append("")
    if dropped_counts.empty:
        lines.append("None.")
    else:
        lines += md_table(
            ["genus", "specimens"],
            [[g, n] for g, n in dropped_counts.sort_values(ascending=False).items()],
        )
    lines.append("")

    lines.append("## Top 15 stateProvince values")
    lines.append("")
    top = df["stateProvince"].fillna("(none)").value_counts().head(15)
    lines += md_table(["stateProvince", "images"], [[p, n] for p, n in top.items()])
    lines.append("")

    out = REPORTS / "dataset_stats.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("stats written to %s", out)


def main() -> None:
    setup_logging(REPORTS / "03_build_dataset.log")
    cfg = load_config(ROOT / "config.yaml")

    src = RAW / "records.parquet"
    if not src.exists():
        log.error("%s missing — run 02_harvest.py first", src)
        sys.exit(1)
    raw = pd.read_parquet(src)
    log.info("input: %d media rows, %d records, %d specimens",
             len(raw), raw["gbifID"].nunique(), raw["specimen_code"].nunique())

    merge_notes = log_merged_genus_keys(raw)

    df = raw[raw["view"].isin(["h", "d", "p"])]
    log.info("after dropping label/unknown views: %d rows", len(df))

    castes = [c.lower() for c in cfg["castes"]]
    caste_table = caste_drop_table(df.drop_duplicates("specimen_code"), castes)
    df = df[df["sex"].fillna("").str.lower().isin(castes)]
    log.info("after caste filter %s: %d rows, %d specimens",
             castes, len(df), df["specimen_code"].nunique())

    df = pick_one_image_per_specimen(df, cfg["views_priority"])
    log.info("one image per specimen (%s priority): %d rows",
             "/".join(cfg["views_priority"]), len(df))

    counts = df.groupby("genus")["specimen_code"].nunique()
    kept_counts = counts[counts >= cfg["min_specimens_per_genus"]]
    dropped_counts = counts[counts < cfg["min_specimens_per_genus"]]
    df = df[df["genus"].isin(kept_counts.index)].copy()
    log.info("genera kept: %d (dropped %d below %d specimens); %d images",
             len(kept_counts), len(dropped_counts), cfg["min_specimens_per_genus"], len(df))

    df["split"] = split_by_specimen(df, cfg["test_fraction"], cfg["seed"])
    log.info("split: %s", df["split"].value_counts().to_dict())

    out = DATA / "dataset.csv"
    df.to_csv(out, index=False)
    log.info("dataset written to %s", out)

    write_stats(df, raw, kept_counts, dropped_counts, caste_table, merge_notes, cfg)


if __name__ == "__main__":
    main()

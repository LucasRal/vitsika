#!/usr/bin/env python3
"""Geographic / temporal coverage of the full manifest (data/dataset_full.csv,
metadata only — no images needed).

stateProvince spellings are normalised with config.yaml `province_aliases`
(Majunga -> Mahajanga, Toliary -> Toliara, Diego-Suarez -> Antsiranana).

Outputs: data/geo_summary.csv (per genus: specimens, species, provinces,
elevation min/median/max, year range), reports/map_specimens.png (specimen
coordinates coloured by subfamily; the island outline emerges from the
points themselves, no basemap) and the coverage gaps (least-sampled
provinces / decades, genera confined to one province) in
reports/09_geo.log.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_client import load_config, setup_logging  # noqa: E402
import viz  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

log = logging.getLogger("geo")


def normalise_provinces(df: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    before = df["stateProvince"].value_counts(dropna=False)
    df["province"] = df["stateProvince"].str.strip().replace(aliases)
    for old, new in aliases.items():
        n = int(before.get(old, 0))
        if n:
            log.info("province alias %-13s -> %-12s (%d rows)", old, new, n)
    log.info("provinces: %s (%d rows without province)",
             df["province"].value_counts().to_dict(), int(df["province"].isna().sum()))
    return df


def genus_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for genus, g in df.groupby("genus"):
        named = g.loc[g["taxonRank"].isin(["SPECIES", "SUBSPECIES"]), "species"].dropna()
        provinces = sorted(g["province"].dropna().unique())
        rows.append({
            "genus": genus,
            "subfamily": g["subfamily"].iloc[0],
            "n_specimens": len(g),
            "n_species": int(named.nunique()),
            "n_unidentified": int(g["species"].isna().sum()),
            "n_provinces": len(provinces),
            "provinces": ";".join(provinces),
            "n_with_coords": int(g["decimalLatitude"].notna().sum()),
            "elev_min": g["elevation"].min(),
            "elev_median": g["elevation"].median(),
            "elev_max": g["elevation"].max(),
            "year_min": g["year"].min(),
            "year_max": g["year"].max(),
            "n_without_date": int(g["year"].isna().sum()),
        })
    out = pd.DataFrame(rows).sort_values(["n_specimens", "genus"], ascending=[False, True])
    for c in ("elev_min", "elev_median", "elev_max", "year_min", "year_max"):
        out[c] = out[c].round().astype("Int64")  # medians can be x.5; NaN stays <NA>
    return out.reset_index(drop=True)


def plot_map(df: pd.DataFrame, out: Path) -> None:
    plt = viz.pyplot()
    d = df.dropna(subset=["decimalLatitude", "decimalLongitude"])
    fig, ax = plt.subplots(figsize=(8.5, 11))
    order = d["subfamily"].value_counts().index.tolist()
    viz.scatter_by_category(ax, d["decimalLongitude"], d["decimalLatitude"], d["subfamily"],
                            order, d["subfamily"].value_counts().to_dict(), size=12)
    # 1° of longitude is shorter than 1° of latitude at these latitudes:
    ax.set_aspect(1 / np.cos(np.radians(d["decimalLatitude"].mean())))
    ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude (°N)")
    ax.set_title(f"AntWeb Malagasy specimens in the manifest — {len(d)} of {len(df)} "
                 f"with coordinates, by subfamily", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("map written to %s", out)


def log_gaps(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    prov = df["province"].value_counts()
    log.info("specimens per province (least covered first): %s",
             prov.sort_values().to_dict())
    genera_per_prov = df.dropna(subset=["province"]).groupby("province")["genus"].nunique()
    log.info("genera per province (of %d): %s", df["genus"].nunique(),
             genera_per_prov.sort_values().to_dict())

    dec = (df["year"] // 10 * 10).dropna().astype(int).value_counts().sort_index()
    log.info("specimens per decade: %s (%d undated)", dec.to_dict(), int(df["year"].isna().sum()))
    log.info("least covered decades: %s", dec.sort_values().head(3).to_dict())

    single = summary[summary["n_provinces"] <= 1]
    log.info("genera recorded in <= 1 province: %s",
             {r.genus: f"{r.n_specimens} spp. in {r.provinces or 'none'}"
              for r in single.itertuples()})
    no_coords = summary[summary["n_with_coords"] < summary["n_specimens"]]
    log.info("%d specimens without coordinates (%d genera affected)",
             int(df["decimalLatitude"].isna().sum()), len(no_coords))

    elev = pd.cut(df["elevation"], [0, 250, 500, 1000, 1500, 2200], right=False)
    log.info("specimens per elevation band (m): %s (%d without elevation)",
             {str(k): int(v) for k, v in elev.value_counts().sort_index().items()},
             int(df["elevation"].isna().sum()))


def main() -> None:
    setup_logging(REPORTS / "09_geo.log")
    cfg = load_config(ROOT / "config.yaml")

    src = DATA / "dataset_full.csv"
    if not src.exists():
        log.error("%s missing — run 03_build_dataset.py first", src)
        sys.exit(1)
    df = pd.read_csv(src)
    log.info("manifest: %d specimens, %d genera, %d subfamilies", len(df),
             df["genus"].nunique(), df["subfamily"].nunique())
    df = normalise_provinces(df, cfg["province_aliases"])
    df["year"] = pd.to_numeric(df["eventDate"].astype("string").str[:4], errors="coerce")

    summary = genus_summary(df)
    summary.to_csv(DATA / "geo_summary.csv", index=False)
    log.info("summary written to %s (%d genera)", DATA / "geo_summary.csv", len(summary))

    plot_map(df, REPORTS / "map_specimens.png")
    log_gaps(df, summary)
    log.info("done")


if __name__ == "__main__":
    main()

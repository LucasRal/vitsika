#!/usr/bin/env python3
"""Figures for the confound analysis, in reports/confounds/figures/.

  genus_x_photographer_heatmap.png   share of each genus's images per
                                     photographer (row %), counts in cells
  background_strips.png              per genus, 8 random images (seed 42)
                                     sorted by background brightness; rows
                                     ordered by the genus's median brightness
  brightness_vs_warmcool.png         background brightness vs warm/cool,
                                     one small panel per genus (that genus
                                     highlighted over all images in grey):
                                     27 genera are too many for one colour each
  scalebar_validation_<style>.jpg    random detections per scale-bar style
                                     (seed 42), bottom 40 % of each image with
                                     the detected bar boxed, for checking
                                     scripts/20_image_stats.py by eye

Offline; reads reports/confounds/image_stats.csv and the images.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz import GRID, INK, MUTED, PALETTE, pyplot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "reports" / "confounds"
FIG = CONF / "figures"
SEED = 42
ACCENT = PALETTE[0]
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", ACCENT])


def genus_order(s: pd.DataFrame) -> list[str]:
    return s.drop_duplicates("genus").sort_values(["subfamily", "genus"])["genus"].tolist()


def heatmap(s: pd.DataFrame, plt) -> None:
    order = genus_order(s)
    counts = pd.crosstab(s["genus"], s["creator"]).reindex(order)
    counts = counts[counts.sum().sort_values(ascending=False).index]
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    sub = s.drop_duplicates("genus").set_index("genus")["subfamily"]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(pct.to_numpy(), cmap=SEQ, vmin=0, vmax=100, aspect="auto")
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            c = counts.iat[i, j]
            if c:
                ax.text(j, i, str(c), ha="center", va="center", fontsize=7,
                        color="white" if pct.iat[i, j] > 55 else INK)
    ax.set_xticks(range(pct.shape[1]))
    ax.set_xticklabels([f"{c} ({counts[c].sum()})" for c in pct.columns], rotation=55, ha="right")
    ax.set_yticks(range(pct.shape[0]))
    ax.set_yticklabels([f"{g} ({sub[g][:5]}., n={counts.loc[g].sum()})" for g in pct.index])
    ax.set_xticks(np.arange(-.5, pct.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, pct.shape[0]), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("share of the genus's images (%)", color=MUTED)
    cb.outline.set_visible(False)
    ax.set_title("Genus x photographer: colour = row %, number = images. Genera ordered by subfamily.",
                 loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "genus_x_photographer_heatmap.png", dpi=130)
    plt.close(fig)


def strips(s: pd.DataFrame, plt, n: int = 8) -> None:
    med = s.groupby("genus")["border_brightness"].median().sort_values()
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(med), n, figsize=(n * 1.55, len(med) * 1.12))
    for i, g in enumerate(med.index):
        rows = s[s["genus"] == g]
        pick = rows.iloc[rng.choice(len(rows), size=min(n, len(rows)), replace=False)]
        pick = pick.sort_values("border_brightness")
        for j in range(n):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if j >= len(pick):
                continue
            r = pick.iloc[j]
            with Image.open(ROOT / r["image_path"]) as im:
                im = im.convert("RGB")
                im.thumbnail((300, 300))
                ax.imshow(np.asarray(im))
            ax.set_title(f"{r['border_brightness']:.0f}", fontsize=6, color=MUTED, pad=1)
        axes[i, 0].set_ylabel(f"{g}\nmedian {med[g]:.0f}", rotation=0, ha="right", va="center",
                              fontsize=8, color=INK)
    fig.suptitle("8 random images per genus (seed 42), sorted by background brightness (number above "
                 "each image, 0-255).\nGenera ordered from darkest to lightest median background.",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.965, bottom=0.005, wspace=0.04, hspace=0.25)
    fig.savefig(FIG / "background_strips.png", dpi=110)
    plt.close(fig)


def scatter(s: pd.DataFrame, plt) -> None:
    med = s.groupby("genus")["border_brightness"].median().sort_values()
    cols = 6
    nrow = int(np.ceil(len(med) / cols))
    fig, axes = plt.subplots(nrow, cols, figsize=(cols * 2.4, nrow * 2.2), sharex=True, sharey=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, g in zip(axes.flat, med.index):
        ax.set_visible(True)
        m = s["genus"] == g
        ax.scatter(s["border_brightness"], s["warm_cool"], s=4, c=GRID, linewidths=0, zorder=1)
        ax.scatter(s.loc[m, "border_brightness"], s.loc[m, "warm_cool"], s=9, c=ACCENT,
                   edgecolors="white", linewidths=0.3, zorder=2)
        ax.scatter([s.loc[m, "border_brightness"].median()], [s.loc[m, "warm_cool"].median()],
                   marker="+", s=90, c=INK, linewidths=1.4, zorder=3)
        ax.axhline(0, color=MUTED, linewidth=0.5, zorder=0)
        ax.set_title(f"{g} (n={int(m.sum())})", fontsize=8, loc="left")
        ax.grid(True, linewidth=0.4)
        for sp in ax.spines.values():
            sp.set_visible(False)
    used = len(med)
    for j in range(cols):  # x labels on the lowest visible panel of each column
        last = (used - 1 - j) // cols * cols + j if j < used else None
        if last is not None:
            ax = axes.flat[last]
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("background brightness")
    for ax in axes[:, 0]:
        ax.set_ylabel("warm (+) / cool (-)\nR - B")
    fig.suptitle("Background brightness vs warm/cool. Each panel: one genus (blue) over all 1,236 "
                 "images (grey); + = genus median. Panels ordered by median brightness.",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "brightness_vs_warmcool.png", dpi=130)
    plt.close(fig)


def validation_sheets(s: pd.DataFrame, n: int = 40, cols: int = 5, w: int = 360, h: int = 170) -> None:
    for style in ("black_bar", "thin_line", "none"):
        rows = s[s["scalebar_style"] == style]
        rows = rows.sample(min(n, len(rows)), random_state=SEED)
        sheet = Image.new("RGB", (cols * w, int(np.ceil(len(rows) / cols)) * h), "white")
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(rows.itertuples()):
            with Image.open(ROOT / r.image_path) as im:
                im = im.convert("RGB")
            if r.scalebar_status in ("found", "truncated"):
                cx, cy = r.scalebar_x_frac * im.width, r.scalebar_y_frac * im.height
                half = (r.scalebar_px / 2 if r.scalebar_status == "found" else 25) + 4
                ImageDraw.Draw(im).rectangle([cx - half, cy - 9, cx + half, cy + 9],
                                             outline=(220, 0, 0), width=4)
            top = 0.6 if style != "none" else 0.0
            im = im.crop((0, int(im.height * top), im.width, im.height))
            im.thumbnail((w - 6, h - 18))
            x, y = (i % cols) * w, (i // cols) * h
            sheet.paste(im, (x + 3, y))
            d.text((x + 4, y + h - 15), f"{i} {r.specimen_code} {r.genus} [{r.scalebar_status}]", fill="black")
        sheet.save(FIG / f"scalebar_validation_{style}.jpg", quality=85)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    s = pd.read_csv(CONF / "image_stats.csv")
    heatmap(s, plt)
    strips(s, plt)
    scatter(s, plt)
    validation_sheets(s)
    for p in sorted(FIG.iterdir()):
        print(p.relative_to(ROOT), f"{p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()

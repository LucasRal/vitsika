"""Shared plotting conventions for the report figures.

Categorical colours are assigned in a FIXED order (never cycled) from an
8-slot CVD-checked palette; a marker shape is paired with each slot as a
secondary encoding so identity never rests on colour alone. Anything past
8 categories folds into 'other' (grey)."""
from __future__ import annotations

import logging

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
OTHER_COLOR, OTHER_MARKER = "#b5b3ac", "."
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def pyplot():
    """matplotlib.pyplot with the Agg backend and recessive chrome."""
    import matplotlib
    matplotlib.use("Agg")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
        "legend.frameon": False, "font.size": 9,
    })
    return plt


def styles(categories: list[str]) -> dict[str, tuple[str, str]]:
    """category -> (colour, marker); order of `categories` = slot order."""
    out = {}
    for i, c in enumerate(categories):
        if i < len(PALETTE):
            out[c] = (PALETTE[i], MARKERS[i])
        else:
            out[c] = (OTHER_COLOR, OTHER_MARKER)
    return out


def scatter_by_category(ax, x, y, cat, order: list[str], counts: dict[str, int],
                        other_label: str | None = None, size: float = 14) -> None:
    """One scatter call per category (fixed colour + marker), legend with counts."""
    sty = styles(order)
    if other_label is not None:
        mask = ~cat.isin(order)
        if mask.any():
            ax.scatter(x[mask], y[mask], s=size * 0.6, c=OTHER_COLOR, marker=OTHER_MARKER,
                       alpha=0.6, linewidths=0, label=f"{other_label} ({int(mask.sum())})",
                       zorder=1)
    for c in order:
        m = cat == c
        colour, marker = sty[c]
        ax.scatter(x[m], y[m], s=size, c=colour, marker=marker, alpha=0.85,
                   edgecolors="white", linewidths=0.3, label=f"{c} ({counts[c]})", zorder=2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=1.4,
              labelcolor=INK)
    ax.grid(True, linewidth=0.5)
    for s in ax.spines.values():
        s.set_visible(False)

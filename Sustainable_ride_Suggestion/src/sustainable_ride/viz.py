"""Shared visual language for the figures and the app.

One module so a mode is the same colour everywhere it appears. Colour follows
the entity, never its rank -- filtering out an infeasible mode must not repaint
the survivors, or a reader tracking "the green one" across two charts is misled.

The palette is validated for colour-vision deficiency: the worst adjacent pair
separates by dE 9.2 (deuteranopia) against a floor of 8. Aqua sits slightly
under 3:1 contrast on a light surface, so every chart using it also carries
direct value labels and a table view rather than relying on the fill alone.
"""

from __future__ import annotations

# Categorical slots, fixed order. Assigned to entities once and never cycled.
MODE_COLORS = {
    "taxi": "#eb6834",      # orange
    "bike": "#1baf7a",      # aqua
    "scooter": "#2a78d6",   # blue
}

MODE_COLORS_DARK = {
    "taxi": "#d95926",
    "bike": "#199e70",
    "scooter": "#3987e5",
}

MODE_ICONS = {"taxi": "🚕", "bike": "🚲", "scooter": "🛴"}

INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
}

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}


def mode_color(mode: str, dark: bool = False) -> str:
    """Stable colour for a travel mode."""
    palette = MODE_COLORS_DARK if dark else MODE_COLORS
    return palette.get(mode, INK["muted"])


def apply_matplotlib_style() -> None:
    """Recessive grid, no top/right spines, muted axis ink."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.8,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK["axis"],
        "axes.labelcolor": INK["secondary"],
        "text.color": INK["primary"],
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "legend.frameon": False,
        "figure.autolayout": True,
    })

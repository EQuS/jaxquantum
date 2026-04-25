"""Shared matplotlib style for landing-page demo plots.

Import `apply_style()` and the color constants from any plot script to
keep the landing-page figures visually consistent.
"""

import matplotlib.pyplot as plt

# Theme colors — chosen to read on both light and dark page backgrounds.
PURPLE = "#7e57c2"
TEAL   = "#26a69a"
ORANGE = "#ef6c00"


def apply_style():
    """Apply the shared landing-page plot style. Saves are transparent by default."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.linewidth": 1.2,
        "axes.edgecolor": "#888",
        "xtick.color": "#888",
        "ytick.color": "#888",
        "axes.labelcolor": "#666",
        "axes.titlecolor": "#444",
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "savefig.transparent": True,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })

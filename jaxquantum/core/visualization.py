"""
Visualization utils.
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

from jaxquantum.core.qutip import jqt2qt

WIGNER = "wigner"
QFUNC = "qfunc"


def plot_qp(state, pts, ax=None, contour=True, qp_type=WIGNER):
    """Plot quasi-probability distribution.

    TODO: decouple this from qutip.

    Args:
        state: statevector
        pts: points to evaluate quasi-probability distribution on
        dim: dimensions of state
        ax: matplotlib axis to plot on
        contour: make the plot use contouring
        qp_type: type of quasi probability distribution ("wigner", "qfunc")

    Returns:
        axis on which the plot was plotted.
    """
    pts = np.array(pts)
    state = jqt2qt(state)
    if ax is None:
        _, ax = plt.subplots(1, figsize=(4, 3), dpi=200)
    # fig = ax.get_figure()

    if qp_type == WIGNER:
        vmin = -1
        vmax = 1
        scale = np.pi / 2
        cmap = "seismic"
    elif qp_type == QFUNC:
        vmin = 0
        vmax = 1
        scale = np.pi
        cmap = "jet"

    QP = scale * getattr(qt, qp_type)(state, pts, pts, g=2)

    if contour:
        im = ax.contourf(
            pts,
            pts,
            QP,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            levels=np.linspace(vmin, vmax, 101),
        )
    else:
        im = ax.pcolormesh(
            pts,
            pts,
            QP,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    ax.axhline(0, linestyle="-", color="black", alpha=0.7)
    ax.axvline(0, linestyle="-", color="black", alpha=0.7)
    ax.grid()
    ax.set_aspect("equal", adjustable="box")
    return im


plot_wigner = lambda state, pts, ax=None, contour=True: plot_qp(
    state, pts, ax=ax, contour=contour, qp_type=WIGNER
)

plot_qfunc = lambda state, pts, ax=None, contour=True: plot_qp(
    state, pts, ax=ax, contour=contour, qp_type=QFUNC
)

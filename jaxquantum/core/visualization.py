"""
Visualization utils.
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

from jaxquantum.core.conversions import jqt2qt
from jaxquantum.core.wigner import wigner, qfunc
import jax.numpy as jnp

WIGNER = "wigner"
QFUNC = "qfunc"


def plot_qp(
    state,
    pts_x,
    pts_y=None,
    axs=None,
    contour=True,
    qp_type=WIGNER,
    cbar_label="",
    axis_scale_factor=1,
    plot_cbar=True,
    x_ticks=None,
    y_ticks=None,
    z_ticks=None,
):
    """Plot quasi-probability distribution.


    Args:
        state: statevector
        pts: points to evaluate quasi-probability distribution on
        dim: dimensions of state
        axs: matplotlib axis to plot on
        contour: make the plot use contouring
        qp_type: type of quasi probability distribution ("wigner", "qfunc")

    Returns:
        axis on which the plot was plotted.
    """
    if pts_y is None:
        pts_y = pts_x
    pts_x = jnp.array(pts_x)
    pts_y = jnp.array(pts_y)

    bdims = state.bdims
    extra_dims = bdims[2:]
    if extra_dims != ():
        state = state.reshape_bdims(
            bdims[0] * int(jnp.prod(jnp.array(extra_dims))), bdims[1]
        )

    if axs is None:
        _, axs = plt.subplots(
            state.bdims[0],
            state.bdims[1],
            figsize=(4 * state.bdims[1], 3 * state.bdims[0]),
            dpi=200,
        )

    if qp_type == WIGNER:
        vmin = -1
        vmax = 1
        scale = np.pi / 2
        cmap = "seismic"
        QP = scale * wigner(state, pts_x, pts_y, g=2)

    elif qp_type == QFUNC:
        vmin = 0
        vmax = 1
        scale = np.pi
        cmap = "jet"
        QP = scale * qfunc(state, pts_x, pts_y, g=jnp.sqrt(2))



    pts_x = pts_x * axis_scale_factor
    pts_y = pts_y * axis_scale_factor


    x_ticks = jnp.linspace(jnp.min(pts_x), jnp.max(pts_x), 5) if x_ticks is None else x_ticks
    y_ticks = jnp.linspace(jnp.min(pts_y), jnp.max(pts_y), 5) if y_ticks is None else y_ticks
    z_ticks = jnp.linspace(vmin, vmax, 11) if z_ticks is None else z_ticks

    for row in range(state.bdims[0]):
        for col in range(state.bdims[1]):
            if contour:
                ax = axs[row, col]
                im = ax.contourf(
                    pts_x,
                    pts_y,
                    QP[row, col],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    levels=np.linspace(vmin, vmax, 101),
                )
            else:
                im = ax.pcolormesh(
                    pts_x,
                    pts_y,
                    QP[row, col],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.axhline(0, linestyle="-", color="black", alpha=0.7)
            ax.axvline(0, linestyle="-", color="black", alpha=0.7)
            ax.grid()
            ax.set_aspect("equal", adjustable="box")

            if plot_cbar:
                cbar = plt.colorbar(
                    im, ax=ax, orientation="vertical", ticks=np.linspace(-1, 1, 11)
                )
                cbar.ax.set_title(cbar_label)
                cbar.set_ticks(z_ticks)

            ax.set_xlabel(r"Re[$\alpha$]")
            ax.set_ylabel(r"Im[$\alpha$]")

    fig = ax.get_figure()
    fig.tight_layout()
    return axs, im


plot_wigner = lambda state, pts, ax=None, contour=True, **kwargs: plot_qp(
    state,
    pts,
    axs=ax,
    contour=contour,
    qp_type=WIGNER,
    cbar_label=r"$\mathcal{W}(\alpha)$",
    **kwargs,
)

plot_qfunc = lambda state, pts, ax=None, contour=True, **kwargs: plot_qp(
    state,
    pts,
    axs=ax,
    contour=contour,
    qp_type=QFUNC,
    cbar_label=r"$\mathcal{Q}(\alpha)$",
    **kwargs,
)

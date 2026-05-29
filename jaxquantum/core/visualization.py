"""
Visualization utils.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from jaxquantum.core.qp_distributions import wigner, qfunc
from jaxquantum.core.cfunctions import cf_wigner
import jax.numpy as jnp
import numpy as np

WIGNER = "wigner"
HUSIMI = "husimi"


def _render_qp_grid(
    axs,
    QP,
    pts_x,
    pts_y,
    *,
    contour,
    cmap,
    vmin,
    vmax,
    x_ticks,
    y_ticks,
    z_ticks,
    cbar_label,
    plot_cbar,
    subtitles,
    decorate=True,
):
    """Draw the per-frame quasi-probability grid onto a 2D array of axes.

    When ``decorate`` is False the cosmetic setup (ticks, axis lines, grid,
    labels, colorbar) is skipped — used for subsequent gif frames after the
    first, where these have already been laid out.
    """
    rows, cols = QP.shape[0], QP.shape[1]
    im = None
    for row in range(rows):
        for col in range(cols):
            ax = axs[row, col]
            if contour:
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
            if decorate:
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
            if subtitles is not None:
                ax.set_title(subtitles[row, col])
    return im


def plot_qp(
    state,
    pts_x,
    pts_y=None,
    g=2,
    axs=None,
    contour=True,
    qp_type=WIGNER,
    cbar_label="",
    axis_scale_factor=1,
    plot_cbar=True,
    x_ticks=None,
    y_ticks=None,
    z_ticks=None,
    subtitles=None,
    figtitle=None,
    gif=False,
    gif_params=None,
):
    """Plot quasi-probability distribution.


    Args:
        state: state with arbitrary number of batch dimensions, result will
        be flattened to a 2d grid to allow for plotting
        pts_x: x points to evaluate quasi-probability distribution at
        pts_y: y points to evaluate quasi-probability distribution at
        g : float, default: 2
        Scaling factor for ``a = 0.5 * g * (x + iy)``.  The value of `g` is
        related to the value of :math:`\\hbar` in the commutation relation
        :math:`[x,\,y] = i\\hbar` via :math:`\\hbar=2/g^2`.
        axs: matplotlib axes to plot on
        contour: make the plot use contouring
        qp_type: type of quasi probability distribution ("wigner", "qfunc")
        cbar_label: label for the cbar
        axis_scale_factor: scale of the axes labels relative
        plot_cbar: whether to plot cbar
        x_ticks: tick position for the x-axis
        y_ticks: tick position for the y-axis
        z_ticks: tick position for the z-axis
        subtitles: subtitles for the subplots
        figtitle: figure title
        gif: if True, render an animation over one batch axis instead of a
            tiled subplot grid. Returns a ``matplotlib.animation.FuncAnimation``.
        gif_params: dict of options for the gif path (ignored if gif=False):
            ``save_path`` (default None) — if set, save the animation here via
            PillowWriter; ``interval_ms`` (default 200) — milliseconds per
            frame; ``ts`` (default None) — optional 1D array of timestamps
            matching the animation axis length; when set, each frame's
            suptitle gets a ``t = ...`` label; ``batch_animation_axis``
            (default 0) — index into ``state.bdims`` selecting which axis
            becomes the animation/time axis (remaining batch dims form the
            per-frame subplot grid).

    Returns:
        ``(axs, im)`` in the static case, or a ``FuncAnimation`` when
        ``gif=True``.
    """
    if pts_y is None:
        pts_y = pts_x
    pts_x = jnp.array(pts_x)
    pts_y = jnp.array(pts_y)

    if len(state.bdims)==1 and state.bdims[0]==1:
        state = state[0]

    if gif:
        return _plot_qp_gif(
            state=state,
            pts_x=pts_x,
            pts_y=pts_y,
            g=g,
            axs=axs,
            contour=contour,
            qp_type=qp_type,
            axis_scale_factor=axis_scale_factor,
            plot_cbar=plot_cbar,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            z_ticks=z_ticks,
            subtitles=subtitles,
            figtitle=figtitle,
            gif_params=gif_params or {},
        )

    bdims = state.bdims
    added_baxes = 0

    if subtitles is not None:
        if subtitles.shape != bdims:
            raise ValueError(
                f"labels must have same shape as bdims, "
                f"got shapes {subtitles.shape} and {bdims}"
            )

    if len(bdims) == 0:
        bdims = (1,)
        added_baxes += 1
    if len(bdims) == 1:
        bdims = (1, bdims[0])
        added_baxes += 1

    extra_dims = bdims[2:]
    if extra_dims != ():
        state = state.reshape_bdims(
            bdims[0] * int(jnp.prod(jnp.array(extra_dims))), bdims[1]
        )
        if subtitles is not None:
            subtitles = subtitles.reshape(
                bdims[0] * int(jnp.prod(jnp.array(extra_dims))), bdims[1]
            )
        bdims = state.bdims

    if axs is None:
        _, axs = plt.subplots(
            bdims[0],
            bdims[1],
            figsize=(4 * bdims[1], 3 * bdims[0]),
            dpi=200,
        )

    if qp_type == WIGNER:
        vmin = -1
        vmax = 1
        scale = np.pi / 2
        cmap = "seismic"
        cbar_label = r"$\mathcal{W}(\alpha)$"
        QP = scale * wigner(state, pts_x, pts_y, g=g)

    elif qp_type == HUSIMI:
        vmin = 0
        vmax = 1
        scale = np.pi
        cmap = "jet"
        cbar_label = r"$\mathcal{Q}(\alpha)$"
        QP = scale * qfunc(state, pts_x, pts_y, g=g)



    for _ in range(added_baxes):
        QP = jnp.array([QP])
        axs = np.array([axs])
        if subtitles is not None:
            subtitles = np.array([subtitles])




    pts_x = pts_x * axis_scale_factor
    pts_y = pts_y * axis_scale_factor

    x_ticks = (
        jnp.linspace(jnp.min(pts_x), jnp.max(pts_x), 5) if x_ticks is None else x_ticks
    )
    y_ticks = (
        jnp.linspace(jnp.min(pts_y), jnp.max(pts_y), 5) if y_ticks is None else y_ticks
    )
    z_ticks = jnp.linspace(vmin, vmax, 3) if z_ticks is None else z_ticks

    im = _render_qp_grid(
        axs,
        QP,
        pts_x,
        pts_y,
        contour=contour,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        z_ticks=z_ticks,
        cbar_label=cbar_label,
        plot_cbar=plot_cbar,
        subtitles=subtitles,
        decorate=True,
    )

    fig = axs[bdims[0] - 1, bdims[1] - 1].get_figure()
    fig.tight_layout()
    if figtitle is not None:
        fig.suptitle(figtitle, y=1.04)
    return axs, im


def _plot_qp_gif(
    state,
    pts_x,
    pts_y,
    *,
    g,
    axs,
    contour,
    qp_type,
    axis_scale_factor,
    plot_cbar,
    x_ticks,
    y_ticks,
    z_ticks,
    subtitles,
    figtitle,
    gif_params,
):
    """Animation path for ``plot_qp(gif=True)``."""
    save_path = gif_params.get("save_path", None)
    interval_ms = gif_params.get("interval_ms", 200)
    ts = gif_params.get("ts", None)
    batch_animation_axis = gif_params.get("batch_animation_axis", 0)

    bdims = state.bdims
    if len(bdims) < 1:
        raise ValueError(
            "gif=True requires the state to have at least one batch dimension"
        )
    if not 0 <= batch_animation_axis < len(bdims):
        raise ValueError(
            f"batch_animation_axis={batch_animation_axis} is out of range "
            f"for state.bdims={bdims}"
        )
    N = bdims[batch_animation_axis]
    if ts is not None and len(ts) != N:
        raise ValueError(
            f"ts has length {len(ts)} but animation axis has length {N}"
        )

    if qp_type == WIGNER:
        vmin, vmax, scale = -1, 1, np.pi / 2
        cmap = "seismic"
        cbar_label = r"$\mathcal{W}(\alpha)$"
        QP = scale * wigner(state, pts_x, pts_y, g=g)
    elif qp_type == HUSIMI:
        vmin, vmax, scale = 0, 1, np.pi
        cmap = "jet"
        cbar_label = r"$\mathcal{Q}(\alpha)$"
        QP = scale * qfunc(state, pts_x, pts_y, g=g)

    QP = jnp.moveaxis(QP, batch_animation_axis, 0)
    rest_bdims = tuple(d for i, d in enumerate(bdims) if i != batch_animation_axis)

    grid_dims = list(rest_bdims)
    added_baxes = 0
    if len(grid_dims) == 0:
        grid_dims = [1]
        added_baxes += 1
    if len(grid_dims) == 1:
        grid_dims = [1, grid_dims[0]]
        added_baxes += 1
    extras = grid_dims[2:]
    rows = grid_dims[0] * int(np.prod(extras)) if extras else grid_dims[0]
    cols = grid_dims[1]

    h, w = QP.shape[-2], QP.shape[-1]
    QP_anim = QP.reshape((N, rows, cols, h, w))

    if subtitles is not None:
        subtitles = np.asarray(subtitles)
        if subtitles.shape != rest_bdims:
            raise ValueError(
                f"subtitles shape {subtitles.shape} must match per-frame "
                f"batch dims {rest_bdims} (state.bdims minus the animation axis)"
            )
        subtitles = subtitles.reshape(rows, cols)

    pts_x_scaled = pts_x * axis_scale_factor
    pts_y_scaled = pts_y * axis_scale_factor
    x_ticks = (
        jnp.linspace(jnp.min(pts_x_scaled), jnp.max(pts_x_scaled), 5)
        if x_ticks is None
        else x_ticks
    )
    y_ticks = (
        jnp.linspace(jnp.min(pts_y_scaled), jnp.max(pts_y_scaled), 5)
        if y_ticks is None
        else y_ticks
    )
    z_ticks = jnp.linspace(vmin, vmax, 3) if z_ticks is None else z_ticks

    if axs is None:
        _, axs = plt.subplots(
            rows, cols, figsize=(4 * cols, 3 * rows), dpi=200
        )
    axs_arr = axs
    for _ in range(added_baxes):
        axs_arr = np.array([axs_arr])
    axs_arr = np.asarray(axs_arr).reshape(rows, cols)
    fig = axs_arr[0, 0].get_figure()

    has_suptitle = figtitle is not None or ts is not None

    def _set_suptitle(k):
        if ts is not None:
            t_str = f"t = {float(ts[k]):.3g}"
            title = f"{figtitle} | {t_str}" if figtitle else t_str
            fig.suptitle(title, y=0.98)
        elif figtitle is not None:
            fig.suptitle(figtitle, y=0.98)

    _render_qp_grid(
        axs_arr,
        QP_anim[0],
        pts_x_scaled,
        pts_y_scaled,
        contour=contour,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        z_ticks=z_ticks,
        cbar_label=cbar_label,
        plot_cbar=plot_cbar,
        subtitles=subtitles,
        decorate=True,
    )
    _set_suptitle(0)
    if has_suptitle:
        # Reserve top strip of the figure so the suptitle isn't clipped in the
        # rendered animation (PillowWriter uses the figure bbox as-is).
        fig.tight_layout(rect=[0, 0, 1, 0.92])
    else:
        fig.tight_layout()

    def update(k):
        for r in range(rows):
            for c in range(cols):
                for coll in list(axs_arr[r, c].collections):
                    coll.remove()
        _render_qp_grid(
            axs_arr,
            QP_anim[k],
            pts_x_scaled,
            pts_y_scaled,
            contour=contour,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            z_ticks=z_ticks,
            cbar_label=cbar_label,
            plot_cbar=plot_cbar,
            subtitles=subtitles,
            decorate=False,
        )
        _set_suptitle(k)
        return []

    anim = FuncAnimation(fig, update, frames=N, interval=interval_ms, blit=False)
    if save_path is not None:
        fps = max(1, round(1000 / interval_ms))
        anim.save(save_path, writer=PillowWriter(fps=fps))

    # Make Jupyter render the animation inline without needing an explicit
    # HTML(anim.to_jshtml()) wrapper, and suppress the static last-frame
    # figure that the inline backend would otherwise emit alongside it.
    anim._repr_html_ = lambda a=anim: a.to_jshtml()
    plt.close(fig)
    return anim


def plot_wigner(
    state,
    pts_x,
    pts_y=None,
    g=2,
    axs=None,
    contour=True,
    cbar_label="",
    axis_scale_factor=1,
    plot_cbar=True,
    x_ticks=None,
    y_ticks=None,
    z_ticks=None,
    subtitles=None,
    figtitle=None,
    gif=False,
    gif_params=None,
):
    """Plot the wigner function of the state.


    Args:
        state: state with arbitrary number of batch dimensions, result will
        be flattened to a 2d grid to allow for plotting
        pts_x: x points to evaluate quasi-probability distribution at
        pts_y: y points to evaluate quasi-probability distribution at
        g : float, default: 2
        Scaling factor for ``a = 0.5 * g * (x + iy)``.  The value of `g` is
        related to the value of :math:`\\hbar` in the commutation relation
        :math:`[x,\,y] = i\\hbar` via :math:`\\hbar=2/g^2`.
        axs: matplotlib axes to plot on
        contour: make the plot use contouring
        cbar_label: label for the cbar
        axis_scale_factor: scale of the axes labels relative
        plot_cbar: whether to plot cbar
        x_ticks: tick position for the x-axis
        y_ticks: tick position for the y-axis
        z_ticks: tick position for the z-axis
        subtitles: subtitles for the subplots
        figtitle: figure title

    Returns:
        axis on which the plot was plotted.
    """
    return plot_qp(
        state=state,
        pts_x=pts_x,
        pts_y=pts_y,
        g=g,
        axs=axs,
        contour=contour,
        qp_type=WIGNER,
        cbar_label=cbar_label,
        axis_scale_factor=axis_scale_factor,
        plot_cbar=plot_cbar,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        z_ticks=z_ticks,
        subtitles=subtitles,
        figtitle=figtitle,
        gif=gif,
        gif_params=gif_params,
    )


def plot_qfunc(
    state,
    pts_x,
    pts_y=None,
    g=2,
    axs=None,
    contour=True,
    cbar_label="",
    axis_scale_factor=1,
    plot_cbar=True,
    x_ticks=None,
    y_ticks=None,
    z_ticks=None,
    subtitles=None,
    figtitle=None,
    gif=False,
    gif_params=None,
):
    """Plot the husimi function of the state.


    Args:
        state: state with arbitrary number of batch dimensions, result will
        be flattened to a 2d grid to allow for plotting
        pts_x: x points to evaluate quasi-probability distribution at
        pts_y: y points to evaluate quasi-probability distribution at
        g : float, default: 2
        Scaling factor for ``a = 0.5 * g * (x + iy)``.  The value of `g` is
        related to the value of :math:`\\hbar` in the commutation relation
        :math:`[x,\,y] = i\\hbar` via :math:`\\hbar=2/g^2`.
        axs: matplotlib axes to plot on
        contour: make the plot use contouring
        cbar_label: label for the cbar
        axis_scale_factor: scale of the axes labels relative
        plot_cbar: whether to plot cbar
        x_ticks: tick position for the x-axis
        y_ticks: tick position for the y-axis
        z_ticks: tick position for the z-axis
        subtitles: subtitles for the subplots
        figtitle: figure title

    Returns:
        axis on which the plot was plotted.
    """
    return plot_qp(
        state=state,
        pts_x=pts_x,
        pts_y=pts_y,
        g=g,
        axs=axs,
        contour=contour,
        qp_type=HUSIMI,
        cbar_label=cbar_label,
        axis_scale_factor=axis_scale_factor,
        plot_cbar=plot_cbar,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        z_ticks=z_ticks,
        subtitles=subtitles,
        figtitle=figtitle,
        gif=gif,
        gif_params=gif_params,
    )


def plot_cf(
        state,
        pts_x,
        pts_y=None,
        axs=None,
        contour=True,
        qp_type=WIGNER,
        cbar_label="",
        axis_scale_factor=1,
        plot_cbar=True,
        plot_grid=True,
        x_ticks=None,
        y_ticks=None,
        z_ticks=None,
        subtitles=None,
        figtitle=None,
):
    """Plot characteristic function.


    Args:
        state: state with arbitrary number of batch dimensions, result will
        be flattened to a 2d grid to allow for plotting
        pts_x: x points to evaluate quasi-probability distribution at
        pts_y: y points to evaluate quasi-probability distribution at
        axs: matplotlib axes to plot on
        contour: make the plot use contouring
        qp_type: type of quasi probability distribution ("wigner")
        cbar_label: labels for the real and imaginary cbar
        axis_scale_factor: scale of the axes labels relative
        plot_cbar: whether to plot cbar
        x_ticks: tick position for the x-axis
        y_ticks: tick position for the y-axis
        z_ticks: tick position for the z-axis
        subtitles: subtitles for the subplots
        figtitle: figure title

    Returns:
        axis on which the plot was plotted.
    """
    if pts_y is None:
        pts_y = pts_x
    pts_x = jnp.array(pts_x)
    pts_y = jnp.array(pts_y)

    bdims = state.bdims
    added_baxes = 0

    if subtitles is not None:
        if subtitles.shape != bdims:
            raise ValueError(
                f"labels must have same shape as bdims, "
                f"got shapes {subtitles.shape} and {bdims}"
            )

    if len(bdims) == 0:
        bdims = (1,)
        added_baxes += 1
    if len(bdims) == 1:
        bdims = (1, bdims[0])
        added_baxes += 1

    extra_dims = bdims[2:]
    if extra_dims != ():
        state = state.reshape_bdims(
            bdims[0] * int(jnp.prod(jnp.array(extra_dims))), bdims[1]
        )
        if subtitles is not None:
            subtitles = subtitles.reshape(
                bdims[0] * int(jnp.prod(jnp.array(extra_dims))), bdims[1]
            )
        bdims = state.bdims

    if axs is None:
        _, axs = plt.subplots(
            bdims[0],
            bdims[1]*2,
            figsize=(4 * bdims[1]*2, 3 * bdims[0]),
            dpi=200,
        )


    if qp_type == WIGNER:
        vmin = -1
        vmax = 1
        scale = 1
        cmap = "seismic"
        cbar_label = [r"$\mathcal{Re}(\chi_W(\alpha))$", r"$\mathcal{"
                                                         r"Im}(\chi_W("
                                                         r"\alpha))$"]
        QP = scale * cf_wigner(state, pts_x, pts_y)

    for _ in range(added_baxes):
        QP = jnp.array([QP])
        axs = np.array([axs])
        if subtitles is not None:
            subtitles = np.array([subtitles])

    if added_baxes==2:
        axs = axs[0] # When the input state is zero-dimensional, remove an
                     # axis that is automatically added due to the subcolumns


    pts_x = pts_x * axis_scale_factor
    pts_y = pts_y * axis_scale_factor

    x_ticks = (
        jnp.linspace(jnp.min(pts_x), jnp.max(pts_x),
                     5) if x_ticks is None else x_ticks
    )
    y_ticks = (
        jnp.linspace(jnp.min(pts_y), jnp.max(pts_y),
                     5) if y_ticks is None else y_ticks
    )
    z_ticks = jnp.linspace(vmin, vmax, 11) if z_ticks is None else z_ticks
    print(axs.shape)
    for row in range(bdims[0]):
        for col in range(bdims[1]):
            for subcol in range(2):
                ax = axs[row, 2 * col + subcol]
                if contour:
                    im = ax.contourf(
                        pts_x,
                        pts_y,
                        jnp.real(QP[row, col]) if subcol==0 else jnp.imag(QP[
                                                                           row, col]),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        levels=np.linspace(vmin, vmax, 101),
                    )
                else:
                    im = ax.pcolormesh(
                        pts_x,
                        pts_y,
                        jnp.real(QP[row, col]) if subcol == 0 else jnp.imag(QP[
                                                                                row, col]),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                # ax.axhline(0, linestyle="-", color="black", alpha=0.7)
                # ax.axvline(0, linestyle="-", color="black", alpha=0.7)

                if plot_grid:
                    ax.grid()
                
                ax.set_aspect("equal", adjustable="box")

                if plot_cbar:
                    cbar = plt.colorbar(
                        im, ax=ax, orientation="vertical",
                        ticks=np.linspace(-1, 1, 11)
                    )
                    cbar.ax.set_title(cbar_label[subcol])
                    cbar.set_ticks(z_ticks)

                ax.set_xlabel(r"Re[$\alpha$]")
                ax.set_ylabel(r"Im[$\alpha$]")
                if subtitles is not None:
                    ax.set_title(subtitles[row, col])

    fig = ax.get_figure()
    fig.tight_layout()
    if figtitle is not None:
        fig.suptitle(figtitle, y=1.04)
    return axs, im

def plot_cf_wigner(
    state,
    pts_x,
    pts_y=None,
    axs=None,
    contour=True,
    cbar_label="",
    axis_scale_factor=1,
    plot_cbar=True,
    plot_grid=True,
    x_ticks=None,
    y_ticks=None,
    z_ticks=None,
    subtitles=None,
    figtitle=None,
):
    """Plot the Wigner characteristic function of the state.


    Args:
        state: state with arbitrary number of batch dimensions, result will
        be flattened to a 2d grid to allow for plotting
        pts_x: x points to evaluate quasi-probability distribution at
        pts_y: y points to evaluate quasi-probability distribution at
        axs: matplotlib axes to plot on
        contour: make the plot use contouring
        cbar_label: label for the cbar
        axis_scale_factor: scale of the axes labels relative
        plot_cbar: whether to plot cbar
        x_ticks: tick position for the x-axis
        y_ticks: tick position for the y-axis
        z_ticks: tick position for the z-axis
        subtitles: subtitles for the subplots
        figtitle: figure title

    Returns:
        axis on which the plot was plotted.
    """
    return plot_cf(
        state=state,
        pts_x=pts_x,
        pts_y=pts_y,
        axs=axs,
        contour=contour,
        qp_type=WIGNER,
        cbar_label=cbar_label,
        axis_scale_factor=axis_scale_factor,
        plot_cbar=plot_cbar,
        plot_grid=plot_grid,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        z_ticks=z_ticks,
        subtitles=subtitles,
        figtitle=figtitle,
    )

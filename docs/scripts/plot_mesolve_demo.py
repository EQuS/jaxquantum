"""Generate the master-equation demo plot for the landing page.

Outputs: docs/assets/readme_demo.png (transparent PNG).

Run from the project root:
    python docs/scripts/plot_mesolve_demo.py
"""

from pathlib import Path
import sys

import jax.numpy as jnp
from jax import jit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jaxquantum as jqt

sys.path.insert(0, str(Path(__file__).parent))
from _plot_style import apply_style, PURPLE, TEAL  # noqa: E402

OUT = Path(__file__).parent.parent / "assets" / "readme_demo.png"


def main():
    apply_style()

    # --- Simulation ---
    N = 100
    omega = 2 * jnp.pi * 5.0
    kappa = 2 * jnp.pi * jnp.array([1.0, 2.0])

    rho0 = (jqt.displace(N, 0.1) @ jqt.basis(N, 0)).to_dm()
    ts = jnp.linspace(0, 4 * 2 * jnp.pi / omega, 101)
    c_ops = jqt.Qarray.from_list([jnp.sqrt(kappa) * jqt.destroy(N)])

    @jit
    def H(t):
        return omega * jqt.num(N)

    states = jqt.mesolve(
        H, rho0, ts, c_ops=c_ops,
        solver_options=jqt.SolverOptions.create(progress_meter=False),
    )
    n_t = jnp.real(jqt.overlap(jqt.num(N), states))      # (101, 2)
    a_t = jqt.overlap(jqt.destroy(N), states)             # (101, 2) complex

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 4.5), sharex=True,
        gridspec_kw={"hspace": 0.3},
    )

    ax1.plot(ts, n_t[:, 0], color=PURPLE, lw=2.2, label=r"$\kappa = 2\pi$")
    ax1.plot(ts, n_t[:, 1], color=TEAL,   lw=2.2, label=r"$\kappa = 4\pi$")
    ax1.set_ylabel(r"$\langle\hat n(t)\rangle$")
    ax1.legend(loc="upper right")
    ax1.set_title("Decaying photon number — JAX-batched mesolve")

    ax2.plot(ts, jnp.real(a_t[:, 0]), color=PURPLE, lw=2.2, label=r"$\kappa = 2\pi$")
    ax2.plot(ts, jnp.real(a_t[:, 1]), color=TEAL,   lw=2.2, label=r"$\kappa = 4\pi$")
    ax2.set_ylabel(r"$\mathrm{Re}\,\langle\hat a(t)\rangle$")
    ax2.set_xlabel("Time")

    fig.savefig(OUT, transparent=True)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()

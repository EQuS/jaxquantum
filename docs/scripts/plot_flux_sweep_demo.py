"""Generate the tunable-transmon flux-sweep demo plot for the landing page.

Outputs: docs/assets/transmon_flux_sweep.png (transparent PNG).

Run from the project root:
    python docs/scripts/plot_flux_sweep_demo.py
"""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jaxquantum.devices as jqtd

sys.path.insert(0, str(Path(__file__).parent))
from _plot_style import apply_style, PURPLE, TEAL, ORANGE  # noqa: E402

OUT = Path(__file__).parent.parent / "assets" / "transmon_flux_sweep.png"


def main():
    apply_style()

    @jax.jit
    def get_freqs(phi_ext):
        t = jqtd.TunableTransmon.create(
            N=4,
            params={"Ec": 0.3, "Ej1": 8.0, "Ej2": 7.0, "phi_ext": phi_ext},
            N_pre_diag=41, basis=jqtd.BasisTypes.charge,
        )
        Es = t.eig_systems["vals"]
        return Es - Es[0]

    phi_vals = jnp.linspace(0.0, 1.0, 200)
    all_freqs = jax.vmap(get_freqs)(phi_vals)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = [PURPLE, TEAL, ORANGE]
    for level, c in zip(range(1, 4), colors):
        ax.plot(phi_vals, all_freqs[:, level], color=c, lw=2.2, label=f"$f_{{0{level}}}$")
    ax.set_xlabel(r"$\Phi_\mathrm{ext}/\Phi_0$")
    ax.set_ylabel("Frequency (GHz)")
    ax.set_title("Tunable transmon spectrum — 200 points via vmap")
    ax.legend(loc="upper right")

    fig.savefig(OUT, transparent=True)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()

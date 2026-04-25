<h1 align="center">
    <img src="https://github.com/EQuS/jaxquantum/raw/main/docs/assets/logo.png" height="120" alt="jaxquantum logo">
</h1>


[![License](https://img.shields.io/github/license/EQuS/jaxquantum.svg)](https://opensource.org/license/apache-2-0) [![](https://img.shields.io/github/release/EQuS/jaxquantum.svg)](https://github.com/EQuS/jaxquantum/releases) [![](https://img.shields.io/pypi/dm/jaxquantum.svg)](https://pypi.org/project/jaxquantum/)

[![code coverage](https://jaxquantum.org/test-results/coverage.svg?raw=true)](https://jaxquantum.org/test-results/cov_html/) [![tests](https://github.com/EQuS/jaxquantum/actions/workflows/pytest.yml/badge.svg)](https://github.com/EQuS/jaxquantum/actions/workflows/pytest.yml) [![ruff](https://github.com/EQuS/jaxquantum/actions/workflows/ruff.yml/badge.svg)](https://github.com/EQuS/jaxquantum/actions/workflows/ruff.yml) [![docs](https://github.com/EQuS/jaxquantum/actions/workflows/docs.yml/badge.svg)](https://github.com/EQuS/jaxquantum/actions/workflows/docs.yml)

[S. R. Jha](https://github.com/Phionx), [S. Chowdhury](https://github.com/shoumikdc), [G. Rolleri](https://github.com/GabrieleRolleri), [M. Hays](https://scholar.google.com/citations?user=06z0MjwAAAAJ), [J. A. Grover](https://scholar.google.com/citations?user=igewch8AAAAJ), [W. D. Oliver](https://scholar.google.com/citations?user=4vNbnqcAAAAJ&hl=en)

**Docs:** [jaxquantum.org](https://jaxquantum.org) &nbsp;|&nbsp; **Discord:** [discord.gg/frWqbjvZ4s](https://discord.gg/frWqbjvZ4s)

`jaxquantum` is a unified [JAX](https://github.com/google/jax)-native toolkit for quantum hardware design, simulation, and control — auto-differentiable and accelerated on CPU, GPU, and TPU. It serves as a QuTiP drop-in replacement and absorbs the prior [`bosonic`](https://github.com/EQuS/bosonic) and [`qcsys`](https://github.com/EQuS/qcsys) projects.


## Highlights

- **Superconducting devices** — ready-to-use Transmon, Fluxonium, and Resonator models with eigenspectrum, wavefunctions, and parameter sweeps. See the [devices tutorial](https://jaxquantum.org/documentation/tutorials/devices.html).
- **Bosonic codes** — Cat, GKP, and Binomial qubit encodings with logical gates and phase-space visualization. See the [bosonic codes tutorial](https://jaxquantum.org/documentation/tutorials/bosonic_codes.html).
- **Gate-based circuits** — hierarchical circuits with unitary, Hamiltonian, and Kraus simulation modes; gradient-based gate optimization. See the [circuits tutorial](https://jaxquantum.org/documentation/tutorials/circuits.html).
- **Sparse backends** — `SparseDIA` and `BCOO` storage for large Hilbert spaces with the same API as dense. See the [sparse backends tutorial](https://jaxquantum.org/documentation/tutorials/sparse_backends.html).
- **First-class JAX** — use `jax.vmap` for parameter sweeps, `jax.jit` for compiled simulation, and `jax.grad` for differentiable physics out of the box.


## Installation

```bash
pip install jaxquantum
```

For GPU (NVIDIA, CUDA13) or TPU, use the `[gpu]` or `[tpu]` extras. For the latest development version, install directly from source:

```bash
pip install git+https://github.com/EQuS/jaxquantum.git
```

For development (editable + dev/docs extras): `pip install -e ".[dev,docs]"`. See the [installation guide](https://jaxquantum.org/documentation/getting_started/installation.html) for full details, hardware checks, and troubleshooting.


## Quick Start

```python
from jax import jit
import jaxquantum as jqt
import jax.numpy as jnp
import matplotlib.pyplot as plt

N = 100

omega_a = 2.0*jnp.pi*5.0
kappa = 2*jnp.pi*jnp.array([1,2]) # Batching to explore two different kappa values!
initial_state = jqt.displace(N, 0.1) @ jqt.basis(N,0)
initial_state_dm = initial_state.to_dm()
ts = jnp.linspace(0, 4*2*jnp.pi/omega_a, 101)

a = jqt.destroy(N)
n = a.dag() @ a

c_ops = jqt.Qarray.from_list([jnp.sqrt(kappa)*a])

@jit
def Ht(t):
    H0 = omega_a*n
    return H0

solver_options = jqt.SolverOptions.create(progress_meter=True)
states = jqt.mesolve(Ht, initial_state_dm, ts, c_ops=c_ops, solver_options=solver_options)
nt = jnp.real(jqt.overlap(n, states))
a_real = jnp.real(jqt.overlap(a, states))
a_imag = jnp.imag(jqt.overlap(a, states))

fig, axs = plt.subplots(2,1, dpi=200, figsize=(6,5))
ax = axs[0]
ax.plot(ts, a_real[:,0], label=r"$Re[\langle a(t)\rangle]$", color="blue") # Batch kappa value 0
ax.plot(ts, a_real[:,1], "--", label=r"$Re[\langle a(t)\rangle]$", color="blue") # Batch kappa value 1
ax.plot(ts, a_imag[:,0], label=r"$Re[\langle a(t)\rangle]$", color="red") # Batch kappa value 0
ax.plot(ts, a_imag[:,1], "--", label=r"$Re[\langle a(t)\rangle]$", color="red") # Batch kappa value 1
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Expectations")
ax.legend()

ax = axs[1]
ax.plot(ts, nt[:,0], label=r"$Re[\langle n(t)\rangle]$", color="green") # Batch kappa value 0
ax.plot(ts, nt[:,1], "--", label=r"$Re[\langle n(t)\rangle]$", color="green") # Batch kappa value 1
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Expectations")
ax.legend()
fig.tight_layout()
```
![Output of above code.](https://github.com/EQuS/jaxquantum/raw/main/docs/assets/readme_demo.png)


## Acknowledgements & History

**Core Devs:** [Shantanu R. Jha](https://github.com/Phionx), [Shoumik Chowdhury](https://github.com/shoumikdc), [Gabriele Rolleri](https://github.com/GabrieleRolleri)


This package was initially a small part of [`bosonic`](https://github.com/EQuS/bosonic). In early 2022, `jaxquantum` was extracted and made into its own package. This package was briefly announced to the world at APS March Meeting 2023 and released to a select few academic groups shortly after. Since then, this package has been open sourced and developed while conducting research in the Engineering Quantum Systems Group at MIT with advice and support from [Prof. William D. Oliver](https://equs.mit.edu/william-d-oliver/).

## Citation

Thank you for taking the time to try our package out. If you found it useful in your research, please cite us as follows:

```bibtex
@software{jha2024jaxquantum,
  author  = {Shantanu R. Jha and Shoumik Chowdhury and Gabriele Rolleri and Max Hays and Jeff A. Grover and William D. Oliver},
  title   = {JAXQuantum: An auto-differentiable and hardware-accelerated toolkit for quantum hardware design, simulation, and control},
  url     = {https://jaxquantum.org},
  version = {0.3.0},
  year    = {2024},
}
```
> S. R. Jha, S. Chowdhury, G. Rolleri, M. Hays, J. A. Grover, and W. D. Oliver. *"JAXQuantum: An auto-differentiable and hardware-accelerated toolkit for quantum hardware design, simulation, and control,"* jaxquantum.org (2025).


## Contributions & Contact

This package is open source and, as such, very open to contributions. Please don't hesitate to open an issue, report a bug, request a feature, or create a pull request. We are also open to deeper collaborations to create a tool that is more useful for everyone. If a discussion would be helpful, please email [shanjha@mit.edu](mailto:shanjha@mit.edu) to set up a meeting.

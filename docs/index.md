# Welcome to jaxquantum

[![License](https://img.shields.io/github/license/EQuS/jaxquantum.svg?style=popout-square)](https://opensource.org/license/apache-2-0) [![](https://img.shields.io/github/release/EQuS/jaxquantum.svg?style=popout-square)](https://github.com/EQuS/jaxquantum/releases) [![](https://img.shields.io/pypi/dm/jaxquantum.svg?style=popout-square)](https://pypi.org/project/jaxquantum/) 

[![code coverage](https://jaxquantum.org/test-results/coverage.svg?raw=true)](https://jaxquantum.org/test-results/cov_html/) [![tests](https://github.com/EQuS/jaxquantum/actions/workflows/pytest.yml/badge.svg)](https://github.com/EQuS/jaxquantum/actions/workflows/pytest.yml) [![ruff](https://github.com/EQuS/jaxquantum/actions/workflows/ruff.yml/badge.svg)](https://github.com/EQuS/jaxquantum/actions/workflows/ruff.yml) [![docs](https://github.com/EQuS/jaxquantum/actions/workflows/docs.yml/badge.svg)](https://github.com/EQuS/jaxquantum/actions/workflows/docs.yml) 

[S. R. Jha](https://github.com/Phionx), [S. Chowdhury](https://github.com/shoumikdc), [G. Rolleri](https://github.com/GabrieleRolleri), [M. Hays](https://scholar.google.com/citations?user=06z0MjwAAAAJ), [J. A. Grover](https://scholar.google.com/citations?user=igewch8AAAAJ), [W. D. Oliver](https://scholar.google.com/citations?user=4vNbnqcAAAAJ&hl=en)

**Docs:** [equs.github.io/jaxquantum](https://equs.github.io/jaxquantum)

**Community Discord:** [discord.gg/frWqbjvZ4s](https://discord.gg/frWqbjvZ4s)

`jaxquantum` leverages [JAX](https://github.com/google/jax) to enable the auto differentiable and (CPU, GPU, TPU) accelerated simulation of quantum dynamical systems, including tooling such as operator construction, unitary evolution and master equation solving. As such, `jaxquantum` serves as a QuTiP drop-in replacement written entirely in JAX.

Moreover, `jaxquantum` has recently absorbed [`bosonic`](https://github.com/EQuS/bosonic) and [`qcsys`](https://github.com/EQuS/qcsys). As such, it is now a unified toolkit for quantum circuit design, simulation and control. 


## Installation


### Installing from source (recommended)

**Recommended:** As this is a rapidly evolving project, we recommend installing the latest version of `jaxquantum` from source as follows:
```bash
pip install git+https://github.com/EQuS/jaxquantum.git
```

If you are installing on a GPU (NVIDIA, CUDA12), then run this instead:
```bash
pip install 'git+https://github.com/EQuS/jaxquantum.git#egg=jaxquantum[gpu]'
```

And, on a TPU, run this:
```bash
pip install 'git+https://github.com/EQuS/jaxquantum.git#egg=jaxquantum[tpu]'
```

If you face issues running JAX on your hardware, visit this page: [https://docs.jax.dev/en/latest/installation.html](https://docs.jax.dev/en/latest/installation.html)

### Installing from source in editable mode (recommended for developers)

If you are interested in contributing to the package, please clone this repository and install this package in editable mode after changing into the root directory of this repository:
```bash
pip install -e ".[dev,docs]"
```
This will also install extras from the `dev` and `docs` flags, which can be useful when developing the package. Since this is installed in editable mode, the package will automatically be updated after pulling new changes in the repository. Again, add the `gpu` or `tpu` extra, if needed.

### Installing from PyPI (not recommended)

`jaxquantum` is also published on PyPI. Simply run the following code to install the package:

```bash
pip install jaxquantum
```

If you are installing on a GPU (NVIDIA, CUDA12), then run this instead:
```bash
pip install 'jaxquantum[gpu]'
```

And, on a TPU, run this:
```bash
pip install 'jaxquantum[tpu]'
```

If you face issues running JAX on your hardware, visit this page: [https://docs.jax.dev/en/latest/installation.html](https://docs.jax.dev/en/latest/installation.html)


For more details, please visit the getting started > installation section of our [docs](https://equs.github.io/jaxquantum/getting_started/installation.html).

### Check Hardware

To check which hardware JAX is running on, run the following python code:
```python
import jax.numpy as jnp
x = jnp.array([1.0, 2.0, 3.0])
print(x.device)
```
This will, for example, print out `cuda:0` if running on a GPU.


## An Example

Here's an example of how to set up a simulation in jaxquantum.

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
![Output of above code.](assets/readme_demo.png)


## Acknowledgements & History

**Core Devs:** [Shantanu R. Jha](https://github.com/Phionx), [Shoumik Chowdhury](https://github.com/shoumikdc), [Gabriele Rolleri](https://github.com/GabrieleRolleri)


This package was initially a small part of [`bosonic`](https://github.com/EQuS/bosonic). In early 2022, `jaxquantum` was extracted and made into its own package. This package was briefly announced to the world at APS March Meeting 2023 and released to a select few academic groups shortly after. Since then, this package has been open sourced and developed while conducting research in the Engineering Quantum Systems Group at MIT with advice and support from [Prof. William D. Oliver](https://equs.mit.edu/william-d-oliver/). 






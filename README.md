<h1 align="center">
    <img src="./docs/assets/logo.png" height="120" alt="jaxquantum logo">
</h1>


[![License](https://img.shields.io/github/license/EQuS/jaxquantum.svg?style=popout-square)](https://opensource.org/license/apache-2-0) [![](https://img.shields.io/github/release/EQuS/jaxquantum.svg?style=popout-square)](https://github.com/EQuS/jaxquantum/releases) [![](https://img.shields.io/pypi/dm/jaxquantum.svg?style=popout-square)](https://pypi.org/project/jaxquantum/)

[S. R. Jha](https://github.com/Phionx), [S. Chowdhury](https://github.com/shoumikdc), [M. Hays](https://scholar.google.com/citations?user=06z0MjwAAAAJ), [J. A. Grover](https://scholar.google.com/citations?user=igewch8AAAAJ), [W. D. Oliver](https://scholar.google.com/citations?user=4vNbnqcAAAAJ&hl=en)

***Docs:** [github.com/pages/EQuS/jaxquantum](https://github.com/pages/EQuS/jaxquantum/)*

`jaxquantum` leverages [JAX](https://github.com/google/jax) to enable the auto differentiable and (CPU, GPU, TPU) accelerated simulation of quantum dynamical systems, including tooling such as operator construction, unitary evolution and master equation solving. As such, `jaxquantum` serves as a QuTiP drop-in replacement written entirely in JAX.

This package also serves as an essential dependency for [`bosonic-jax`](https://github.com/EQuS/bosonic-jax) and [`qcsys`](https://github.com/EQuS/qcsys). Together, these packages form an end-to-end toolkit for quantum circuit design, simulation and control. 


## Installation

`jaxquantum` is published on PyPI. So, to install the latest version from PyPI, simply run the following code to install the package:

```bash
pip install jaxquantum
```

For more details, please visit the getting started > installation section of our [docs](https://github.com/pages/EQuS/jaxquantum/).

## An Example

Here's an example of how to set up a simulation in jaxquantum.

```
from jax import jit
import jaxquantum as jqt
import jax.numpy as jnp
import matplotlib.pyplot as plt

omega_q = 5.0 #GHz
Omega = .1
g_state = jqt.basis(2,0) ^ jqt.basis(2,0)
g_state_dm = g_state.to_dm()

ts = jnp.linspace(0,5*jnp.pi/Omega,101)
c_ops = [0.1*jqt.sigmam()^jqt.identity(N=2)]

sz0 = jqt.sigmaz() ^ jqt.identity(N=2)

@jit
def Ht(t):
    H0 = omega_q/2.0*((jqt.sigmaz()^jqt.identity(N=2)) + (jqt.identity(N=2)^jqt.sigmaz()))
    H1 = Omega*jnp.cos((omega_q)*t)*((jqt.sigmax()^jqt.identity(N=2)) + (jqt.identity(N=2)^jqt.sigmax()))
    return H0 + H1


states = jqt.mesolve(g_state_dm, ts, c_ops=c_ops, Ht=Ht) 
szt = jnp.real(jqt.calc_expect(sz0, states))


fig, ax = plt.subplots(1, dpi=200, figsize=(4,3))
ax.plot(ts, szt)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("<σz(t)>")
fig.tight_layout()
```

## Acknowledgements & History

**Core Devs:** [Shantanu A. Jha](https://github.com/Phionx), [Shoumik Chowdhury](https://github.com/shoumikdc)


This package was initially a small part of [`bosonic-jax`](https://github.com/EQuS/bosonic-jax). In early 2022, `jaxquantum` was extracted and made into its own package. This package was briefly announced to the world at APS March Meeting 2023 and released to a select few academic groups shortly after. Since then, this package has been open sourced and developed while conducting research in the Engineering Quantum Systems Group at MIT with invaluable advice from [Prof. William D. Oliver](https://equs.mit.edu/william-d-oliver/). 

## Citation

Thank you for taking the time to try our package out. If you found it useful in your research, please cite us as follows:

``` 
@unpublished{jha2024jaxquantum,
  title  = {An auto differentiable and hardware accelerated software toolkit for quantum circuit design, simulation and control},
  author = {Shantanu R. Jha, Shoumik Chowdhury, Max Hays, Jeff A. Grover, William D. Oliver},
  year   = {2024},
  url    = {https://github.com/EQuS/jaxquantum}
}
```
> S. R. Jha, S. Chowdhury, M. Hays, J. A. Grover, W. D. Oliver. An auto differentiable and hardware accelerated software toolkit for quantum circuit design, simulation and control (2024), in preparation.


## Contributions & Contact

This package is open source and, as such, very open to contributions. Please don't hesitate to open an issue, report a bug, request a feature, or create a pull request. We are also open to deeper collaborations to create a tool that is more useful for everyone. If a discussion would be helpful, please email [shanjha@mit.edu](mailto:shanjha@mit.edu) to set up a meeting. 
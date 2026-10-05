# Custom direct channels

Direct channels update a local density matrix without constructing full Kraus
operators. They remain pure JAX functions, so they work with `jit`, `vmap`, and
automatic differentiation.

For an elementwise map, supply its matrix of factors:

```python
import jax.numpy as jnp
import jaxquantum as jqt
import jaxquantum.circuits as jqtc

eta = 0.9
channel = jqtc.ElementwiseChannel(
    2,
    jnp.array([[1.0, eta], [eta, 1.0]]),
    name="Dephasing",
)
rho_out = jqtc.apply_channel(channel, jqt.basis(2, 0).to_dm().data)
```

`ShiftedChannel` covers maps whose Kraus branches shift basis indices, such as
bosonic loss and gain. `Channel` accepts any pure kernel with signature
`apply(rho, params)`, where `rho` ends in `(dimension, dimension)`.

An optional `kraus` list or `kraus(params)` callable provides interoperability
with code that accesses `channel.KM`. It is generated lazily:

```python
def apply_phase_flip(rho, params):
    z = jnp.diag(jnp.array([1.0, -1.0]))
    p = params["p"]
    return (1 - p) * rho + p * z @ rho @ z

def phase_flip_kraus(params):
    p = params["p"]
    return jqt.Qarray.from_list([
        jnp.sqrt(1 - p) * jqt.identity(2),
        jnp.sqrt(p) * jqt.sigmaz(),
    ])

channel = jqtc.Channel(
    2,
    apply_phase_flip,
    params={"p": 0.1},
    kraus=phase_flip_kraus,
    name="PhaseFlip",
)
```

Inside a circuit, direct kernels act only on the target modes. Use
`apply_channel` for standalone density matrices and `apply_kraus_map` when a
dense Kraus stack is already available.

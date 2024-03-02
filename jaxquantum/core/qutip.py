"""
Common jax <-> qutip-inspired functions
"""

from jax import config

import jax.numpy as jnp
import numpy as np
from qutip import Qobj


config.update("jax_enable_x64", True)

# Convert between QuTiP and JAX
# ===============================================================
def qt2jax(qt_obj, dtype=jnp.complex128):
    """QuTiP state -> JAX array.

    Args:
        qt_obj: QuTiP state.
        dtype: JAX dtype.

    Returns:
        JAX array.
    """
    if isinstance(qt_obj, jnp.ndarray) or qt_obj is None:
        return qt_obj
    return jnp.array(qt_obj, dtype=dtype)


def jax2qt(jax_obj, dims=None):
    """JAX array -> QuTiP state.

    Args:
        jax_obj: JAX array.
        dims: QuTiP dims.

    Returns:
        QuTiP state.
    """
    if isinstance(jax_obj, Qobj) or jax_obj is None:
        return jax_obj
    if dims is not None:
        dims = np.array(dims).astype(int).tolist()
    return Qobj(np.array(jax_obj), dims=dims)

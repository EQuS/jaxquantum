""" States. """

from jax import config
from jax.nn import one_hot

import jax.numpy as jnp

from jaxquantum.core.operators import displace


config.update("jax_enable_x64", True)

def basis(N, k):
    """Creates a |k> (i.e. fock state) ket in a specified Hilbert Space.

    Args:
        N: Hilbert space dimension
        k: fock number

    Returns:
        Fock State |k>
    """
    return one_hot(k, N).reshape(N, 1)


def coherent(N, α) -> jnp.ndarray:
    """Coherent state.

    TODO: add trimming!

    Args:
        N: Hilbert Space Size.
        α: coherent state amplitude.

    Return:
        Coherent state |α⟩.
    """
    return displace(N, α) @ basis(N, 0)
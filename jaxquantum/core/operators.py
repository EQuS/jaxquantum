""" States. """

from jax import config

import jax.numpy as jnp
from jax.nn import one_hot

from jaxquantum.core.qarray import Qarray

config.update("jax_enable_x64", True)



def sigmax() -> jnp.ndarray:
    """σx

    Returns:
        σx Pauli Operator
    """
    return Qarray.create(jnp.array([[0.0, 1.0], [1.0, 0.0]]))


def sigmay() -> jnp.ndarray:
    """σy

    Returns:
        σy Pauli Operator
    """
    return Qarray.create(jnp.array([[0.0, -1.0j], [1.0j, 0.0]]))


def sigmaz() -> jnp.ndarray:
    """σz

    Returns:
        σz Pauli Operator
    """
    return Qarray.create(jnp.array([[1.0, 0.0], [0.0, -1.0]]))


def sigmam() -> jnp.ndarray:
    """σ-

    Returns:
        σ- Pauli Operator
    """
    return Qarray.create(jnp.array([[0.0, 0.0], [1.0, 0.0]]))


def sigmap() -> jnp.ndarray:
    """σ+

    Returns:
        σ+ Pauli Operator
    """
    return Qarray.create(jnp.array([[0.0, 1.0], [0.0, 0.0]]))


def destroy(N) -> jnp.ndarray:
    """annihilation operator

    Args:
        N: Hilbert space size

    Returns:
        annilation operator in Hilber Space of size N
    """
    return Qarray.create(jnp.diag(jnp.sqrt(jnp.arange(1, N)), k=1))


def create(N) -> jnp.ndarray:
    """creation operator

    Args:
        N: Hilbert space size

    Returns:
        creation operator in Hilber Space of size N
    """
    return Qarray.create(jnp.diag(jnp.sqrt(jnp.arange(1, N)), k=-1))


def num(N) -> jnp.ndarray:
    """Number operator

    Args:
        N: Hilbert Space size

    Returns:
        number operator in Hilber Space of size N
    """
    return Qarray.create(jnp.diag(jnp.arange(N)))


def identity(*args, **kwargs) -> jnp.ndarray:
    """Identity matrix.

    Returns:
        Identity matrix.
    """
    return Qarray.create(jnp.eye(*args, **kwargs))


def displace(N, α) -> jnp.ndarray:
    """Displacement operator

    Args:
        N: Hilbert Space Size
        α: Phase space displacement

    Returns:
        Displace operator D(α)
    """
    a = destroy(N)
    return (α * a.dag() - jnp.conj(α) * a).expm()


# States ---------------------------------------------------------------------

def basis(N, k):
    """Creates a |k> (i.e. fock state) ket in a specified Hilbert Space.

    Args:
        N: Hilbert space dimension
        k: fock number

    Returns:
        Fock State |k>
    """
    return Qarray.create(one_hot(k, N).reshape(N, 1))


def coherent(N, α) -> jnp.ndarray:
    """Coherent state.

    Args:
        N: Hilbert Space Size.
        α: coherent state amplitude.

    Return:
        Coherent state |α⟩.
    """
    return displace(N, α) @ basis(N, 0)
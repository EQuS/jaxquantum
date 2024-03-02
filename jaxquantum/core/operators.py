""" States. """

from jax import config

import jax.numpy as jnp

from jaxquantum.core.operations import expm, dag


config.update("jax_enable_x64", True)



def sigmax() -> jnp.ndarray:
    """σx

    Returns:
        σx Pauli Operator
    """
    return jnp.array([[0.0, 1.0], [1.0, 0.0]])


def sigmay() -> jnp.ndarray:
    """σy

    Returns:
        σy Pauli Operator
    """
    return jnp.array([[0.0, -1.0j], [1.0j, 0.0]])


def sigmaz() -> jnp.ndarray:
    """σz

    Returns:
        σz Pauli Operator
    """
    return jnp.array([[1.0, 0.0], [0.0, -1.0]])


def sigmam() -> jnp.ndarray:
    """σ-

    Returns:
        σ- Pauli Operator
    """
    return jnp.array([[0.0, 0.0], [1.0, 0.0]])


def sigmap() -> jnp.ndarray:
    """σ+

    Returns:
        σ+ Pauli Operator
    """
    return jnp.array([[0.0, 1.0], [0.0, 0.0]])


def destroy(N) -> jnp.ndarray:
    """annihilation operator

    Args:
        N: Hilbert space size

    Returns:
        annilation operator in Hilber Space of size N
    """
    return jnp.diag(jnp.sqrt(jnp.arange(1, N)), k=1)


def create(N) -> jnp.ndarray:
    """creation operator

    Args:
        N: Hilbert space size

    Returns:
        creation operator in Hilber Space of size N
    """
    return jnp.diag(jnp.sqrt(jnp.arange(1, N)), k=-1)


def num(N) -> jnp.ndarray:
    """Number operator

    Args:
        N: Hilbert Space size

    Returns:
        number operator in Hilber Space of size N
    """
    return jnp.diag(jnp.arange(N))


def identity(*args, **kwargs) -> jnp.ndarray:
    """Identity matrix.

    Returns:
        Identity matrix.
    """
    return jnp.eye(*args, **kwargs)


def displace(N, α) -> jnp.ndarray:
    """Displacement operator

    Args:
        N: Hilbert Space Size
        α: Phase space displacement

    Returns:
        Displace operator D(α)
    """
    a = destroy(N)
    return expm(α * dag(a) - jnp.conj(α) * a)
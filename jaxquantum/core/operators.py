""" States. """

from jax import config
from math import prod

import jax.numpy as jnp
from jax.nn import one_hot

from jaxquantum.core.qarray import Qarray

config.update("jax_enable_x64", True)



def sigmax() -> Qarray:
    """σx

    Returns:
        σx Pauli Operator
    """
    return Qarray.create(jnp.array([[0.0, 1.0], [1.0, 0.0]]))


def sigmay() -> Qarray:
    """σy

    Returns:
        σy Pauli Operator
    """
    return Qarray.create(jnp.array([[0.0, -1.0j], [1.0j, 0.0]]))


def sigmaz() -> Qarray:
    """σz

    Returns:
        σz Pauli Operator
    """
    return Qarray.create(jnp.array([[1.0, 0.0], [0.0, -1.0]]))


def hadamard() -> Qarray:
    """H

    Returns:
        H: Hadamard gate
    """
    return Qarray.create(jnp.array([[1, 1], [1, -1]]) / jnp.sqrt(2))

def sigmam() -> Qarray:
    """σ-

    Returns:
        σ- Pauli Operator
    """
    return Qarray.create(jnp.array([[0.0, 0.0], [1.0, 0.0]]))


def sigmap() -> Qarray:
    """σ+

    Returns:
        σ+ Pauli Operator
    """
    return Qarray.create(jnp.array([[0.0, 1.0], [0.0, 0.0]]))


def destroy(N) -> Qarray:
    """annihilation operator

    Args:
        N: Hilbert space size

    Returns:
        annilation operator in Hilber Space of size N
    """
    return Qarray.create(jnp.diag(jnp.sqrt(jnp.arange(1, N)), k=1))


def create(N) -> Qarray:
    """creation operator

    Args:
        N: Hilbert space size

    Returns:
        creation operator in Hilber Space of size N
    """
    return Qarray.create(jnp.diag(jnp.sqrt(jnp.arange(1, N)), k=-1))


def num(N) -> Qarray:
    """Number operator

    Args:
        N: Hilbert Space size

    Returns:
        number operator in Hilber Space of size N
    """
    return Qarray.create(jnp.diag(jnp.arange(N)))


def identity(*args, **kwargs) -> Qarray:
    """Identity matrix.

    Returns:
        Identity matrix.
    """
    return Qarray.create(jnp.eye(*args, **kwargs))

def identity_like(A) -> Qarray:
    """Identity matrix with the same shape as A.

    Args:
        A: Matrix.

    Returns:
        Identity matrix with the same shape as A.
    """
    space_dims = A.space_dims 
    total_dim = prod(space_dims)
    return Qarray.create(jnp.eye(total_dim, total_dim), dims=[space_dims, space_dims])


def displace(N, α) -> Qarray:
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


def coherent(N, α) -> Qarray:
    """Coherent state.

    Args:
        N: Hilbert Space Size.
        α: coherent state amplitude.

    Return:
        Coherent state |α⟩.
    """
    return displace(N, α) @ basis(N, 0)
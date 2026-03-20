"""States."""

from typing import List
from jax import config
from math import prod

import jax.numpy as jnp
from jax.nn import one_hot

from jaxquantum.core.qarray import Qarray, tensor, QarrayImplType

config.update("jax_enable_x64", True)


def sigmax(implementation: QarrayImplType = QarrayImplType.DENSE) -> Qarray:
    """σx

    Args:
        implementation: Qarray implementation type, e.g. "sparse" or "dense".

    Returns:
        σx Pauli Operator
    """
    return Qarray.create(jnp.array([[0.0, 1.0], [1.0, 0.0]]), implementation=implementation)


def sigmay(implementation: QarrayImplType = QarrayImplType.DENSE) -> Qarray:
    """σy

    Returns:
        σy Pauli Operator
    """
    return Qarray.create(jnp.array([[0.0, -1.0j], [1.0j, 0.0]]), implementation=implementation)


def sigmaz(implementation: QarrayImplType = QarrayImplType.DENSE) -> Qarray:
    """σz

    Returns:
        σz Pauli Operator
    """
    return Qarray.create(jnp.array([[1.0, 0.0], [0.0, -1.0]]), implementation=implementation)


def hadamard(implementation: QarrayImplType = QarrayImplType.DENSE) -> Qarray:
    """H

    Returns:
        H: Hadamard gate
    """
    return Qarray.create(jnp.array([[1, 1], [1, -1]]) / jnp.sqrt(2), implementation=implementation)


def sigmam(implementation: QarrayImplType = QarrayImplType.DENSE) -> Qarray:
    """σ-

    Returns:
        σ- Pauli Operator
    """
    return Qarray.create(jnp.array([[0.0, 0.0], [1.0, 0.0]]), implementation=implementation)


def sigmap(implementation: QarrayImplType = QarrayImplType.DENSE) -> Qarray:
    """σ+

    Returns:
        σ+ Pauli Operator
    """
    return Qarray.create(jnp.array([[0.0, 1.0], [0.0, 0.0]]), implementation=implementation)


def qubit_rotation(theta: float, nx, ny, nz) -> Qarray:
    """Single qubit rotation.

    Args:
        theta: rotation angle.
        nx: rotation axis x component.
        ny: rotation axis y component.
        nz: rotation axis z component.

    Returns:
        Single qubit rotation operator.
    """
    return jnp.cos(theta / 2) * identity(2) - 1j * jnp.sin(theta / 2) * (
        nx * sigmax() + ny * sigmay() + nz * sigmaz()
    )


def destroy(N, implementation: QarrayImplType = QarrayImplType.DENSE) -> Qarray:
    """annihilation operator

    Args:
        N: Hilbert space size
        implementation: Qarray implementation type, e.g. "sparse" or "dense".

    Returns:
        annilation operator in Hilber Space of size N
    """
    return Qarray.create(jnp.diag(jnp.sqrt(jnp.arange(1, N)), k=1), implementation=implementation)


def create(N, implementation: QarrayImplType = QarrayImplType.DENSE) -> Qarray:
    """creation operator

    Args:
        N: Hilbert space size
        implementation: Qarray implementation type, e.g. "sparse" or "dense".

    Returns:
        creation operator in Hilber Space of size N
    """
    return Qarray.create(jnp.diag(jnp.sqrt(jnp.arange(1, N)), k=-1), implementation=implementation)


def num(N, implementation: QarrayImplType = QarrayImplType.DENSE) -> Qarray:
    """Number operator

    Args:
        N: Hilbert Space size
        implementation: Qarray implementation type, e.g. "sparse" or "dense".

    Returns:
        number operator in Hilber Space of size N
    """
    return Qarray.create(jnp.diag(jnp.arange(N)), implementation=implementation)


def identity(*args, implementation: QarrayImplType = QarrayImplType.DENSE, **kwargs) -> Qarray:
    """Identity matrix.

    Args:
        implementation: Qarray implementation type, e.g. "sparse" or "dense".

    Returns:
        Identity matrix.
    """
    return Qarray.create(jnp.eye(*args, **kwargs), implementation=implementation)


qeye = identity

def identity_like(A, implementation: QarrayImplType = QarrayImplType.DENSE) -> Qarray:
    """Identity matrix with the same shape as A.

    Args:
        A: Matrix.
        implementation: Qarray implementation type, e.g. "sparse" or "dense".

    Returns:
        Identity matrix with the same shape as A.
    """
    space_dims = A.space_dims
    total_dim = prod(space_dims)
    return Qarray.create(jnp.eye(total_dim, total_dim), dims=[space_dims, space_dims], implementation=implementation)


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

def squeeze(N, z):
    """Single-mode Squeezing operator.


    Args:
        N: Hilbert Space Size
        z: squeezing parameter

    Returns:
        Sqeezing operator
    """
    
    a = destroy(N)
    op = (1 / 2.0) * jnp.conj(z) * (a @ a) - (1 / 2.0) * z * (a.dag() @ a.dag())
    return op.expm()


def squeezing_linear_to_dB(z):
    return 20 * jnp.log10(jnp.exp(jnp.abs(z)))

def squeezing_dB_to_linear(z_dB):
    return jnp.log(10**(z_dB/20))

# States ---------------------------------------------------------------------


def basis(N: int, k: int, implementation: QarrayImplType = QarrayImplType.DENSE):
    """Creates a |k> (i.e. fock state) ket in a specified Hilbert Space.

    Args:
        N: Hilbert space dimension
        k: fock number
        implementation: Qarray implementation type, e.g. "sparse" or "dense".

    Returns:
        Fock State |k>
    """
    return Qarray.create(one_hot(k, N).reshape(N, 1), implementation=implementation)

def multi_mode_basis_set(Ns: List[int]) -> Qarray:
    """Creates a multi-mode basis set.

    Args:
        Ns: List of Hilbert space dimensions for each mode.

    Returns:
        Multi-mode basis set.
    """
    data = jnp.eye(prod(Ns))
    dims = (tuple(Ns), tuple([1 for _ in Ns]))
    return Qarray.create(data, dims=dims, bdims=(prod(Ns),))


def coherent(N: int, α: complex) -> Qarray:
    """Coherent state.

    Args:
        N: Hilbert Space Size.
        α: coherent state amplitude.

    Return:
        Coherent state |α⟩.
    """
    return displace(N, α) @ basis(N, 0)


def thermal_dm(N: int, n: float) -> Qarray:
    """Thermal state.

    Args:
        N: Hilbert Space Size.
        n: average photon number.

    Return:
        Thermal state.
    """

    beta = jnp.log(1 + 1 / n)

    return Qarray.create(
        jnp.where(
            jnp.isposinf(beta),
            basis(N, 0).to_dm().data,
            jnp.diag(jnp.exp(-beta * jnp.linspace(0, N - 1, N))),
        )
    ).unit()


def basis_like(A: Qarray, ks: List[int]) -> Qarray:
    """Creates a |k> (i.e. fock state) ket with the same space dims as A.

    Args:
        A: state or operator.
        k: fock number.

    Returns:
        Fock State |k> with the same space dims as A.
    """
    space_dims = A.space_dims
    assert len(space_dims) == len(ks), "len(ks) must be equal to len(space_dims)"

    kets = []
    for j, k in enumerate(ks):
        kets.append(basis(space_dims[j], k))
    return tensor(*kets)

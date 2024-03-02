""" Relevant linear algebra operations. """

from jax import config, Array

import jax.numpy as jnp
import jax.scipy as jsp

from jaxquantum.utils.utils import is_1d

config.update("jax_enable_x64", True)



# Kets & Density Matrices -----------------------------------------------------

def ket(vec: Array) -> Array:
    """Turns a vector array into a ket.

    Args:
        vec: vector

    Returns:
        ket
    """
    vec = jnp.asarray(vec)
    return vec.reshape(vec.shape[0], 1)


def ket2dm(ket: jnp.ndarray) -> jnp.ndarray:
    """Turns ket into density matrix.

    Args:
        ket: ket

    Returns:
        Density matrix
    """
    ket = ket.reshape(ket.shape[0], 1)
    return ket @ dag(ket)



# Linear Algebra Operations -----------------------------------------------------


def dag(op: jnp.ndarray) -> jnp.ndarray:
    """Conjugate transpose.

    Args:
        op: operator

    Returns:
        conjugate transpose of op
    """
    op = op.reshape(op.shape[0], -1)  # adds dimension to 1D array if needed
    return jnp.conj(op).T


def batch_dag(op: jnp.ndarray) -> jnp.ndarray:
    """Conjugate transpose.

    Args:
        op: operator

    Returns:
        conjugate of op, and transposes last two axes
    """
    return jnp.moveaxis(
        jnp.conj(op), -1, -2
    )  # transposes last two axes, good for batching


def unit(rho: jnp.ndarray, use_density_matrix=False):
    """Normalize density matrix.

    Args:
        rho: density matrix

    Returns:
        normalized density matrix
    """
    if use_density_matrix:
        evals, _ = jnp.linalg.eigh(rho @ jnp.conj(rho).T)
        rho_norm = jnp.sum(jnp.sqrt(jnp.abs(evals)))
        return rho / rho_norm
    return rho / jnp.linalg.norm(rho)

def ptrace(rho, indx, dims):
    """Partial Trace.

    Args:
        rho: density matrix
        indx: index to trace out
        dims: list of dimensions of the tensored hilbert spaces

    Returns:
        partial traced out density matrix

    TODO: Fix weird tracing errors that arise with reshape
    """
    if is_1d(rho):
        rho = ket2dm(rho)

    Nq = len(dims)

    if isinstance(dims, jnp.ndarray):
        dims2 = jnp.concatenate(jnp.array([dims, dims]))
    else:
        dims2 = dims + dims

    rho = rho.reshape(dims2)

    indxs = [indx, indx + Nq]
    for j in range(Nq):
        if j == indx:
            continue
        indxs.append(j)
        indxs.append(j + Nq)
    rho = rho.transpose(indxs)

    for j in range(Nq - 1):
        rho = jnp.trace(rho, axis1=2, axis2=3)

    return rho


def expm(*args, **kwargs) -> jnp.ndarray:
    """Matrix exponential wrapper.

    Returns:
        matrix exponential
    """
    return jsp.linalg.expm(*args, **kwargs)


def tensor(*args, **kwargs) -> jnp.ndarray:
    """Tensor product.

    Args:
        *args: tensors to take the product of

    Returns:
        Tensor product of given tensors

    """
    ans = args[0]
    for arg in args[1:]:
        ans = jnp.kron(ans, arg)
    return ans


def tr(*args, **kwargs) -> jnp.ndarray:
    """Full trace.

    Returns:
        Full trace.
    """
    return jnp.trace(*args, **kwargs)

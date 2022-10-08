"""
Common jax <-> qutip-inspired functions
"""

from jax.config import config
from jax.nn import one_hot
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import qutip as qt

from jaxquantum.utils.utils import is_1d

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
    if isinstance(jax_obj, qt.Qobj) or jax_obj is None:
        return jax_obj
    if dims is not None:
        dims = np.array(dims).astype(int).tolist()
    return qt.Qobj(np.array(jax_obj), dims=dims)


# QuTiP alternatives in JAX (some are a WIP)
# ===============================================================


def unit(rho, use_density_matrix=False):
    if use_density_matrix:
        evals, _ = jnp.linalg.eigh(rho @ jnp.conj(rho).T)
        rho_norm = jnp.sum(jnp.sqrt(jnp.abs(evals)))
        return rho / rho_norm
    return rho / jnp.linalg.norm(rho)


def dag(op: jnp.ndarray) -> jnp.ndarray:
    return jnp.conj(op).T


def ket2dm(ket: jnp.ndarray) -> jnp.ndarray:
    ket = ket.reshape(ket.shape[0], 1)
    return ket @ dag(ket)


def basis(N, k):
    return one_hot(k, N).reshape(N, 1)


def sigmax() -> jnp.ndarray:
    return jnp.array([[0.0, 1.0], [1.0, 0.0]])


def sigmay() -> jnp.ndarray:
    return jnp.array([[0.0, -1.0j], [1.0j, 0.0]])


def sigmaz() -> jnp.ndarray:
    return jnp.array([[1.0, 0.0], [0.0, -1.0]])


def sigmam() -> jnp.ndarray:
    return jnp.array([[0.0, 0.0], [1.0, 0.0]])


def sigmap() -> jnp.ndarray:
    return jnp.array([[0.0, 1.0], [0.0, 0.0]])


def destroy(N) -> jnp.ndarray:
    return jnp.diag(jnp.sqrt(jnp.arange(1, N)), k=1)


def create(N) -> jnp.ndarray:
    return jnp.diag(jnp.sqrt(jnp.arange(1, N)), k=-1)


def num(N) -> jnp.ndarray:
    return jnp.diag(jnp.arange(N))


def coherent(N, a) -> jnp.ndarray:
    # TODO: replace with JAX implementation
    return qt2jax(qt.coherent(int(N), complex(a)))


def identity(*args, **kwargs) -> jnp.ndarray:
    return jnp.eye(*args, **kwargs)


def displace(N, alpha) -> jnp.ndarray:
    # TODO: replace with JAX implementation
    return qt2jax(qt.displace(int(N), float(alpha)))


def ptrace(rho, indx, dims):
    """
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
    return jsp.linalg.expm(*args, **kwargs)


def tensor(*args, **kwargs) -> jnp.ndarray:
    ans = args[0]
    for arg in args[1:]:
        ans = jnp.kron(ans, arg)
    return ans


def tr(*args, **kwargs) -> jnp.ndarray:
    return jnp.trace(*args, **kwargs)

"""
JAX Utils
"""

from numbers import Number
from typing import Dict

from jax import lax, jit
from jax import device_put
from jax.config import config
from jax._src.scipy.special import gammaln
import jax.numpy as jnp
import numpy as np

config.update("jax_enable_x64", True)


def device_put_params(params: Dict, non_device_params=None):
    non_device_params = [] if non_device_params is None else non_device_params
    for param in params:
        if param in non_device_params:
            continue
        if isinstance(params[param], Number) or isinstance(params[param], np.ndarray):
            params[param] = device_put(params[param])
    return params


def comb(N, k):
    """
    NCk

    #TODO: replace with jsp.special.comb once issue is closed:
    https://github.com/google/jax/issues/9709

    Args:
        N: total items
        k: # of items to choose

    Returns:
        NCk: N choose k
    """
    one = 1
    N_plus_1 = lax.add(N, one)
    k_plus_1 = lax.add(k, one)
    return lax.exp(
        lax.sub(
            gammaln(N_plus_1), lax.add(gammaln(k_plus_1), gammaln(lax.sub(N_plus_1, k)))
        )
    )


def is_1d(jax_obj) -> bool:
    return len(jax_obj.shape) == 1 or jax_obj.shape[1] == 1


@jit
def complex_to_real_iso_matrix(A): 
    return jnp.block([[jnp.real(A), -jnp.imag(A)], [jnp.imag(A), jnp.real(A)]])


@jit
def real_to_complex_iso_matrix(A):
    N = A.shape[0]
    return A[:N//2,:N//2] + 1j*A[N//2:,:N//2]


@jit
def complex_to_real_iso_vector(v): 
    return jnp.block([[jnp.real(v)], [jnp.imag(v)]])


@jit
def real_to_complex_iso_vector(v): 
    N = v.shape[0]
    return v[:N//2,:] + 1j*v[N//2:,:]


@jit
def imag_times_iso_vector(v):
    N = v.shape[0]
    return jnp.block([[-v[N//2:,:]], [v[:N//2,:]]])


@jit
def imag_times_iso_matrix(A):
    N = A.shape[0]
    Ar = A[:N//2,:N//2]
    Ai = A[N//2:,:N//2]
    return jnp.block([[-Ai, -Ar],[Ar,-Ai]])

@jit
def conj_transpose_iso_matrix(A):
    N = A.shape[0]
    Ar = A[:N//2,:N//2].T
    Ai = A[N//2:,:N//2].T
    return jnp.block([[Ar, Ai],[-Ai,Ar]])

"""
JAX Utils
"""

from numbers import Number
from typing import Dict

from jax import lax
from jax import device_put
from jax.config import config
from jax._src.scipy.special import gammaln
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

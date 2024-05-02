"""
Converting between different object types.
"""

from numbers import Number
from jax import config, Array
from qutip import Qobj
from typing import Optional, Union, List
import jax.numpy as jnp
import numpy as np


from jaxquantum.core.qarray import Qarray, DIMS_TYPE


config.update("jax_enable_x64", True)

# Convert between QuTiP and JAX
# ===============================================================
def qt2jqt(qt_obj, dtype=jnp.complex128):
    """QuTiP state -> Qarray.

    Args:
        qt_obj: QuTiP state.
        dtype: JAX dtype.

    Returns:
        Qarray.
    """
    if isinstance(qt_obj, Qarray) or qt_obj is None:
        return qt_obj
    return Qarray.create(jnp.array(qt_obj, dtype=dtype), dims=qt_obj.dims)


def jqt2qt(jqt_obj):
    """Qarray -> QuTiP state.

    Args:
        jqt_obj: Qarray.
        dims: QuTiP dims.

    Returns:
        QuTiP state.
    """
    if isinstance(jqt_obj, Qobj) or jqt_obj is None:
        return jqt_obj
    
    return Qobj(np.array(jqt_obj.data), dims=jqt_obj.dims)


def extract_dims(arr: Array, dims: Optional[Union[DIMS_TYPE, List[int]]] = None):
    """Extract dims from a JAX array or Qarray.

    Args:
        arr: JAX array or Qarray.
        dims: Qarray dims.

    Returns:
        Qarray dims.
    """
    if isinstance(dims[0], Number):
        is_op = len(arr.shape) == 2 and arr.shape[0] == arr.shape[1]
        if is_op:
            dims = [dims, dims]
        else:
            dims = [dims, [1] * len(dims)] # defaults to ket 
    return dims
                 

def jnp2jqt(arr: Array, dims: Optional[Union[DIMS_TYPE, List[int]]] = None):
    """JAX array -> QuTiP state.

    Args:
        jnp_obj: JAX array.
        dims: Qarray dims.

    Returns:
        QuTiP state.
    """
    dims = extract_dims(arr, dims) if dims is not None else None
    return Qarray.create(arr, dims=dims)


def jnps2jqts(arrs: Array, dims: Optional[DIMS_TYPE] = None):
    """JAX array -> QuTiP state.

    Args:
        jnp_obj: JAX array.

    Returns:
        QuTiP state.
    """

    dims = extract_dims(arrs[0], dims) if dims is not None else None
    return [Qarray.create(arr, dims=dims) for arr in arrs]

def jqts2jnps(qarrs: Qarray):
    """QuTiP state -> JAX array.

    Args:
        qt_obj: QuTiP state.

    Returns:
        JAX array.
    """
    return jnp.array([qarr.data for qarr in qarrs])
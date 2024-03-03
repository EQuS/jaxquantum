"""
Common jax <-> qutip-inspired functions
"""

from jax import config
from qutip import Qobj
import jax.numpy as jnp
import numpy as np


from jaxquantum.core.qarray import Qarray


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

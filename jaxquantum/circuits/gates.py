""" Gates. """

import functools
from flax import struct
from enum import Enum
from jax import Array, config
from typing import List
from math import prod
from copy import deepcopy
from numbers import Number
import jax.numpy as jnp
import jax.scipy as jsp


from jaxquantum.core.settings import SETTINGS
from jaxquantum.core.qarray import Qarray

config.update("jax_enable_x64", True)


@struct.dataclass
class Gate:
    _U: Array
    _H: Array
    _params
    _ts: Array
    _name: str = struct.field(pytree_node=False)
    
    @classmethod
    def create(
        name: str = "Gate",
        params: Optional[Dict[str, Any]] = None,
        ts: Optional[Array] = None,
        gen_U: Optional[Callable[[Dict[str, Any]], Qarray]] = None,
        gen_H: Optional[Callable[[Dict[str, Any]], Qarray]] = None,
    ):

        # TODO: add params to device?
        
        return Gate(
            _U = gen_U(params) if gen_U is not None else jnp.array([]),
            _H = gen_H(params) if gen_H is not None else jnp.array([]),
            _params = params if params is not None else {},
            _ts=ts if ts is not None else jnp.array([]),
            _name = name,
        )
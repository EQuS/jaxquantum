""" Circuits. """

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

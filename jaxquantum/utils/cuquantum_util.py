
from cuquantum.densitymat.jax import (  # noqa: E402
    OperatorTerm,
)

from math import prod




OperatorTerm.shape = property(lambda self: (prod(self.dims), prod(self.dims)))
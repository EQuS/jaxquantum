from math import prod

from cuquantum.densitymat.jax import (
    OperatorTerm,
)

OperatorTerm.shape = property(lambda self: (prod(self.dims), prod(self.dims)))

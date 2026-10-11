"""Small compatibility helpers for the optional cuQuantum JAX extension."""

from math import prod

from cuquantum.densitymat.jax import OperatorTerm


if not hasattr(OperatorTerm, "shape"):
    OperatorTerm.shape = property(lambda self: (prod(self.dims), prod(self.dims)))

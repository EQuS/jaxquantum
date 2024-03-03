""" Relevant linear algebra operations. """

from jax import config, Array

import jax.numpy as jnp


config.update("jax_enable_x64", True)


# Linear Algebra Operations -----------------------------------------------------

def batch_dag_data(op: Array) -> Array:
    """Conjugate transpose.

    Args:
        op: operator

    Returns:
        conjugate of op, and transposes last two axes
    """
    return jnp.moveaxis(
        jnp.conj(op), -1, -2
    )  # transposes last two axes, good for batching









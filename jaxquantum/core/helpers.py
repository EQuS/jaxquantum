""" Helpers. """

from typing import List
from jax import config
from math import prod

import jax.numpy as jnp
from jax.nn import one_hot

from jaxquantum.core.qarray import Qarray, Qtypes, tensor, powm
from jaxquantum.core.dims import Qtypes

config.update("jax_enable_x64", True)


def isvec(A: Qarray) -> bool:
    """Check if A is a ket or bra.

    Args:
        A: state.

    Returns:
        True if A is a ket or bra, False otherwise.
    """
    return A.qtype == Qtypes.ket or A.qtype == Qtypes.bra

# Calculations ----------------------------------------------------------------

def overlap(A: Qarray, B: Qarray) -> complex:
    """Overlap between two states.

    Args:
        A: state.
        B: state.

    Returns:
        Overlap between A and B.
    """
    # A.qtype

    if isvec(A) and isvec(B):
        return jnp.abs(((A.to_ket().dag() @ B.to_ket()).trace()))**2
    elif isvec(A):
        A = A.to_ket()
        res = (A.dag() @ B @ A).data
        return res.squeeze(-1).squeeze(-1)
    elif isvec(B):
        return overlap(B, A)
    else:
        return (A.dag() @ B).trace()


def fidelity(A: Qarray, B: Qarray) -> float:
    """Fidelity between two states.

    Args:
        A: state.
        B: state.

    Returns:
        Fidelity between A and B.
    """
    A = A.to_dm()
    B = B.to_dm()

    sqrtA = powm(A, 0.5)

    return ((powm(sqrtA @ B @ sqrtA, 0.5)).tr())**2


def quantum_state_tomography(A: Qarray, meas_basis: List, logical_basis: List) -> Qarray:
    """Perform quantum state tomography of a logical encoding on a physical
    qubit.

        Args:
            A: state.
            meas_basis: list of physical operators forming a complete basis
            in the physical Hilbert space.
            logical_basis: list of logical operators forming a complete
            basis in the logical Hilbert space.

        Returns:
            Logical density matrix of state A.
        """
    dm = jnp.zeros_like(meas_basis[0])

    for meas_op, logical_op in zip(meas_basis, logical_basis):
        p_i = (A @ meas_op).trace()
        dm += p_i * logical_op

    return Qarray.create(dm)


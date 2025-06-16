""" Helpers. """

from typing import List
from jax import config
from math import prod

import jax.numpy as jnp
from jax.nn import one_hot
from tqdm import tqdm

from jaxquantum.core.qarray import Qarray, Qtypes, tensor, powm
from jaxquantum.core.dims import Qtypes
from jaxquantum.core.operators import identity, sigmax, sigmay, sigmaz

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
    dm = jnp.zeros_like(logical_basis[0].data)
    A = A.to_dm()

    for meas_op, logical_op in tqdm(zip(meas_basis, logical_basis), total=len(meas_basis)):
        p_i = (A @ meas_op).trace()
        dm += p_i * logical_op.data

    return Qarray.create(dm)


def get_physical_basis(qubits: List) -> List:

    qubit = qubits[0]
    qubits = qubits[1:]

    ops = [identity(qubit.params["N"]), qubit.common_gates["X"],
           qubit.common_gates["Y"], qubit.common_gates["Z"]]

    if len(qubits)==0:
        return ops

    sub_basis = get_physical_basis(qubits)
    basis = []

    for op in ops:
        for sub_op in sub_basis:
            basis.append(op ^ sub_op)

    return basis

def get_logical_basis(n_qubits: int) -> List:

    n_qubits -= 1

    ops = [identity(2)/2, sigmax()/2, sigmay()/2, sigmaz()/2]

    if n_qubits == 0:
        return ops

    sub_basis = get_logical_basis(n_qubits)
    basis = []

    for op in ops:
        for sub_op in sub_basis:
            basis.append(op ^ sub_op)

    return basis



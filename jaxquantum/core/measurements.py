""" Helpers. """

from typing import List
from jax import config, Array

import jax.numpy as jnp
from tqdm import tqdm

from jaxquantum.core.qarray import Qarray, powm
from jaxquantum.core.dims import Qtypes
from jaxquantum.core.operators import identity, sigmax, sigmay, sigmaz

config.update("jax_enable_x64", True)


# Calculations ----------------------------------------------------------------

def overlap(rho: Qarray, sigma: Qarray) -> Array:
    """Overlap between two states or operators.

    Args:
        rho: state/operator.
        sigma: state/operator.

    Returns:
        Overlap between rho and sigma.
    """

    if rho.isvec() and sigma.isvec():
        return jnp.abs(((rho.to_ket().dag() @ sigma.to_ket()).trace())) ** 2
    elif rho.isvec():
        rho = rho.to_ket()
        res = (rho.dag() @ sigma @ rho).data
        return res.squeeze(-1).squeeze(-1)
    elif sigma.isvec():
        sigma = sigma.to_ket()
        res = (sigma.dag() @ rho @ sigma).data
        return res.squeeze(-1).squeeze(-1)
    else:
        return (rho.dag() @ sigma).trace()


def fidelity(rho: Qarray, sigma: Qarray) -> float:
    """Fidelity between two states.

    Args:
        rho: state.
        sigma: state.

    Returns:
        Fidelity between rho and sigma.
    """
    rho = rho.to_dm()
    sigma = sigma.to_dm()

    sqrt_rho = powm(rho, 0.5)

    return ((powm(sqrt_rho @ sigma @ sqrt_rho, 0.5)).tr()) ** 2


def quantum_state_tomography(rho: Qarray, physical_basis: List, logical_basis:
List) -> Qarray: #TODO use batch dimensions instead of lists
    """Perform quantum state tomography to retrieve the density matrix in 
    the logical basis. 

        Args:
            rho: state expressed in the physical Hilbert space basis.
            physical_basis: list of logical operators expressed in the physical
            Hilbert space basis forming a complete logical operator basis.
            logical_basis: list of logical operators expressed in the 
            logical Hilbert space basis forming a complete operator basis.


        Returns:
            Density matrix of state rho expressed in the logical basis.
        """
    dm = jnp.zeros_like(logical_basis[0].data)
    rho = rho.to_dm()

    for meas_op, logical_op in tqdm(zip(physical_basis, logical_basis),
                                    total=len(physical_basis)):
        p_i = (rho @ meas_op).trace()
        dm += p_i * logical_op.data

    return Qarray.create(dm)


def get_physical_basis(qubits: List) -> List:
    """Compute a complete operator basis of a QEC code on a
    physical system specified by a number of qubits.

            Args:
                qubits: list of qubit codes, must have
                common_gates and params attributes.

            Returns:
                List containing the complete operator basis.
            """
    qubit = qubits[0]
    qubits = qubits[1:]
    try:
        operators = [identity(qubit.params["N"]), qubit.common_gates["X"],
                     qubit.common_gates["Y"], qubit.common_gates["Z"]]
    except KeyError:
        print("QEC code must have common_gates for all three axes.")
    except AttributeError:
        print("QEC code must have common_gates and params attribute.")

    if len(qubits) == 0:
        return operators

    sub_basis = get_physical_basis(qubits)
    basis = []

    for op in operators:
        for sub_op in sub_basis:
            basis.append(op ^ sub_op)

    return basis


def get_logical_basis(n_qubits: int) -> List:
    """Compute a complete operator basis of a system composed of logical
    qubits.

                Args:
                    n_qubits: number of qubits

                Returns:
                    List containing the complete operator basis.
                """
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1.")

    n_qubits -= 1

    operators = [identity(2) / 2, sigmax() / 2, sigmay() / 2, sigmaz() / 2]

    if n_qubits == 0:
        return operators

    sub_basis = get_logical_basis(n_qubits)
    basis = []

    for op in operators:
        for sub_op in sub_basis:
            basis.append(op ^ sub_op)

    return basis

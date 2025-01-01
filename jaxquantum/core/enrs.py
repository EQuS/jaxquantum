""" Enrs. """

# Adopted for jaxquantum from energy_restricted.py in QuTip 

import numpy as np
from jaxquantum.core.qarray import Qarray
import jax.numpy as jnp
from jax.nn import one_hot


def enr_state_dictionaries(dims: list, excitations: int) -> tuple[int, dict, dict]:
    """Generate state dictionaries about the restricted space

    Args:
        dims (list): List of dimensions of each subsystem
        excitations (int): number of total excitations for the system to be restricted
        
    Returns:
        nstates (int): the number of states
        state2idx (dict): dict from state tuple to integer index
        idx2state (dict): dict from integer index to state tuple
    """
    
    if any(dim <= 0 for dim in dims):
        raise ValueError("Invalid dims")
    if excitations < 0:
        raise ValueError("Invalid excitations")

    N = len(dims)
    state_list = []
    state_stack = [[0, ]]
    sum_stack = [0]
    while len(state_stack) != 0:
        state = state_stack.pop()
        partial_sum = sum_stack.pop()
        if len(state) == N:
            state_list.append(tuple(state))
        if len(state) < N:
            state_stack.append(state + [0,])
            sum_stack.append(partial_sum)
        if partial_sum < excitations and state[-1] < dims[len(state)-1]-1:
            new_state = state.copy()
            new_state[-1] += 1
            state_stack.append(new_state)
            sum_stack.append(partial_sum+1)

    nstates = len(state_list)
    state2idx = dict()
    idx2state = dict()

    for idx, state in enumerate(state_list):
        state2idx[state] = idx
        idx2state[idx] = state

    return nstates, state2idx, idx2state

def enr_fock(dims: list, excitations: int, state: list[int]) -> Qarray:
    """Return the ket living in the restricted space

    Args:
        dims (list): List of dimensions of each subsystem
        excitations (int): number of total excitations for the system to be restricted
        state (list[int]): state represented by excitation in each subspace
            
    Returns:
        ket (QArray): the ket array living in the restricted space defined by dim and excitations.
    """

    nstates, state2idx, _ = enr_state_dictionaries(dims, excitations)

    state_tuple = tuple(state)
    if state_tuple not in state2idx:
        raise ValueError('Given state is not in the enr space.')

    idx = state2idx[state_tuple]
    
    return Qarray.create(one_hot(idx, nstates).reshape(nstates, 1))
    

def enr_destroy(dims: list, excitations: int) -> list[Qarray]:
    """Generate a list of annihilation operator for each subsystem in the restricted space

    Args:
        dims (list): List of dimensions of each subsystem
        excitations (int): number of total excitations for the system to be restricted
        
    Returns:
        a_ops (list[QArray]): a list of annihilation operators for each subsystem to be used in the restricted space.
    """

    nstates, state2idx, idx2state = enr_state_dictionaries(dims, excitations)

    np_a_ops = [np.zeros((nstates, nstates)) for _ in range(len(dims)) ]
    for n, st in idx2state.items():
        for idx, s in enumerate(st):
            if s > 0:
                st2 = st[:idx] + (s-1,) + st[idx+1:]
                n2 = state2idx[st2]
                np_a_ops[idx][n2, n] = np.sqrt(s)

    return [Qarray.create(jnp.array(np_a_op)) for np_a_op in np_a_ops]

def enr_identity(dims: list, excitations: int) -> Qarray:
    """Return the identity operator in the restricted space

    Args:
        dims (list): List of dimensions of each subsystem
        excitations (int): number of total excitations for the system to be restricted
        
    Returns:
        identity (QArray): Identity operator in the enr space
    """

    nstates, _, _ = enr_state_dictionaries(dims, excitations)

    return Qarray.create(jnp.eye(nstates))
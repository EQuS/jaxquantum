""" Circuit simulation methods. """

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
from jaxquantum.circuits.circuits import Circuit

config.update("jax_enable_x64", True)

@struct.dataclass
class Result:
    states: List[Qarray] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        states: List[Qarray]
    ):
        return Result(
            states = states
        )

    def __getitem__(self, j: int):
        return self.states[j]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.states)

    def append(self, state: Qarray):
        self.states.append(state) 

    def __getitem__(self, j: int):
        return self.states[j]

    def __len__(self):
        return len(self.states)

@struct.dataclass
class Results:
    results: List[Result] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        results: List[Result]
    ):
        return Results(
            results = results
        )

    def __getitem__(self, j: int):
        return self.results[j]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.results)

    def append(self, result: Result):
        self.results.append(result)

    def __getitem__(self, j: int):
        return self.results[j]

    def __len__(self):
        return len(self.results)

def simulate(
    circuit: Circuit,
    initial_state: Qarray,
    mode: str = "unitary"
) -> Qarray:
    """Simulate a circuit on a state.

    Args:
        circuit: The circuit to simulate.
        state: The initial state to simulate the circuit on.
        mode: The simulation mode. Can be "unitary" or "hamiltonian".

    Returns:
        The state after simulating the circuit.
    """

    results = Results.create([])

    if mode == "unitary":
        state = initial_state
        results.append(Result.create([state]))

        for layer in circuit.layers:
            U = layer.gen_U()
            state = U @ state 
            results.append(Result.create([state]))
        
    return results


        
    

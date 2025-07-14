"""Circuit simulation methods."""

from flax import struct
from jax import config
from typing import List


from jaxquantum.core.qarray import Qarray, ket2dm
from jaxquantum.circuits.circuits import Circuit, Layer
from jaxquantum.circuits.constants import SimulateMode

config.update("jax_enable_x64", True)


@struct.dataclass
class Results:
    results: List[Qarray] = struct.field(pytree_node=False)

    @classmethod
    def create(cls, results: List[Qarray]):
        return Results(results=results)

    def __getitem__(self, j: int):
        return self.results[j]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.results)

    def append(self, result: Qarray):
        self.results.append(result)

    def __len__(self):
        return len(self.results)


def simulate(
    circuit: Circuit, initial_state: Qarray, mode: SimulateMode = SimulateMode.UNITARY
) -> Results:
    """
    Simulates the evolution of a quantum state through a given quantum circuit.

    Args:
        circuit (Circuit): The quantum circuit to simulate. The circuit is composed of layers,
                           each of which can generate unitary or Kraus operators.
        initial_state (Qarray): The initial quantum state to be evolved. This can be a state vector
                                or a density matrix.
        mode (SimulateMode, optional): The mode of simulation. It can be either SimulateMode.UNITARY
                                       for unitary evolution or SimulateMode.KRAUS for Kraus operator
                                       evolution. Defaults to SimulateMode.UNITARY.

    Returns:
        Results: An object containing the results of the simulation, which includes the quantum states
                 at each step of the circuit.
    """

    results = Results.create([])
    state = initial_state
    results.append(Qarray.from_list([state]))

    for layer in circuit.layers:
        result = simulate_layer(layer, state, mode=mode)
        results.append(result)
        state = result[-1]

    return results


def simulate_layer(
    layer: Layer, initial_state: Qarray, mode: SimulateMode = SimulateMode.UNITARY
) -> Qarray:
    """
    Simulates the evolution of a quantum state through a given layer.

    Args:
        layer (Layer): The layer through which the quantum state evolves.
                       This layer should have methods to generate unitary (gen_U)
                       and Kraus (gen_KM) operators.
        initial_state (Qarray): The initial quantum state to be evolved.
                                This can be a state vector or a density matrix.
        mode (SimulateMode, optional): The mode of simulation. It can be either
                                       SimulateMode.UNITARY for unitary evolution
                                       or SimulateMode.KRAUS for Kraus operator evolution
                                       or SimulateMode.DEFAULT to use the default simulate mode in the layer.
                                       Defaults to SimulateMode.UNITARY.
    Returns:
        Qarray: The result of the simulation containing the evolved quantum state.
    """

    state = initial_state

    if mode == SimulateMode.DEFAULT:
        mode = layer._default_simulate_mode

    if mode == SimulateMode.UNITARY:
        U = layer.gen_U()
        if state.is_dm():
            state = U @ state @ U.dag()
        else:
            state = U @ state

        result = Qarray.from_list([state])

    elif mode == SimulateMode.KRAUS:
        KM = layer.gen_KM()

        state = ket2dm(state)
        state = (KM @ state @ KM.dag()).collapse()

        # new_state = 0
        # for op_j in range(len(KM)):
        #     op = KM[op_j]
        #     new_state += op @ state @ op.dag()
        # state = new_state

        result = Qarray.from_list([state])

    return result

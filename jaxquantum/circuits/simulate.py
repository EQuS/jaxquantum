"""Circuit simulation methods."""

from flax import struct
from jax import config
from typing import List
from tqdm import tqdm


from jaxquantum.core.qarray import Qarray, ket2dm
from jaxquantum.circuits.circuits import Circuit, Layer
from jaxquantum.circuits.constants import SimulateMode
from jaxquantum.core.solvers import mesolve, sesolve, SolverOptions


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
    circuit: Circuit, initial_state: Qarray, mode: SimulateMode = SimulateMode.DEFAULT, **kwargs
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

    start_time = 0

    for layer in tqdm(circuit.layers):
        result_dict = _simulate_layer(layer, state, mode=mode, start_time=start_time, **kwargs)
        result = result_dict["result"]
        start_time = result_dict["start_time"]
        results.append(result)
        state = result[-1]

    return results


def _simulate_layer(
    layer: Layer, initial_state: Qarray, mode: SimulateMode = SimulateMode.UNITARY, start_time: float = 0, **kwargs
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

    elif mode == SimulateMode.HAMILTONIAN:

        solver_options = kwargs.get("solver_options", SolverOptions.create(progress_meter=True))

        Ht = layer.gen_Ht()
        c_ops = layer.gen_c_ops()
        ts = layer.gen_ts()

        ts = ts + start_time

        if state.is_dm() or (c_ops is not None and len(c_ops) > 0):
            intermediate_states = mesolve(Ht, state, ts, c_ops=c_ops, solver_options=solver_options)
        else:
            intermediate_states = sesolve(Ht, state, ts, solver_options=solver_options)

        result = intermediate_states
        state = intermediate_states[-1]
        start_time = ts[-1]

    elif mode == SimulateMode.KRAUS:
        KM = layer.gen_KM()

        state = ket2dm(state)
        state = (KM @ state @ KM.dag()).collapse()
        result = Qarray.from_list([state])

    return {
        "result": result,
        "start_time": start_time
    }

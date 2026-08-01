"""Circuit simulation methods."""

from flax import struct
from jax import config
import jax.numpy as jnp
from math import prod
from typing import List

from jaxquantum.core.qarray import DenseImpl, Qarray, Qtypes, ket2dm
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


def _apply_matrix_to_axes(data, matrix, target_axes, system_shape):
    """Apply a matrix to selected tensor axes without forming a full operator."""
    n_system_axes = len(system_shape)
    batch_shape = data.shape[:-n_system_axes]
    target_axes = tuple(target_axes)

    if target_axes == tuple(range(target_axes[0], target_axes[-1] + 1)):
        start, stop = target_axes[0], target_axes[-1] + 1
        left = prod(system_shape[:start])
        target = prod(system_shape[start:stop])
        right = prod(system_shape[stop:])
        data = data.reshape(batch_shape + (left, target, right))
        data = jnp.einsum("...ij,...ljr->...lir", matrix, data)
        return data.reshape(data.shape[:-3] + tuple(system_shape))

    other_axes = tuple(i for i in range(n_system_axes) if i not in target_axes)
    order = other_axes + target_axes
    n_batch_axes = len(batch_shape)

    data = jnp.transpose(
        data,
        tuple(range(n_batch_axes))
        + tuple(n_batch_axes + axis for axis in order),
    )
    other_shape = tuple(system_shape[axis] for axis in other_axes)
    target_shape = tuple(system_shape[axis] for axis in target_axes)
    data = data.reshape(batch_shape + (prod(other_shape), prod(target_shape)))
    data = jnp.einsum("...ij,...kj->...ki", matrix, data)

    out_batch_shape = data.shape[:-2]
    data = data.reshape(out_batch_shape + other_shape + target_shape)
    return jnp.transpose(
        data,
        tuple(range(len(out_batch_shape)))
        + tuple(len(out_batch_shape) + order.index(axis) for axis in range(n_system_axes)),
    )


def _apply_local_unitary(state: Qarray, operation) -> Qarray:
    dims = tuple(operation.register.dims)
    unitary = operation.gate.U.to_dense().data
    n_modes = len(dims)

    if state.qtype == Qtypes.ket:
        data = state.data.reshape(state.data.shape[:-1] + dims)
        data = _apply_matrix_to_axes(data, unitary, operation.indices, dims)
        data = data.reshape(data.shape[:-n_modes] + (prod(dims),))
    else:
        system_shape = dims + dims
        data = state.to_dense().data.reshape(state.data.shape[:-2] + system_shape)
        data = _apply_matrix_to_axes(data, unitary, operation.indices, system_shape)
        bra_axes = tuple(n_modes + index for index in operation.indices)
        data = _apply_matrix_to_axes(data, jnp.conj(unitary), bra_axes, system_shape)
        data = data.reshape(data.shape[:-2 * n_modes] + (prod(dims), prod(dims)))

    return Qarray._from_impl(DenseImpl._make(data), state._qdims)


def _single_state_batch(state: Qarray) -> Qarray:
    impl = type(state._impl).from_data(jnp.expand_dims(state.data, axis=0))
    return Qarray._from_impl(impl, state._qdims)


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
    results.append(_single_state_batch(state))

    start_time = 0

    for layer in circuit.layers:
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
        for operation in layer.operations:
            state = _apply_local_unitary(state, operation)
        result = _single_state_batch(state)

    elif mode == SimulateMode.HAMILTONIAN:

        solver_options = kwargs.pop("solver_options", SolverOptions.create(progress_meter=False))

        Ht = layer.gen_Ht()
        c_ops = layer.gen_c_ops()
        ts = layer.gen_ts()

        ts = ts + start_time

        if state.is_dm() or (c_ops is not None and len(c_ops) > 0):
            intermediate_states = mesolve(Ht, state, ts, c_ops=c_ops, solver_options=solver_options, **kwargs)
        else:
            intermediate_states = sesolve(Ht, state, ts, solver_options=solver_options, **kwargs)

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

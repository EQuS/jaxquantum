"""Circuit simulation methods."""

from math import prod

import jax
import jax.numpy as jnp
from flax import struct
from jax import config, lax

from jaxquantum.circuits.channels import apply_kraus_map
from jaxquantum.circuits.circuits import Circuit, Layer
from jaxquantum.circuits.constants import SimulateMode
from jaxquantum.core.measurements import overlap
from jaxquantum.core.qarray import DenseImpl, Qarray, Qtypes, ket2dm
from jaxquantum.core.solvers import SolverOptions, mesolve, sesolve, solve

config.update("jax_enable_x64", True)


@struct.dataclass
class Results:
    results: list[Qarray] = struct.field(pytree_node=False)

    @classmethod
    def create(cls, results: list[Qarray]):
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
        + tuple(
            len(out_batch_shape) + order.index(axis)
            for axis in range(n_system_axes)
        ),
    )


def _apply_local_unitary(state: Qarray, operation) -> Qarray:
    dims = tuple(operation.register.dims)
    unitary = operation.gate.U.to_dense().data
    n_modes = len(dims)
    state_data = state.to_dense().data

    if state.qtype == Qtypes.ket:
        data = state_data.reshape(state_data.shape[:-1] + dims)
        data = _apply_matrix_to_axes(data, unitary, operation.indices, dims)
        data = data.reshape(data.shape[:-n_modes] + (prod(dims),))
    else:
        system_shape = dims + dims
        data = state_data.reshape(state_data.shape[:-2] + system_shape)
        data = _apply_matrix_to_axes(data, unitary, operation.indices, system_shape)
        bra_axes = tuple(n_modes + index for index in operation.indices)
        data = _apply_matrix_to_axes(data, jnp.conj(unitary), bra_axes, system_shape)
        data = data.reshape(data.shape[:-2 * n_modes] + (prod(dims), prod(dims)))

    return Qarray._from_impl(DenseImpl._make(data), state._qdims)


def _apply_local_kraus(state: Qarray, operation) -> Qarray:
    """Apply a local Kraus map without promoting it to the full register."""
    state = ket2dm(state)
    dims = tuple(operation.register.dims)
    n_modes = len(dims)
    system_shape = dims + dims
    data = state.to_dense().data.reshape(state.data.shape[:-2] + system_shape)
    direct_apply = operation.gate.channel_apply
    kraus = None if direct_apply is not None else operation.gate.KM.to_dense().data
    if kraus is not None and kraus.shape[0] == 0:
        return state

    target_axes = tuple(operation.indices) + tuple(
        n_modes + index for index in operation.indices
    )
    other_axes = tuple(
        index for index in range(2 * n_modes) if index not in target_axes
    )
    order = other_axes + target_axes
    n_batch_axes = data.ndim - 2 * n_modes
    data = jnp.transpose(
        data,
        tuple(range(n_batch_axes))
        + tuple(n_batch_axes + index for index in order),
    )
    other_shape = tuple(system_shape[index] for index in other_axes)
    target_shape = tuple(dims[index] for index in operation.indices)
    target_size = prod(target_shape)
    data = data.reshape(
        data.shape[:n_batch_axes]
        + (prod(other_shape), target_size, target_size)
    )
    if direct_apply is not None:
        data = direct_apply(data, operation.gate.params)
    else:
        data = apply_kraus_map(kraus[..., None, :, :], data)
    out_batch_shape = data.shape[:-3]
    data = data.reshape(out_batch_shape + other_shape + target_shape + target_shape)
    data = jnp.transpose(
        data,
        tuple(range(len(out_batch_shape)))
        + tuple(
            len(out_batch_shape) + order.index(index)
            for index in range(2 * n_modes)
        ),
    )
    data = data.reshape(out_batch_shape + (prod(dims), prod(dims)))
    return Qarray._from_impl(DenseImpl._make(data), state._qdims)


def _local_batch_shape(layer, state_data, time, density_matrix):
    state_rank = 2 if density_matrix else 1
    batch_shape = state_data.shape[:-state_rank]
    for operation in layer.operations:
        if operation.gate.Ht is not None:
            shape = operation.gate.Ht(time).data.shape[:-2]
            batch_shape = jnp.broadcast_shapes(batch_shape, shape)
        if len(operation.gate.c_ops):
            shape = operation.gate.c_ops.data.shape[1:-2]
            batch_shape = jnp.broadcast_shapes(batch_shape, shape)
    return batch_shape


def _solve_local_hamiltonian(
    layer,
    state,
    times,
    saveat_times,
    solver_options,
):
    dims = tuple(layer.operations[0].register.dims)
    n_modes = len(dims)
    has_c_ops = any(len(operation.gate.c_ops) for operation in layer.operations)
    density_matrix = state.is_dm() or has_c_ops
    state = state.to_dm().to_dense() if density_matrix else state.to_ket().to_dense()
    qdims = state.qdims
    data = state.data
    batch_shape = _local_batch_shape(layer, data, times[0], density_matrix)
    state_rank = 2 if density_matrix else 1
    data = jnp.broadcast_to(data, batch_shape + data.shape[-state_rank:])

    def rhs(time, value, _):
        system_shape = dims + dims if density_matrix else dims
        tensor = value.reshape(value.shape[:-state_rank] + system_shape)
        derivative = jnp.zeros_like(tensor)

        for operation in layer.operations:
            indices = tuple(operation.indices)
            hamiltonian = operation.gate.Ht
            if hamiltonian is not None:
                matrix = hamiltonian(time).to_dense().data
                left = _apply_matrix_to_axes(tensor, matrix, indices, system_shape)
                if density_matrix:
                    bra_indices = tuple(n_modes + index for index in indices)
                    right = _apply_matrix_to_axes(
                        tensor,
                        jnp.swapaxes(matrix, -1, -2),
                        bra_indices,
                        system_shape,
                    )
                    derivative += -1.0j * (left - right)
                else:
                    derivative += -1.0j * left

            if not density_matrix or len(operation.gate.c_ops) == 0:
                continue
            collapse_ops = operation.gate.c_ops.to_dense().data
            bra_indices = tuple(n_modes + index for index in indices)

            def dissipator(
                matrix,
                indices=indices,
                bra_indices=bra_indices,
            ):
                left = _apply_matrix_to_axes(
                    tensor,
                    matrix,
                    indices,
                    system_shape,
                )
                sandwich = _apply_matrix_to_axes(
                    left,
                    jnp.conj(matrix),
                    bra_indices,
                    system_shape,
                )
                product = jnp.swapaxes(jnp.conj(matrix), -1, -2) @ matrix
                anti_left = _apply_matrix_to_axes(
                    tensor,
                    product,
                    indices,
                    system_shape,
                )
                anti_right = _apply_matrix_to_axes(
                    tensor,
                    jnp.swapaxes(product, -1, -2),
                    bra_indices,
                    system_shape,
                )
                return sandwich - 0.5 * (anti_left + anti_right)

            def add_dissipator(index, total, collapse_ops=collapse_ops):
                return total + dissipator(collapse_ops[index])

            derivative += lax.fori_loop(
                1,
                collapse_ops.shape[0],
                add_dissipator,
                dissipator(collapse_ops[0]),
            )

        return derivative.reshape(value.shape)

    ys = solve(
        rhs,
        data,
        times,
        saveat_times,
        None,
        solver_options=solver_options,
    ).ys
    return Qarray._from_impl(DenseImpl._make(ys), qdims)


def _single_state_batch(state: Qarray) -> Qarray:
    impl = type(state._impl).from_data(state.data.reshape(1, *state.data.shape))
    return Qarray._from_impl(impl, state._qdims)


def _evolve_circuit(circuit, state, mode, start_time=0.0, **kwargs):
    for layer in circuit.layers:
        output = _simulate_layer(
            layer,
            state,
            mode=mode,
            start_time=start_time,
            **kwargs,
        )
        state = output["result"][-1]
        start_time = output["start_time"]
    return state, start_time


def simulate(
    circuit: Circuit,
    initial_state: Qarray,
    mode: SimulateMode = SimulateMode.DEFAULT,
    save_states: bool = True,
    **kwargs,
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

    results = Results.create(
        [_single_state_batch(initial_state)] if save_states else []
    )
    state = initial_state
    start_time = 0
    if not save_states:
        kwargs.setdefault("saveat_tlist", jnp.array([]))

    for layer in circuit.layers:
        result_dict = _simulate_layer(
            layer,
            state,
            mode=mode,
            start_time=start_time,
            **kwargs,
        )
        result = result_dict["result"]
        start_time = result_dict["start_time"]
        state = result[-1]
        if save_states:
            results.append(result)

    if not save_states:
        results.append(_single_state_batch(state))
    return results


def simulate_final(
    circuit: Circuit,
    initial_state: Qarray,
    mode: SimulateMode = SimulateMode.DEFAULT,
    **kwargs,
) -> Qarray:
    """Return only the final circuit state."""
    kwargs.setdefault("saveat_tlist", jnp.array([]))
    return _evolve_circuit(circuit, initial_state, mode, **kwargs)[0]


def simulate_repeated(
    circuit: Circuit,
    initial_state: Qarray,
    repetitions: int,
    mode: SimulateMode = SimulateMode.DEFAULT,
    **kwargs,
) -> Qarray:
    """Apply one circuit repeatedly with a compiled loop."""
    if repetitions < 0:
        raise ValueError("repetitions must be non-negative")
    if repetitions == 0:
        return initial_state

    kwargs.setdefault("saveat_tlist", jnp.array([]))
    state, start_time = _evolve_circuit(
        circuit,
        initial_state,
        mode,
        **kwargs,
    )

    def repeat(_, carry):
        return _evolve_circuit(circuit, carry[0], mode, carry[1], **kwargs)

    return lax.fori_loop(
        1,
        repetitions,
        repeat,
        (state, start_time),
    )[0]


def _expectations(state, observables):
    return jnp.stack([overlap(state, observable) for observable in observables], -1)


def simulate_expectations(
    circuit: Circuit,
    initial_state: Qarray,
    observables: list[Qarray],
    mode: SimulateMode = SimulateMode.DEFAULT,
    include_initial: bool = True,
    **kwargs,
):
    """Return the final state and per-layer expectation values."""
    if not observables:
        raise ValueError("observables must not be empty")

    kwargs.setdefault("saveat_tlist", jnp.array([]))
    state = initial_state
    start_time = 0.0
    values = [_expectations(state, observables)] if include_initial else []
    for layer in circuit.layers:
        output = _simulate_layer(layer, state, mode, start_time, **kwargs)
        state = output["result"][-1]
        start_time = output["start_time"]
        values.append(_expectations(state, observables))
    if not values:
        return state, _expectations(state, observables)[None][:0]
    return state, jnp.stack(values)


def simulate_repeated_expectations(
    circuit: Circuit,
    initial_state: Qarray,
    repetitions: int,
    observables: list[Qarray],
    mode: SimulateMode = SimulateMode.DEFAULT,
    include_initial: bool = True,
    **kwargs,
):
    """Return the final state and per-repetition expectation values."""
    if repetitions < 0:
        raise ValueError("repetitions must be non-negative")
    if not observables:
        raise ValueError("observables must not be empty")
    if repetitions == 0:
        values = _expectations(initial_state, observables)[None]
        return initial_state, values if include_initial else values[:0]

    kwargs.setdefault("saveat_tlist", jnp.array([]))
    state, start_time = _evolve_circuit(
        circuit,
        initial_state,
        mode,
        **kwargs,
    )
    first = _expectations(state, observables)

    def repeat(carry, _):
        state, start_time = _evolve_circuit(
            circuit,
            carry[0],
            mode,
            carry[1],
            **kwargs,
        )
        return (state, start_time), _expectations(state, observables)

    (state, _), rest = lax.scan(
        repeat,
        (state, start_time),
        None,
        length=repetitions - 1,
    )
    values = jnp.concatenate((first[None], rest), axis=0)
    if include_initial:
        values = jnp.concatenate(
            (_expectations(initial_state, observables)[None], values),
            axis=0,
        )
    return state, values


def _simulate_layer(
    layer: Layer,
    initial_state: Qarray,
    mode: SimulateMode = SimulateMode.UNITARY,
    start_time: float = 0,
    **kwargs,
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

        solver_options = kwargs.pop(
            "solver_options",
            SolverOptions.create(progress_meter=False),
        )
        ts = layer.gen_ts()
        ts = ts + start_time
        saveat_times = kwargs.pop("saveat_tlist", ts)
        local_operators = kwargs.pop("local_operators", None)
        has_c_ops = any(len(operation.gate.c_ops) for operation in layer.operations)
        if local_operators is None:
            local_operators = (
                state.is_dm() or has_c_ops or jax.default_backend() == "cpu"
            )

        if local_operators:
            intermediate_states = _solve_local_hamiltonian(
                layer,
                state,
                ts,
                saveat_times,
                solver_options,
            )
        else:
            hamiltonian = layer.gen_Ht()
            collapse_ops = layer.gen_c_ops()
            if state.is_dm() or len(collapse_ops):
                intermediate_states = mesolve(
                    hamiltonian,
                    state,
                    ts,
                    saveat_tlist=saveat_times,
                    c_ops=collapse_ops,
                    solver_options=solver_options,
                )
            else:
                intermediate_states = sesolve(
                    hamiltonian,
                    state,
                    ts,
                    saveat_tlist=saveat_times,
                    solver_options=solver_options,
                )

        result = intermediate_states
        state = intermediate_states[-1]
        start_time = ts[-1]

    elif mode == SimulateMode.KRAUS:
        for operation in layer.operations:
            state = _apply_local_kraus(state, operation)
        result = _single_state_batch(state)

    else:
        raise ValueError(f"Unsupported simulation mode: {mode}")

    return {"result": result, "start_time": start_time}

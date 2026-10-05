"""Compile, runtime, and memory benchmarks for noisy circuit operations."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

import jaxquantum as jqt
from jaxquantum.circuits import (
    Amp_Damp,
    Amp_Gain,
    Circuit,
    Dephasing_Ch_Qb,
    Dephasing_Reset,
    Dephasing_Ch,
    IP_Reset,
    MX,
    MZ,
    Register,
    Reset,
    SimulateMode,
    Thermal_Ch,
    Thermal_Ch_Qb,
    simulate,
    simulate_final,
    simulate_repeated,
)
from jaxquantum.core.qarray import ket2dm


def _measure(name, function, *args, iterations):
    return {
        "name": name,
        **jqt.benchmark_jax_function(
            function,
            *args,
            iterations=iterations,
        ),
    }


def _register(n, spectator):
    return Register.create([2, n] if spectator == 1 else [2, n, spectator])


def _channel_runner(channel_factory, n, max_l, spectator):
    register = _register(n, spectator)

    def run(error_probability, state):
        circuit = Circuit.create(register)
        circuit.append(
            channel_factory(n, error_probability, max_l),
            1,
            default_simulate_mode=SimulateMode.KRAUS,
        )
        return simulate(circuit, state).results[-1].data

    return run


def _promoted_channel_runner(channel_factory, n, max_l, spectator, rounds=1):
    register = _register(n, spectator)

    def run(error_probability, state):
        state = ket2dm(state)
        for _ in range(rounds):
            circuit = Circuit.create(register)
            circuit.append(channel_factory(n, error_probability, max_l), 1)
            kraus = circuit.layers[0].gen_KM()
            state = (kraus @ state @ kraus.dag()).collapse()
        return state.data

    return run


def _repeated_channel_runner(channel_factory, n, max_l, spectator, rounds):
    register = _register(n, spectator)

    def run(error_probability, state):
        circuit = Circuit.create(register)
        for _ in range(rounds):
            circuit.append(
                channel_factory(n, error_probability, max_l),
                1,
                default_simulate_mode=SimulateMode.KRAUS,
            )
        return simulate(circuit, state).results[-1].data

    return run


def _scanned_channel_runner(channel_factory, n, max_l, spectator, rounds):
    register = _register(n, spectator)

    def run(error_probability, state):
        circuit = Circuit.create(register)
        circuit.append(
            channel_factory(n, error_probability, max_l),
            1,
            default_simulate_mode=SimulateMode.KRAUS,
        )
        return simulate_repeated(circuit, state, rounds).data

    return run


def _reset_runner(n, max_l, rounds=1, scanned=False, promoted=False):
    register = Register.create([2, n])

    def run(probability, state):
        if promoted:
            state = ket2dm(state)
            for _ in range(rounds):
                circuit = Circuit.create(register)
                circuit.append(
                    Dephasing_Reset(n, probability, 0.4, 0.7, max_l),
                    [0, 1],
                )
                kraus = circuit.layers[0].gen_KM()
                state = (kraus @ state @ kraus.dag()).collapse()
            return state.data

        circuit = Circuit.create(register)
        repetitions = 1 if scanned else rounds
        for _ in range(repetitions):
            circuit.append(
                Dephasing_Reset(n, probability, 0.4, 0.7, max_l),
                [0, 1],
                default_simulate_mode=SimulateMode.KRAUS,
            )
        if scanned:
            return simulate_repeated(circuit, state, rounds).data
        return simulate(circuit, state).results[-1].data

    return run


def _qubit_runner(factory, n, promoted=False):
    register = Register.create([2, n])

    def run(probability, state):
        circuit = Circuit.create(register)
        circuit.append(
            factory(probability),
            0,
            default_simulate_mode=SimulateMode.KRAUS,
        )
        if not promoted:
            return simulate_final(circuit, state).data
        state = ket2dm(state)
        kraus = circuit.layers[0].gen_KM()
        return (kraus @ state @ kraus.dag()).collapse().data

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--max-loss", type=int, default=20)
    parser.add_argument("--thermal-max-loss", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--spectator-dimension", type=int, default=1)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    n = args.dimension
    register = _register(n, args.spectator_dimension)
    size = jnp.prod(jnp.asarray(register.dims)).item()
    ket = (
        jqt.basis(size, 0).reshape_qdims(*register.dims)
        + 0.4j * jqt.basis(size, size - 1).reshape_qdims(*register.dims)
    ).unit()
    state = ket.to_dm()
    error_probability = jnp.asarray(0.017)

    results = []
    for name, factory, max_l in (
        ("amplitude_damping", Amp_Damp, args.max_loss),
        ("amplitude_gain", Amp_Gain, args.max_loss),
        (
            "thermal",
            lambda dimension, probability, order: Thermal_Ch(
                dimension,
                probability,
                0.03,
                order,
            ),
            args.thermal_max_loss,
        ),
        ("dephasing", Dephasing_Ch, args.max_loss),
    ):
        results.append(
            _measure(
                name,
                _channel_runner(
                    factory,
                    n,
                    max_l,
                    args.spectator_dimension,
                ),
                error_probability,
                state,
                iterations=args.iterations,
            )
        )
        results.append(
            _measure(
                f"{name}_{args.rounds}_rounds",
                _repeated_channel_runner(
                    factory,
                    n,
                    max_l,
                    args.spectator_dimension,
                    args.rounds,
                ),
                error_probability,
                state,
                iterations=args.iterations,
            )
        )
        results.append(
            _measure(
                f"{name}_{args.rounds}_rounds_scanned",
                _scanned_channel_runner(
                    factory,
                    n,
                    max_l,
                    args.spectator_dimension,
                    args.rounds,
                ),
                error_probability,
                state,
                iterations=args.iterations,
            )
        )
        if args.include_reference:
            results.append(
                _measure(
                    f"{name}_promoted_reference",
                    _promoted_channel_runner(
                        factory,
                        n,
                        max_l,
                        args.spectator_dimension,
                    ),
                    error_probability,
                    state,
                    iterations=args.iterations,
                )
            )

    reset_probability = jnp.asarray(0.13)
    reset_state = (
        (
            jqt.basis(2 * n, 0).reshape_qdims(2, n)
            + 0.4j * jqt.basis(2 * n, 2 * n - 1).reshape_qdims(2, n)
        )
        .unit()
        .to_dm()
    )
    for name, runner in (
        ("dephasing_reset", _reset_runner(n, args.max_loss)),
        (
            f"dephasing_reset_{args.rounds}_rounds",
            _reset_runner(n, args.max_loss, args.rounds),
        ),
        (
            f"dephasing_reset_{args.rounds}_rounds_scanned",
            _reset_runner(n, args.max_loss, args.rounds, scanned=True),
        ),
    ):
        results.append(
            _measure(
                name,
                runner,
                reset_probability,
                reset_state,
                iterations=args.iterations,
            )
        )
    if args.include_reference:
        results.append(
            _measure(
                "dephasing_reset_promoted_reference",
                _reset_runner(n, args.max_loss, promoted=True),
                reset_probability,
                reset_state,
                iterations=args.iterations,
            )
        )

    for name, factory in (
        ("measure_z", lambda probability: MZ()),
        ("measure_x", lambda probability: MX()),
        ("qubit_reset", lambda probability: Reset()),
        ("qubit_imperfect_reset", lambda probability: IP_Reset(probability, 0.9)),
        (
            "qubit_thermal",
            lambda probability: Thermal_Ch_Qb(probability, 0.03),
        ),
        ("qubit_dephasing", Dephasing_Ch_Qb),
    ):
        results.append(
            _measure(
                name,
                _qubit_runner(factory, n),
                error_probability,
                reset_state,
                iterations=args.iterations,
            )
        )
        if args.include_reference:
            results.append(
                _measure(
                    f"{name}_promoted_reference",
                    _qubit_runner(factory, n, promoted=True),
                    error_probability,
                    reset_state,
                    iterations=args.iterations,
                )
            )

    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxquantum": jqt.__version__,
        "jaxquantum_source": str(Path(jqt.__file__).resolve()),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "parameters": {
            "dimension": n,
            "max_loss": args.max_loss,
            "thermal_max_loss": args.thermal_max_loss,
            "rounds": args.rounds,
            "spectator_dimension": args.spectator_dimension,
        },
        "results": results,
    }
    output = json.dumps(payload, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(output + "\n")
    print(output)


if __name__ == "__main__":
    main()

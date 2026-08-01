"""Benchmarks for local circuit Hamiltonian evolution."""

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
from jaxquantum.circuits import Circuit, D, Register, SimulateMode, simulate_final


def _runner(dimension, spectator, times, collapse):
    register = Register.create([2, dimension, spectator])
    collapse_ops = (
        jqt.Qarray.from_list([0.03 * jqt.destroy(dimension)]) if collapse else None
    )
    options = jqt.SolverOptions.create(
        progress_meter=False,
        stepsize_controller="ConstantStepSize",
    )

    def run(amplitude, state):
        circuit = Circuit.create(register)
        circuit.append(
            D(dimension, amplitude, ts=times, c_ops=collapse_ops),
            1,
            default_simulate_mode=SimulateMode.HAMILTONIAN,
        )
        return simulate_final(
            circuit,
            state,
            solver_options=options,
        ).data

    return run


def _promoted_runner(dimension, spectator, times, collapse):
    register = Register.create([2, dimension, spectator])
    collapse_ops = (
        jqt.Qarray.from_list([0.03 * jqt.destroy(dimension)]) if collapse else None
    )
    options = jqt.SolverOptions.create(
        progress_meter=False,
        stepsize_controller="ConstantStepSize",
    )

    def run(amplitude, state):
        circuit = Circuit.create(register)
        circuit.append(
            D(dimension, amplitude, ts=times, c_ops=collapse_ops),
            1,
            default_simulate_mode=SimulateMode.HAMILTONIAN,
        )
        layer = circuit.layers[0]
        if collapse:
            return jqt.mesolve(
                layer.gen_Ht(),
                state,
                times,
                saveat_tlist=jnp.array([]),
                c_ops=layer.gen_c_ops(),
                solver_options=options,
            ).data
        return jqt.sesolve(
            layer.gen_Ht(),
            state,
            times,
            saveat_tlist=jnp.array([]),
            solver_options=options,
        ).data

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--spectator-dimension", type=int, default=3)
    parser.add_argument("--steps", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    register = Register.create([2, args.dimension, args.spectator_dimension])
    size = 2 * args.dimension * args.spectator_dimension
    ket = (
        jqt.basis(size, 0).reshape_qdims(*register.dims)
        + 0.2j * jqt.basis(size, size - 1).reshape_qdims(*register.dims)
    ).unit()
    times = jnp.linspace(0.0, 1.0, args.steps)
    amplitude = jnp.asarray(0.13 + 0.07j)

    reports = {}
    for name, collapse, state in (
        ("schrodinger", False, ket),
        ("lindblad", True, ket.to_dm()),
    ):
        reports[name] = jqt.benchmark_jax_function(
            _runner(
                args.dimension,
                args.spectator_dimension,
                times,
                collapse,
            ),
            amplitude,
            state,
            iterations=args.iterations,
        )
        if args.include_reference:
            reports[f"{name}_promoted_reference"] = jqt.benchmark_jax_function(
                _promoted_runner(
                    args.dimension,
                    args.spectator_dimension,
                    times,
                    collapse,
                ),
                amplitude,
                state,
                iterations=args.iterations,
            )

    payload = {
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxquantum": jqt.__version__,
        "jaxquantum_source": str(Path(jqt.__file__).resolve()),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "parameters": {
            "dimension": args.dimension,
            "spectator_dimension": args.spectator_dimension,
            "steps": args.steps,
        },
        "results": reports,
    }
    output = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

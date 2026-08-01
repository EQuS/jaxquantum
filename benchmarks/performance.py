"""Reproducible compile and warmed-up benchmarks for jaxquantum."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

import jaxquantum as jqt
from jaxquantum.circuits import Circuit, Register, Rx, simulate
from jaxquantum.devices import Transmon


def _ready(value) -> None:
    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _measure(name, function, *args, iterations):
    jax.clear_caches()
    lowered = jax.jit(function).lower(*args)

    start = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - start

    _ready(compiled(*args))
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ready(compiled(*args))
        samples.append(time.perf_counter() - start)

    return {
        "name": name,
        "compile_s": compile_s,
        "warm_median_s": statistics.median(samples),
        "warm_min_s": min(samples),
        "hlo_chars": len(lowered.as_text()),
    }


def _dense_chain(a, b, ket, scale):
    value = a
    for _ in range(6):
        value = scale * (value @ b) + (1.0 - scale) * a
    return value @ ket


def _matvec_chain(operator, ket):
    for _ in range(16):
        ket = operator @ ket
    return ket


def _transmon_hamiltonian(flux):
    device = Transmon.create(
        N=8,
        N_pre_diag=31,
        params={"Ec": 0.22, "Ej": 18.0, "ng": flux},
    )
    return device.get_H().data


def _transmon_operators(flux):
    device = Transmon.create(
        N=6,
        N_pre_diag=25,
        params={"Ec": 0.22, "Ej": 18.0, "ng": flux},
    )
    return tuple(operator.data for operator in device.ops.values())


def _circuit_runner(n_qubits, depth):
    register = Register.create([2] * n_qubits)

    def run(angles, state):
        circuit = Circuit.create(register)
        for layer in range(depth):
            circuit.append(Rx(angles[layer]), layer % n_qubits)
        return simulate(circuit, state).results[-1].data

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--circuit-qubits", type=int, default=8)
    parser.add_argument("--circuit-depth", type=int, default=20)
    parser.add_argument("--output")
    args = parser.parse_args()

    key_a, key_b, key_v = jax.random.split(jax.random.key(17), 3)
    n = 64
    a = jqt.Qarray.create(jax.random.normal(key_a, (n, n), dtype=jnp.float64))
    b = jqt.Qarray.create(jax.random.normal(key_b, (n, n), dtype=jnp.float64))
    ket = jqt.Qarray.create(
        jax.random.normal(key_v, (n,), dtype=jnp.float64),
        qtype="ket",
    )

    n_vec = 256
    diagonal = jnp.exp(1j * jnp.linspace(0.0, 0.4, n_vec))
    operator = jqt.Qarray.create(jnp.diag(diagonal))
    vector = jqt.Qarray.create(
        jnp.ones(n_vec, dtype=jnp.complex128) / jnp.sqrt(n_vec),
        qtype="ket",
    )

    circuit_runner = _circuit_runner(args.circuit_qubits, args.circuit_depth)
    angles = jnp.linspace(0.1, 1.1, args.circuit_depth)
    initial = jqt.basis(2**args.circuit_qubits, 0).reshape_qdims(
        *([2] * args.circuit_qubits)
    )

    results = [
        _measure(
            "qarray_dense_chain",
            _dense_chain,
            a,
            b,
            ket,
            jnp.asarray(0.37),
            iterations=args.iterations,
        ),
        _measure(
            "qarray_matvec_chain",
            _matvec_chain,
            operator,
            vector,
            iterations=args.iterations,
        ),
        _measure(
            "transmon_hamiltonian",
            _transmon_hamiltonian,
            jnp.asarray(0.13),
            iterations=args.iterations,
        ),
        _measure(
            "transmon_operators",
            _transmon_operators,
            jnp.asarray(0.13),
            iterations=args.iterations,
        ),
        _measure(
            "unitary_circuit",
            circuit_runner,
            angles,
            initial,
            iterations=args.iterations,
        ),
    ]
    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxquantum": jqt.__version__,
        "jaxquantum_source": str(Path(jqt.__file__).resolve()),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "results": results,
    }
    output = json.dumps(payload, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(output + "\n")
    print(output)


if __name__ == "__main__":
    main()

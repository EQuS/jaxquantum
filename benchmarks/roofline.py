"""Measure empirical JAX memory and compute roofs on the active device."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time

import jax
import jax.numpy as jnp

import jaxquantum as jqt
from jaxquantum.circuits import Circuit, Register, Rx, simulate

jax.config.update("jax_enable_x64", True)


def _ready(value):
    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _measure(function, *args, iterations):
    compiled = jax.jit(function).lower(*args).compile()
    _ready(compiled(*args))
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ready(compiled(*args))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), min(samples)


def _stream(size, iterations):
    key_x, key_y = jax.random.split(jax.random.key(1))
    x = jax.random.normal(key_x, (size,), dtype=jnp.float64)
    y = jax.random.normal(key_y, (size,), dtype=jnp.float64)
    median_s, min_s = _measure(lambda a, b: 1.5 * a + b, x, y, iterations=iterations)
    bytes_moved = 3 * size * x.dtype.itemsize
    result = {
        "name": "stream_float64",
        "arithmetic_intensity_flop_per_byte": 2 * size / bytes_moved,
        "median_s": median_s,
        "min_s": min_s,
        "median_gb_s": bytes_moved / median_s / 1e9,
        "peak_gb_s": bytes_moved / min_s / 1e9,
    }
    del x, y
    gc.collect()
    jax.clear_caches()
    return result


def _gemm(name, size, dtype, iterations):
    a = jnp.ones((size, size), dtype=dtype)
    b = jnp.ones((size, size), dtype=dtype)
    median_s, min_s = _measure(lambda x, y: x @ y, a, b, iterations=iterations)
    complex_factor = 4 if jnp.issubdtype(dtype, jnp.complexfloating) else 1
    flops = 2 * complex_factor * size**3
    bytes_moved = 3 * size**2 * a.dtype.itemsize
    result = {
        "name": name,
        "size": size,
        "arithmetic_intensity_flop_per_byte": flops / bytes_moved,
        "median_s": median_s,
        "min_s": min_s,
        "median_tflop_s": flops / median_s / 1e12,
        "peak_tflop_s": flops / min_s / 1e12,
    }
    del a, b
    gc.collect()
    jax.clear_caches()
    return result


def _qarray_matvec(size, iterations):
    key_operator, key_ket = jax.random.split(jax.random.key(2))
    operator = jqt.Qarray.create(
        jax.random.normal(
            key_operator,
            (size, size),
            dtype=jnp.complex128,
        )
    )
    ket = jqt.Qarray.create(
        jax.random.normal(key_ket, (size,), dtype=jnp.complex128),
        qtype="ket",
    )
    median_s, min_s = _measure(
        lambda a, b: a @ b,
        operator,
        ket,
        iterations=iterations,
    )
    flops = 8 * size**2
    bytes_moved = (size**2 + 2 * size) * operator.dtype.itemsize
    result = {
        "name": "qarray_matvec_complex128",
        "size": size,
        "arithmetic_intensity_flop_per_byte": flops / bytes_moved,
        "median_s": median_s,
        "min_s": min_s,
        "median_gflop_s": flops / median_s / 1e9,
        "median_gb_s": bytes_moved / median_s / 1e9,
    }
    del operator, ket
    gc.collect()
    jax.clear_caches()
    return result


def _local_circuit(n_qubits, depth, iterations):
    register = Register.create([2] * n_qubits)
    initial = jqt.basis(2**n_qubits, 0).reshape_qdims(*register.dims)
    angles = jnp.linspace(0.1, 1.1, depth)

    def run(values, state):
        circuit = Circuit.create(register)
        for layer in range(depth):
            circuit.append(Rx(values[layer]), layer % n_qubits)
        return simulate(circuit, state).results[-1].data

    median_s, min_s = _measure(run, angles, initial, iterations=iterations)
    amplitudes = 2**n_qubits
    flops = 16 * amplitudes * depth
    bytes_moved = 2 * amplitudes * initial.dtype.itemsize * depth
    return {
        "name": "local_circuit_complex128",
        "qubits": n_qubits,
        "depth": depth,
        "arithmetic_intensity_flop_per_byte": flops / bytes_moved,
        "median_s": median_s,
        "min_s": min_s,
        "median_gflop_s": flops / median_s / 1e9,
        "median_gb_s": bytes_moved / median_s / 1e9,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("all", "roofs", "applications"),
        default="all",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--stream-size", type=int, default=2**25)
    parser.add_argument("--matvec-size", type=int, default=6144)
    parser.add_argument("--circuit-qubits", type=int, default=20)
    parser.add_argument("--circuit-depth", type=int, default=64)
    parser.add_argument("--output")
    args = parser.parse_args()

    jnp.zeros(1).block_until_ready()
    results = []
    if args.suite in ("all", "roofs"):
        results.extend(
            [
                _stream(args.stream_size, args.iterations),
                _gemm("gemm_float32", 8192, jnp.float32, args.iterations),
                _gemm("gemm_float64", 4096, jnp.float64, args.iterations),
                _gemm("gemm_complex64", 8192, jnp.complex64, args.iterations),
                _gemm("gemm_complex128", 4096, jnp.complex128, args.iterations),
            ]
        )
    if args.suite in ("all", "applications"):
        results.extend(
            [
                _qarray_matvec(args.matvec_size, args.iterations),
                _local_circuit(
                    args.circuit_qubits,
                    args.circuit_depth,
                    args.iterations,
                ),
            ]
        )
    payload = {
        "jax": jax.__version__,
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

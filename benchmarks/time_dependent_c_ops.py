"""Compare static and callable Lindblad collapse operators on identical dynamics.

Run from the repository root:
    python benchmarks/time_dependent_c_ops.py --iterations 20

Cold time includes JIT compilation and first execution. Warm timings explicitly
synchronize JAX execution; do not interpret one run as a stable speed claim.
"""
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
import numpy as np

import jaxquantum as jqt


def _profile(label, fn, iterations):
    compiled = jax.jit(fn)

    start = time.perf_counter()
    first = compiled()
    jax.block_until_ready(first)
    cold_ms = 1e3 * (time.perf_counter() - start)

    samples_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = compiled()
        jax.block_until_ready(result)
        samples_ms.append(1e3 * (time.perf_counter() - start))

    return {
        "name": label,
        "cold_compile_and_execute_ms": round(cold_ms, 3),
        "warm_median_ms": round(statistics.median(samples_ms), 4),
        "warm_min_ms": round(min(samples_ms), 4),
        "warm_max_ms": round(max(samples_ms), 4),
        "result": np.asarray(first),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    times = jnp.linspace(0.0, 8.0, 161)
    rho0 = jqt.basis(2, 1).to_dm()
    lowering = jqt.destroy(2)
    static_stack = jqt.Qarray.from_list([jnp.sqrt(0.25) * lowering])
    options = jqt.SolverOptions(progress_meter=None)

    def run_static():
        return jqt.mesolve(
            0.0, rho0, times, c_ops=static_stack, solver_options=options
        ).data

    def run_constant_callable():
        return jqt.mesolve(
            0.0, rho0, times, c_ops=lambda t: static_stack,
            solver_options=options,
        ).data

    def run_varying_callable():
        def c_ops(t):
            rate = 0.15 + 0.20 * jnp.exp(-t / 1.2)
            return jqt.Qarray.from_list([jnp.sqrt(rate) * lowering])

        return jqt.mesolve(
            0.0, rho0, times, c_ops=c_ops, solver_options=options
        ).data

    results = [
        _profile("static", run_static, args.iterations),
        _profile("constant_callable", run_constant_callable, args.iterations),
        _profile("varying_callable", run_varying_callable, args.iterations),
    ]
    np.testing.assert_allclose(
        results[0]["result"],
        results[1]["result"],
        rtol=2e-5,
        atol=2e-5,
    )
    for entry in results:
        entry.pop("result")

    warm_static = results[0]["warm_median_ms"]
    warm_callable = results[1]["warm_median_ms"]
    output = {
        "python": platform.python_version(),
        "system": platform.platform(),
        "jax": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "iterations": args.iterations,
        "results": results,
        "constant_callable_over_static_warm_ratio": round(
            warm_callable / warm_static, 3
        ),
        "notes": (
            "The first two variants solve identical dynamics. The varying "
            "variant solves a different equation and is not an apples-to-apples "
            "performance comparison."
        ),
    }
    report = json.dumps(output, indent=2)
    print(report)
    if args.output:
        Path(args.output).write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Benchmarks for bosonic gate construction."""

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
from jax.scipy.special import gammaln

import jaxquantum as jqt
from jaxquantum.circuits import Amp_Damp, CD, ECD, Thermal_Ch


def _reference(dimension, echoed):
    g, e = jqt.basis(2, 0), jqt.basis(2, 1)
    left = e @ g.dag() if echoed else g @ g.dag()
    right = g @ e.dag() if echoed else e @ e.dag()

    def build(beta):
        return (
            (left ^ jqt.displace(dimension, beta / 2))
            + (right ^ jqt.displace(dimension, -beta / 2))
        ).data

    return build


def _optimized(dimension, echoed):
    constructor = ECD if echoed else CD
    return lambda beta: constructor(dimension, beta).U.data


def _gate_setup(dimension, max_l, probability, iterations):
    def measure(materialize):
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            gate = Amp_Damp(dimension, probability, max_l)
            leaves = jax.tree.leaves(gate.params)
            if materialize:
                leaves.extend(jax.tree.leaves(gate.KM.data))
            jqt.block_until_ready(leaves)
            samples.append(time.perf_counter() - start)
        return {
            "median_seconds": statistics.median(samples),
            "min_seconds": min(samples),
            "max_seconds": max(samples),
        }

    measure(False)
    measure(True)
    return {
        "lazy": measure(False),
        "eager_reference": measure(True),
    }


def _thermal_reference(dimension, max_l):
    a = jqt.destroy(dimension).data
    adag = jnp.conj(a.T)
    powers = jnp.arange(max_l + 1)

    def build(probability, n_bar):
        a_powers = jnp.stack([jnp.linalg.matrix_power(a, i) for i in range(max_l + 1)])
        adag_powers = jnp.stack(
            [jnp.linalg.matrix_power(adag, i) for i in range(max_l + 1)]
        )
        middle = jnp.diag(jnp.power(jnp.sqrt(1 - probability), jnp.arange(dimension)))
        gain, loss = jnp.meshgrid(powers, powers, indexing="ij")
        gain, loss = gain.ravel(), loss.ravel()
        prefactor = jnp.sqrt(
            jnp.power(probability * (1 + n_bar), loss)
            * jnp.power(probability * n_bar, gain)
            / (jnp.exp(gammaln(loss + 1)) * jnp.exp(gammaln(gain + 1)))
        )
        return prefactor[:, None, None] * (middle @ a_powers[loss] @ adag_powers[gain])

    return build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--max-loss", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()
    beta = jnp.asarray(0.17 + 0.09j)

    reports = {}
    for name, echoed in (("cd", False), ("ecd", True)):
        reports[f"{name}_reference"] = jqt.benchmark_jax_function(
            _reference(args.dimension, echoed),
            beta,
            iterations=args.iterations,
        )
        reports[name] = jqt.benchmark_jax_function(
            _optimized(args.dimension, echoed),
            beta,
            iterations=args.iterations,
        )

    reports["avoided_amp_damp_kraus"] = jqt.benchmark_jax_function(
        lambda probability: Amp_Damp(
            args.dimension,
            probability,
            args.max_loss,
        ).KM.data,
        jnp.asarray(0.04),
        iterations=args.iterations,
    )
    reports["thermal_kraus_reference"] = jqt.benchmark_jax_function(
        _thermal_reference(args.dimension, args.max_loss),
        jnp.asarray(0.04),
        jnp.asarray(0.1),
        iterations=args.iterations,
    )
    reports["thermal_kraus"] = jqt.benchmark_jax_function(
        lambda probability, n_bar: Thermal_Ch(
            args.dimension,
            probability,
            n_bar,
            args.max_loss,
        ).KM.data,
        jnp.asarray(0.04),
        jnp.asarray(0.1),
        iterations=args.iterations,
    )
    reports["amp_damp_gate_setup"] = _gate_setup(
        args.dimension,
        args.max_loss,
        jnp.asarray(0.04),
        args.iterations,
    )

    payload = {
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxquantum": jqt.__version__,
        "jaxquantum_source": str(Path(jqt.__file__).resolve()),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "parameters": vars(args) | {"output": None},
        "results": reports,
    }
    output = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

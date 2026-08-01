"""Benchmarks for device Hamiltonian construction."""

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
from jaxquantum.devices import KNO, Transmon
from jaxquantum.devices.analysis import run_jax_sweep


def _transmon(offset, repeated):
    device = Transmon.create(
        N=6,
        N_pre_diag=21,
        params={"Ec": 0.22, "Ej": 18.0, "ng": offset},
    )
    if not repeated:
        return device.get_H_full().data
    return (
        device.original_ops["H_charge"] - device.Ej * device.original_ops["cos(φ)"]
    ).data


def _kno(anharmonicity, repeated):
    device = KNO.create(
        N=32,
        params={"f": 5.0, "α": anharmonicity},
    )
    if not repeated:
        return device.get_H_full().data
    linear = (
        device.get_linear_frequency()
        * device.linear_ops["a_dag"]
        @ device.linear_ops["a"]
    )
    nonlinear = (
        anharmonicity
        / 2
        * device.linear_ops["a_dag"]
        @ device.linear_ops["a_dag"]
        @ device.linear_ops["a"]
        @ device.linear_ops["a"]
    )
    return (linear + nonlinear).data


def _wavefunctions(phases, repeated):
    device = Transmon.create(
        N=6,
        N_pre_diag=13,
        params={"Ec": 0.22, "Ej": 18.0, "ng": 0.13},
    )
    if not repeated:
        return device.calculate_wavefunctions(phases)
    n_labels = jnp.diag(device.original_ops["n"].data)
    return jnp.stack(
        [
            jnp.stack(
                [
                    (1j**level / jnp.sqrt(2 * jnp.pi))
                    * jnp.sum(
                        device.eig_systems["vecs"][:, level]
                        * jnp.exp(1j * phase * n_labels)
                    )
                    for phase in phases
                ]
            )
            for level in range(device.N_pre_diag)
        ]
    )


def _transmon_sweep(offsets, vectorized):
    def metric(params):
        return (
            Transmon.create(
                N=6,
                N_pre_diag=13,
                params={"Ec": 0.22, "Ej": 18.0, "ng": params["ng"]},
            )
            .get_H()
            .data
        )

    if vectorized:
        return run_jax_sweep({}, {"ng": offsets}, metric)
    return jnp.stack([metric({"ng": offset}) for offset in offsets])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--wave-points", type=int, default=21)
    parser.add_argument("--sweep-points", type=int, default=17)
    parser.add_argument("--output")
    args = parser.parse_args()
    inputs = {"transmon": jnp.asarray(0.13), "kno": jnp.asarray(-0.2)}
    functions = {"transmon": _transmon, "kno": _kno}
    reports = {}
    for name, value in inputs.items():
        for suffix, repeated in (("reference", True), ("hoisted", False)):
            reports[f"{name}_{suffix}"] = jqt.benchmark_jax_function(
                lambda parameter, fn=functions[name], old=repeated: fn(
                    parameter,
                    old,
                ),
                value,
                iterations=args.iterations,
            )
    phases = jnp.linspace(-0.4, 0.4, args.wave_points)
    for suffix, repeated in (("reference", True), ("vectorized", False)):
        reports[f"wavefunctions_{suffix}"] = jqt.benchmark_jax_function(
            lambda values, old=repeated: _wavefunctions(values, old),
            phases,
            iterations=args.iterations,
        )
    offsets = jnp.linspace(-0.4, 0.4, args.sweep_points)
    for suffix, vectorized in (("unrolled", False), ("vmapped", True)):
        reports[f"sweep_{suffix}"] = jqt.benchmark_jax_function(
            lambda values, use_vmap=vectorized: _transmon_sweep(
                values,
                use_vmap,
            ),
            offsets,
            iterations=args.iterations,
        )

    payload = {
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxquantum": jqt.__version__,
        "jaxquantum_source": str(Path(jqt.__file__).resolve()),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "results": reports,
    }
    output = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

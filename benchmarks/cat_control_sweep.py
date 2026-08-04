"""Sweep cat-sBs control amplitudes with the shared device model."""

from __future__ import annotations

import argparse
from itertools import product
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    os.environ.get("JAXQUANTUM_ROOT", str(Path(__file__).resolve().parents[1])),
)

from experiments.circuit import sbs_device as model


def _values(text):
    return [float(value) for value in text.split(",")]


def _load_module(path):
    spec = importlib.util.spec_from_file_location("cat_sbs_control_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cat_sbs_results(args, points, device):
    if args.cat_sbs_source is None:
        return [None] * len(points)
    source = _load_module(args.cat_sbs_source.resolve())
    params = source.device_parameters()
    params["resonator"].update(
        T1=device.storage_t1,
        Tphi=device.storage_tphi,
    )
    params["qubit"].update(
        T1=device.qubit_t1,
        T2echo=1 / (1 / device.qubit_tphi + 1 / (2 * device.qubit_t1)),
    )
    params["timing"].update(
        small_cd=(device.cd_durations[0], device.cd_durations[2]),
        big_cd=device.cd_durations[1],
        identity_cd=device.extra_storage_duration,
        qubit_pulse=device.rotation_durations[0],
        reset=device.reset_duration,
    )
    return [
        source.simulate_corrected_device_bit_decay(
            args.nbar,
            cycles=args.cycles,
            N=args.dimension,
            params=params,
            delta=delta,
            ratio=ratio,
            small_scale=1.0,
            big_scale=1.0,
            unconditional_fraction=0.0,
            alternate_cd_direction=args.alternate_cd_direction,
        )
        for delta, ratio in points
    ]


def _colleague_results(args, points, device):
    if args.colleague_root is None:
        return [None] * len(points)
    source = args.colleague_root / "project-strobo-cat" / "stroboscopic-cats-mh" / "src"
    sys.path.insert(0, str(source))
    from stroboscopic_cats import gate_simulator, physics

    timing = gate_simulator.DeviceTiming(
        small_cd_1=device.cd_durations[0],
        big_cd=device.cd_durations[1],
        small_cd_2=device.cd_durations[2],
        identity_cd=device.extra_storage_duration,
        qubit_pulse=device.rotation_durations[0],
        number_of_qubit_pulses=len(device.rotation_durations),
        reset=device.reset_duration,
        round_time=model.round_time(device),
    )
    rates = gate_simulator.simulate_device_gate_rates(
        physics.cat_point(args.nbar, 0.0),
        [delta for delta, _ in points],
        [ratio for _, ratio in points],
        timing=timing,
        n_fock=args.dimension,
        storage_t1=device.storage_t1,
        storage_tphi=device.storage_tphi,
        auxiliary_t1=device.qubit_t1,
        auxiliary_tphi=device.qubit_tphi,
        microsteps=args.microsteps,
        burn_in=args.burn_in,
        candidate_blocks=tuple(args.candidate_blocks),
        max_losses=args.max_loss,
    )
    return [
        {
            "lifetime_ms": 1e3 / rate,
            "rate_per_s": rate,
            "upper_bound": bool(upper),
            "block_size": int(block),
            "stability": stability,
            "trace_error": trace,
            "minimum_eigenvalue": eigenvalue,
        }
        for rate, upper, block, stability, trace, eigenvalue in zip(
            rates.rates,
            rates.upper_bound,
            rates.block_size,
            rates.stability,
            rates.trace_errors,
            rates.positivity_min_eigenvalue,
        )
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratios", type=_values, default=[3.125])
    parser.add_argument("--deltas", type=_values, default=[0.6])
    parser.add_argument("--nbar", type=float, default=4.0)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--cycles", type=int, default=480)
    parser.add_argument("--fit-start", type=int, default=4)
    parser.add_argument("--fit-floor", type=float, default=1e-10)
    parser.add_argument("--max-loss", type=int, default=8)
    parser.add_argument("--microsteps", type=int, default=2)
    parser.add_argument("--burn-in", type=int, default=20)
    parser.add_argument(
        "--candidate-blocks",
        type=lambda text: [int(value) for value in text.split(",")],
        default=[10, 100, 1000],
    )
    parser.add_argument("--colleague-root", type=Path)
    parser.add_argument("--cat-sbs-source", type=Path)
    parser.add_argument("--alternate-cd-direction", action="store_true")
    parser.add_argument(
        "--parameters",
        choices=("nominal", "measured"),
        default="nominal",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = {
        "nominal": model.CAT_DEVICE,
        "measured": model.CAT_MEASURED_DEVICE,
    }[args.parameters]
    initial, observables = model.cat_problem(args.dimension, args.nbar)
    cycle_time = model.round_time(device)
    points = list(product(args.deltas, args.ratios))
    protocols = [
        model.cat_protocol(
            args.dimension,
            args.nbar,
            delta=delta,
            ratio=ratio,
            microsteps=args.microsteps,
            max_loss=args.max_loss,
            device=device,
            alternate_cd_direction=args.alternate_cd_direction,
        )
        for delta, ratio in points
    ]
    shared = model.simulate_decay_variants(
        initial,
        observables,
        protocols,
        args.cycles,
        cycle_time,
        fit_start=args.fit_start,
        fit_floor=args.fit_floor,
    )
    colleague = _colleague_results(args, points, device)
    cat_sbs = _cat_sbs_results(args, points, device)
    results = []
    for (delta, ratio), decay, reference, legacy in zip(
        points,
        shared,
        colleague,
        cat_sbs,
    ):
        result = {
            "delta": delta,
            "ratio": ratio,
            "shared": {
                "lifetime_ms": decay.lifetime * 1e3,
                "rate_per_s": decay.rate,
                "r2": decay.r2,
                "trace_error": decay.trace_error,
                "minimum_eigenvalue": decay.minimum_eigenvalue,
            },
        }
        if reference is not None:
            result["colleague"] = reference
            result["relative_lifetime_difference"] = (
                decay.lifetime * 1e3 / reference["lifetime_ms"] - 1
            )
        if legacy is not None:
            result["cat_sbs"] = {
                "lifetime_ms": legacy["z_cf_lifetime_us"] / 1e3,
                "r2": legacy["z_cf_r2"],
                "trace_error": legacy["trace_error"],
            }
        results.append(result)
    report = {"configuration": vars(args), "results": results}
    text = json.dumps(
        report,
        indent=2,
        default=lambda value: (
            value.item() if isinstance(value, np.generic) else str(value)
        ),
    )
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()

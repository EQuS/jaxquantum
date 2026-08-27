"""Profile the shared cat/GKP sBs device simulation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(
    0,
    os.environ.get("JAXQUANTUM_ROOT", str(Path(__file__).resolve().parents[1])),
)

import jax
import numpy as np

import jaxquantum as jqt
import jaxquantum.circuits as jqtc
from jaxquantum.circuits.library import sbs as model


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _decay_summary(result):
    output = asdict(result)
    output.pop("contrast")
    output["end_contrast_fraction"] = float(
        abs(result.contrast[-1] / result.contrast[0])
    )
    return output


def _problem_and_protocol(args, enabled=model.ERROR_CHANNELS):
    if args.code == "gkp":
        device, delta, small_ratio = _gkp_parameters(args)
        initial, observables = model.gkp_problem(args.dimension, delta)
        half_rounds = model.gkp_protocol(
            args.dimension,
            delta=delta,
            small_ratio=small_ratio,
            device=device,
            enabled=enabled,
            microsteps=args.microsteps,
            jump_samples=args.jump_samples,
            max_loss=args.max_loss,
            max_reset=args.max_reset,
        )
        cycle_time = model.round_time(device, 2)
    else:
        delta = 0.6 if args.delta is None else args.delta
        initial, observables = model.cat_problem(
            args.dimension,
            args.nbar,
            args.kind,
        )
        half_rounds = model.cat_protocol(
            args.dimension,
            args.nbar,
            delta=delta,
            ratio=args.cat_ratio,
            enabled=enabled,
            microsteps=args.microsteps,
            jump_samples=args.jump_samples,
            max_loss=args.max_loss,
        )
        cycle_time = model.round_time(model.CAT_DEVICE)
    return initial, observables, half_rounds, cycle_time


def _prepared_protocol(args):
    if args.code == "gkp":
        device, delta, small_ratio = _gkp_parameters(args)
        return model.prepare_gkp_protocol(
            args.dimension,
            delta=delta,
            small_ratio=small_ratio,
            device=device,
            microsteps=args.microsteps,
            jump_samples=args.jump_samples,
            max_loss=args.max_loss,
            max_reset=args.max_reset,
        )
    return model.prepare_cat_protocol(
        args.dimension,
        args.nbar,
        delta=0.6 if args.delta is None else args.delta,
        ratio=args.cat_ratio,
        microsteps=args.microsteps,
        jump_samples=args.jump_samples,
        max_loss=args.max_loss,
    )


def _gkp_parameters(args):
    legacy = args.parameters == "legacy"
    device = model.GKP_LEGACY_DEVICE if legacy else model.GKP_DEVICE
    delta = args.delta if args.delta is not None else (0.42 if legacy else 0.428)
    ratio = (
        args.small_ratio if args.small_ratio is not None else (1.3 if legacy else 1.083)
    )
    return device, delta, ratio


def run(args):
    start = time.perf_counter()
    initial, observables, half_rounds, cycle_time = _problem_and_protocol(args)
    build_s = time.perf_counter() - start

    def simulate(states, measured, rounds):
        return jqtc.simulate_sbs(states, measured, rounds, args.cycles)

    report = {
        "platform": platform.platform(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "configuration": vars(args),
        "round_time_s": cycle_time,
        "build_s": build_s,
    }
    if not args.only_budget:
        report["profile"] = jqt.benchmark_jax_function(
            simulate,
            initial,
            observables,
            half_rounds,
            iterations=args.iterations,
            warmup=args.warmup,
            compare_precision=args.compare_precision,
        )
        report["decay"] = _decay_summary(
            model.simulate_decay(
                initial,
                observables,
                half_rounds,
                args.cycles,
                cycle_time,
                fit_start=args.fit_start,
                fit_floor=args.fit_floor,
            )
        )

    if args.budget or args.only_budget:
        start = time.perf_counter()
        protocol = _prepared_protocol(args)
        budget = model.compute_error_budget(
            protocol,
            initial,
            observables,
            args.cycles,
            cycle_time,
            fit_start=args.fit_start,
            fit_floor=args.fit_floor,
        )
        report["budget"] = budget.summary()
        report["budget_wall_s"] = time.perf_counter() - start
    return _jsonable(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", choices=("gkp", "cat"), default="gkp")
    parser.add_argument("--dimension", type=int, default=60)
    parser.add_argument("--cycles", type=int, default=512)
    parser.add_argument("--fit-start", type=int, default=4)
    parser.add_argument("--fit-floor", type=float, default=1e-10)
    parser.add_argument("--microsteps", type=int, default=1)
    parser.add_argument("--jump-samples", type=int, default=4)
    parser.add_argument("--max-loss", type=int, default=8)
    parser.add_argument("--max-reset", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--small-ratio", type=float)
    parser.add_argument(
        "--parameters",
        choices=("current", "legacy"),
        default="current",
    )
    parser.add_argument("--nbar", type=float, default=2.0)
    parser.add_argument("--cat-ratio", type=float, default=3.125)
    parser.add_argument("--kind", choices=("bit", "phase"), default="bit")
    parser.add_argument("--compare-precision", action="store_true")
    parser.add_argument("--budget", action="store_true")
    parser.add_argument("--only-budget", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args)
    text = json.dumps(report, indent=2, allow_nan=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()

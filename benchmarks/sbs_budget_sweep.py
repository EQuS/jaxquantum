"""Sweep shared cat-sBs error budgets over cat size."""

from __future__ import annotations

import argparse
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

from experiments.circuit import sbs_device as model


def _budget_result(budget):
    excess = budget.all_on.rate - budget.baseline.rate
    contributions = {
        **budget.isolated_increments,
        "interaction": budget.interaction_rate,
    }
    return {
        **budget.summary(),
        "baseline_lifetime_s": budget.baseline.lifetime,
        "all_on_lifetime_s": budget.all_on.lifetime,
        "all_on_r2": budget.all_on.r2,
        "end_contrast_fraction": float(
            abs(budget.all_on.contrast[-1] / budget.all_on.contrast[0])
        ),
        "trace_error": budget.all_on.trace_error,
        "minimum_eigenvalue": budget.all_on.minimum_eigenvalue,
        "additive_fraction": {
            name: value / excess for name, value in contributions.items()
        },
    }


def run(args):
    report = {
        "platform": platform.platform(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "configuration": {
            **vars(args),
            "nbars": args.nbars,
        },
        "round_time_s": model.round_time(model.CAT_DEVICE),
        "results": [],
    }
    for nbar in args.nbars:
        start = time.perf_counter()
        initial, observables = model.cat_problem(args.dimension, nbar, args.kind)
        protocol = model.prepare_cat_protocol(
            args.dimension,
            nbar,
            delta=args.delta,
            ratio=args.ratio,
            microsteps=args.microsteps,
            jump_samples=args.jump_samples,
            max_loss=args.max_loss,
        )
        budget = model.compute_error_budget(
            protocol,
            initial,
            observables,
            args.cycles,
            report["round_time_s"],
            fit_start=args.fit_start,
            fit_floor=args.fit_floor,
        )
        report["results"].append(
            {
                "nbar": nbar,
                "wall_s": time.perf_counter() - start,
                **_budget_result(budget),
            }
        )
        if args.output:
            _write(args.output, report)
    return report


def _write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=True, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nbars",
        type=lambda text: [float(value) for value in text.split(",")],
        default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    )
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--cycles", type=int, default=480)
    parser.add_argument("--fit-start", type=int, default=4)
    parser.add_argument("--fit-floor", type=float, default=1e-10)
    parser.add_argument("--microsteps", type=int, default=1)
    parser.add_argument("--jump-samples", type=int, default=4)
    parser.add_argument("--max-loss", type=int, default=8)
    parser.add_argument("--delta", type=float, default=0.6)
    parser.add_argument("--ratio", type=float, default=3.125)
    parser.add_argument("--kind", choices=("bit", "phase"), default="bit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args)
    text = json.dumps(report, indent=2, allow_nan=True, default=str)
    print(text)
    if args.output:
        _write(args.output, report)


if __name__ == "__main__":
    main()

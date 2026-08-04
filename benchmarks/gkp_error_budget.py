"""Run and archive the July 30 experimental GKP logical error budget."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(
    0,
    os.environ.get("JAXQUANTUM_ROOT", str(Path(__file__).resolve().parents[1])),
)

import jax
import matplotlib.pyplot as plt
import numpy as np

from experiments.circuit import sbs_device as model
from experiments.circuit.sbs_parameters import GKP_JULY30


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decay(result):
    values = asdict(result)
    values.pop("contrast")
    values["end_contrast_fraction"] = float(
        abs(result.contrast[-1] / result.contrast[0])
    )
    return values


def _budget(budget):
    return {
        **budget.summary(),
        "baseline": _decay(budget.baseline),
        "all_on": _decay(budget.all_on),
        "isolated": {name: _decay(result) for name, result in budget.isolated.items()},
        "without": {name: _decay(result) for name, result in budget.without.items()},
    }


def _git_metadata(root):
    safe = f"safe.directory={root.as_posix()}"

    def git(*arguments):
        return subprocess.run(
            ["git", "-c", safe, "-C", str(root), *arguments],
            capture_output=True,
            check=False,
            text=True,
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--short")}


def _save_data(path, times, budgets):
    arrays = {"time_s": times}
    for axis, budget in budgets.items():
        arrays[f"{axis}_baseline"] = budget.baseline.contrast
        arrays[f"{axis}_all_on"] = budget.all_on.contrast
        for name, result in budget.isolated.items():
            arrays[f"{axis}_isolated_{name}"] = result.contrast
        for name, result in budget.without.items():
            arrays[f"{axis}_without_{name}"] = result.contrast
    np.savez_compressed(path, **arrays)


def _save_plot(path, times, budgets):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    time_us = 1e6 * times
    for axis, budget in budgets.items():
        for label, result, style in (
            ("all on", budget.all_on, "-"),
            ("ideal baseline", budget.baseline, "--"),
        ):
            contrast = np.abs(result.contrast / result.contrast[0])
            axes[0].semilogy(
                time_us,
                contrast,
                style,
                label=f"{axis.upper()} {label}",
            )
    axes[0].set(
        xlabel="Elapsed time (µs)",
        ylabel="Normalized logical contrast",
        ylim=(1e-4, 1.2),
        title="GKP logical decay",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    channels = model.ERROR_CHANNELS
    positions = np.arange(len(channels))
    width = 0.38
    for offset, (axis, budget) in zip((-width / 2, width / 2), budgets.items()):
        rates = [budget.context_increments[name] / 1e3 for name in channels]
        axes[1].bar(
            positions + offset,
            rates,
            width,
            label=axis.upper(),
        )
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set(
        ylabel="All-on-context increment (kHz)",
        title="Logical error budget",
        xticks=positions,
        xticklabels=[name.replace("_", "\n") for name in channels],
    )
    axes[1].tick_params(axis="x", labelsize=7)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_summary(path, report):
    axes = report["results"]
    scope = "all three CDs" if report["cd_t1_scope"] == "all" else "only the big CD"
    rows = []
    for axis, result in axes.items():
        rows.append(
            f"| {axis.upper()} | {1e6 * result['all_on']['lifetime']:.3f} | "
            f"{result['all_on']['r2']:.6f} | "
            f"{', '.join(result['ranking'][:3])} |"
        )
    path.write_text(
        "\n".join(
            (
                "# July 30 GKP logical error budget",
                "",
                "The simulation uses the retained four-way `+Z,+X,-Z,-X` "
                "control. One saved simulation cycle is a Z/X pair "
                "(8.712 µs); two cycles are one experimental round (17.424 µs).",
                "",
                "| Axis | All-on lifetime (µs) | fit R² | leading channels |",
                "|---|---:|---:|---|",
                *rows,
                "",
                f"The 56.38 µs repeated-ECD contrast lifetime is applied to "
                f"{scope}. It was measured at the large-CD operating point; "
                "the 1% reset error is a sensitivity assumption.",
                "",
                "![Logical decay and error budget](analysis.png)",
                "",
            )
        ),
        encoding="utf-8",
    )


def run(args):
    if args.cycles % 2:
        raise ValueError("cycles must be even to end on a complete four-way round")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    control = GKP_JULY30["control"]
    device = model.GKP_JULY30_DEVICE
    if args.cd_t1_scope == "big":
        device = replace(
            device,
            qubit_t1_cd=(device.qubit_t1, device.qubit_t1_cd, device.qubit_t1),
            qubit_cd_excited_population=(0.0, 0.5, 0.0),
        )
    protocol_args = {
        name: control[name]
        for name in (
            "delta",
            "small_ratio",
            "small_displacement_scales",
            "big_displacement",
            "epsilon_model",
            "final_storage_rotation",
            "alternate_cd_direction",
        )
    }
    cycle_time = model.round_time(device, 2)
    budgets = {}
    wall = {}
    for axis in ("x", "z"):
        start = time.perf_counter()
        budgets[axis] = model.gkp_error_budget(
            dimension=args.dimension,
            state_delta=control["state_delta"],
            kind=axis,
            cycles=args.cycles,
            microsteps=args.microsteps,
            device=device,
            fit_start=args.fit_start,
            fit_floor=args.fit_floor,
            max_loss=args.max_loss,
            max_reset=args.max_reset,
            **protocol_args,
        )
        wall[axis] = time.perf_counter() - start

    root = Path(__file__).resolve().parents[1]
    parameter_set = deepcopy(GKP_JULY30)
    parameter_set["device"] = asdict(device)
    parameter_set["provenance"]["qubit_t1_cd_scope"] = args.cd_t1_scope
    parameters = {
        "parameter_set": "gkp_july30",
        "cd_t1_scope": args.cd_t1_scope,
        **parameter_set,
        "simulation": {
            "dimension": args.dimension,
            "cycles": args.cycles,
            "fit_start": args.fit_start,
            "fit_floor": args.fit_floor,
            "microsteps": args.microsteps,
            "jump_samples": 4,
            "max_loss": args.max_loss,
            "max_reset": args.max_reset,
            "cycle_definition": "complementary Z/X stabilizer pair",
            "cd_t1_scope": args.cd_t1_scope,
        },
    }
    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jaxquantum": _git_metadata(root),
        "parameter_set": "gkp_july30",
        "cycle_time_s": cycle_time,
        "four_way_round_time_s": 2 * cycle_time,
        "wall_s": wall,
        "results": {axis: _budget(budget) for axis, budget in budgets.items()},
    }
    times = np.arange(args.cycles + 1) * cycle_time
    (output / "parameters.json").write_text(
        json.dumps(_jsonable(parameters), indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "results.json").write_text(
        json.dumps(_jsonable(report), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    _save_data(output / "data.npz", times, budgets)
    _save_plot(output / "analysis.png", times, budgets)
    _save_summary(output / "summary.md", _jsonable(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=60)
    parser.add_argument("--cycles", type=int, default=80)
    parser.add_argument("--fit-start", type=int, default=4)
    parser.add_argument("--fit-floor", type=float, default=1e-10)
    parser.add_argument("--microsteps", type=int, default=1)
    parser.add_argument("--max-loss", type=int, default=8)
    parser.add_argument("--max-reset", type=int, default=12)
    parser.add_argument("--cd-t1-scope", choices=("all", "big"), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps(_jsonable(report), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

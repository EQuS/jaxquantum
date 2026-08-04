"""Compare the shared GKP sBs simulation with a colleague implementation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

sys.path.insert(
    0,
    os.environ.get("JAXQUANTUM_ROOT", str(Path(__file__).resolve().parents[1])),
)

import jax
import jax.numpy as jnp
import numpy as np

import jaxquantum as jqt
import jaxquantum.circuits as jqtc
import jaxquantum.codes as jqcodes
from experiments.circuit import sbs_device as shared


def _load_colleague(source):
    jaxopt = types.ModuleType("jaxopt")
    jaxopt.GaussNewton = object
    jax_tqdm = types.ModuleType("jax_tqdm")
    jax_tqdm.scan_tqdm = lambda *args, **kwargs: lambda function: function
    sys.modules.setdefault("jaxopt", jaxopt)
    sys.modules.setdefault("jax_tqdm", jax_tqdm)
    spec = importlib.util.spec_from_file_location("gkp_sbs_colleague", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _block(tree):
    jqt.block_until_ready(tree)
    return tree


def _stats(final, values, cycle_time, fit_start):
    final, values = jax.device_get((final, values))
    contrast = np.asarray(values[:, 0] - values[:, 1])
    lifetime, rate, r2 = shared.fit_decay(
        np.arange(values.shape[0]) * cycle_time,
        contrast,
        start=fit_start,
    )
    traces = np.trace(final, axis1=-2, axis2=-1)
    return {
        "lifetime_us": lifetime * 1e6,
        "rate_s_inv": rate,
        "r2": r2,
        "contrast": contrast,
        "trace_error": float(np.max(np.abs(traces - 1))),
        "hermiticity_error": float(
            np.max(np.abs(final - final.conj().swapaxes(-1, -2)))
        ),
        "minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(final))),
    }


def _parameters(name):
    if name == "legacy":
        return {
            "delta": 0.42,
            "small_ratio": 1.3,
            "device": shared.GKP_LEGACY_DEVICE,
        }
    return {
        "delta": 0.428,
        "small_ratio": 1.083,
        "device": shared.GKP_DEVICE,
    }


def _colleague_arguments(dimension, parameters):
    device = parameters["device"]
    code = jqcodes.GKPQubit({"delta": parameters["delta"], "N": dimension})
    initial = [
        jqt.basis(2, 0) ^ code.basis["-x"],
        jqt.basis(2, 0) ^ code.basis["+x"],
    ]
    scale = 1e9
    channels = {
        "resonator T1": {
            "T1": device.storage_t1 * scale,
            "n_bar": device.storage_nbar,
        },
        "resonator Tphi": {"Tphi": device.storage_tphi * scale},
        "qubit T1": {
            "T1": device.qubit_t1 * scale,
            "n_bar": device.qubit_excited_population,
        },
        "qubit Tphi": {"Tphi": device.qubit_tphi * scale},
        "qubit T1 CD": {
            "T1": device.qubit_t1_cd * scale,
            "n_bar": device.qubit_excited_population,
        },
        "qubit reset": {
            "reset_p_ee": device.reset_error,
            "t_rst": device.reset_duration * scale,
            "chi": device.reset_chi / scale,
        },
        "qubit X": {"thetas": jnp.zeros(3)},
    }
    return code, initial, channels


def _colleague_run(module, dimension, cycles, toggles, parameters):
    code, initial, channels = _colleague_arguments(dimension, parameters)
    device = parameters["device"]
    scale = 1e9
    return module.sbs_fast_batch(
        initial_states=initial,
        delta=parameters["delta"],
        sd_ratio=parameters["small_ratio"],
        T=cycles,
        observable=code.common_gates["X_0"],
        t_sqg=device.rotation_durations[1] * scale,
        t_rst=device.reset_duration * scale,
        error_channels=channels,
        speedup=1.0,
        channel_toggles=toggles,
        t_big=device.cd_durations[1] * scale,
        t_small1=device.cd_durations[0] * scale,
        t_small2=device.cd_durations[2] * scale,
        N_block=1,
    )


def _shared_run(dimension, cycles, enabled, jump_samples, parameters):
    initial, observables = shared.gkp_problem(
        dimension,
        parameters["delta"],
    )
    protocol = shared.gkp_protocol(
        dimension,
        delta=parameters["delta"],
        small_ratio=parameters["small_ratio"],
        device=parameters["device"],
        enabled=enabled,
        jump_samples=jump_samples,
    )
    return jqtc.simulate_sbs(initial, observables, protocol, cycles)


def _timed(function):
    start = time.perf_counter()
    result = _block(function())
    return result, time.perf_counter() - start


def _restore_jump_names(module, correct):
    original = (module.jqt.sigmam, module.jqt.sigmap)
    if correct:
        module.jqt.sigmam, module.jqt.sigmap = original[::-1]
    module._CD_Ancilla_Decay_Kraus_Map_JIT.clear_cache()
    return original


def _colleague_budget(
    module,
    dimension,
    cycles,
    cycle_time,
    fit_start,
    parameters,
    context,
):
    channels = shared.ERROR_CHANNELS

    def run(enabled):
        toggles = [channel in enabled for channel in channels] + [False]
        return _stats(
            *_colleague_final_values(
                _block(
                    _colleague_run(
                        module,
                        dimension,
                        cycles,
                        toggles,
                        parameters,
                    )
                )
            ),
            cycle_time,
            fit_start,
        )

    variants = [
        (),
        channels,
        *((channel,) for channel in channels),
    ]
    if context:
        variants.extend(set(channels) - {channel} for channel in channels)
    results = [run(set(enabled)) for enabled in variants]
    baseline, all_on = results[:2]
    split = 2 + len(channels)
    isolated = dict(zip(channels, results[2:split]))
    isolated_increments = {
        channel: isolated[channel]["rate_s_inv"] - baseline["rate_s_inv"]
        for channel in channels
    }
    context_increments = {}
    if context:
        without = dict(zip(channels, results[split:]))
        context_increments = {
            channel: all_on["rate_s_inv"] - without[channel]["rate_s_inv"]
            for channel in channels
        }
    return {
        "baseline_rate": baseline["rate_s_inv"],
        "all_on_rate": all_on["rate_s_inv"],
        "isolated_increments": isolated_increments,
        "context_increments": context_increments,
        "interaction_rate": (
            all_on["rate_s_inv"]
            - baseline["rate_s_inv"]
            - sum(isolated_increments.values())
        ),
        "ranking": sorted(
            channels,
            key=(context_increments or isolated_increments).get,
            reverse=True,
        ),
    }


def _colleague_final_values(output):
    _, values, final = output
    return final, values


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(args):
    colleague = _load_colleague(args.source)
    parameters = _parameters(args.parameters)
    device = parameters["device"]
    cycle_time = shared.round_time(device, 2)
    all_shared = shared.ERROR_CHANNELS
    all_colleague = [True] * 6 + [False]

    jax.clear_caches()
    colleague_output, colleague_cold = _timed(
        lambda: _colleague_run(
            colleague,
            args.dimension,
            args.cycles,
            all_colleague,
            parameters,
        )
    )
    _, colleague_warm = _timed(
        lambda: _colleague_run(
            colleague,
            args.dimension,
            args.cycles,
            all_colleague,
            parameters,
        )
    )

    jax.clear_caches()
    start = time.perf_counter()
    shared_initial, shared_observables = shared.gkp_problem(
        args.dimension,
        parameters["delta"],
    )
    shared_protocol = shared.gkp_protocol(
        args.dimension,
        delta=parameters["delta"],
        small_ratio=parameters["small_ratio"],
        device=device,
        enabled=all_shared,
        jump_samples=args.jump_samples,
    )
    shared_build = time.perf_counter() - start
    shared_output, shared_cold = _timed(
        lambda: jqtc.simulate_sbs(
            shared_initial,
            shared_observables,
            shared_protocol,
            args.cycles,
        )
    )
    _, shared_warm = _timed(
        lambda: jqtc.simulate_sbs(
            shared_initial,
            shared_observables,
            shared_protocol,
            args.cycles,
        )
    )

    cd_population = device.qubit_excited_population
    source_parameters = {
        **parameters,
        "device": replace(
            device,
            qubit_cd_excited_population=cd_population / (1 + 2 * cd_population),
        ),
    }
    source_initial, source_observables = shared.gkp_problem(
        args.dimension,
        parameters["delta"],
    )
    source_protocol = shared.gkp_protocol(
        args.dimension,
        delta=parameters["delta"],
        small_ratio=parameters["small_ratio"],
        device=source_parameters["device"],
        enabled=all_shared,
        jump_samples=args.jump_samples,
    )
    source_output = _block(
        jqtc.simulate_sbs(
            source_initial,
            source_observables,
            source_protocol,
            args.cycles,
        )
    )

    original_names = _restore_jump_names(colleague, correct=True)
    corrected_output = _block(
        _colleague_run(
            colleague,
            args.dimension,
            args.cycles,
            all_colleague,
            parameters,
        )
    )
    colleague_budget = None
    if args.budget:
        budget_start = time.perf_counter()
        colleague_budget = _colleague_budget(
            colleague,
            args.dimension,
            args.budget_cycles,
            cycle_time,
            args.fit_start,
            parameters,
            args.budget_context,
        )
        colleague_budget["wall_s"] = time.perf_counter() - budget_start
    colleague.jqt.sigmam, colleague.jqt.sigmap = original_names
    colleague._CD_Ancilla_Decay_Kraus_Map_JIT.clear_cache()

    ideal_colleague = _block(
        _colleague_run(
            colleague,
            args.dimension,
            args.cycles,
            [False] * 7,
            parameters,
        )
    )
    ideal_shared = _block(
        _shared_run(
            args.dimension,
            args.cycles,
            (),
            args.jump_samples,
            parameters,
        )
    )

    def colleague_stats(output):
        _, values, final = output
        return _stats(final, values, cycle_time, args.fit_start)

    shared_stats = _stats(
        shared_output[0],
        shared_output[1],
        cycle_time,
        args.fit_start,
    )
    source_stats = _stats(
        source_output[0],
        source_output[1],
        cycle_time,
        args.fit_start,
    )
    colleague_stats_written = colleague_stats(colleague_output)
    colleague_stats_corrected = colleague_stats(corrected_output)
    ideal_colleague_stats = colleague_stats(ideal_colleague)
    ideal_shared_stats = _stats(
        ideal_shared[0],
        ideal_shared[1],
        cycle_time,
        args.fit_start,
    )

    def contrast_error(left, right):
        difference = left["contrast"] - right["contrast"]
        return {
            "max_absolute": float(np.max(np.abs(difference))),
            "relative_l2": float(
                np.linalg.norm(difference) / np.linalg.norm(right["contrast"])
            ),
        }

    return _jsonable(
        {
            "configuration": vars(args),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "timings_s": {
                "colleague_full_cold": colleague_cold,
                "colleague_full_warm": colleague_warm,
                "shared_build": shared_build,
                "shared_kernel_cold": shared_cold,
                "shared_kernel_warm": shared_warm,
            },
            "all_on": {
                "shared": shared_stats,
                "shared_colleague_cd_population": source_stats,
                "colleague_as_written": colleague_stats_written,
                "colleague_jump_corrected": colleague_stats_corrected,
                "shared_vs_written": contrast_error(
                    shared_stats,
                    colleague_stats_written,
                ),
                "shared_vs_jump_corrected": contrast_error(
                    shared_stats,
                    colleague_stats_corrected,
                ),
            },
            "ideal": {
                "shared": ideal_shared_stats,
                "colleague": ideal_colleague_stats,
                "difference": contrast_error(
                    ideal_shared_stats,
                    ideal_colleague_stats,
                ),
            },
            "corrected_colleague_budget": colleague_budget,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--dimension", type=int, default=60)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--fit-start", type=int, default=4)
    parser.add_argument("--jump-samples", type=int, default=12)
    parser.add_argument(
        "--parameters",
        choices=("current", "legacy"),
        default="current",
    )
    parser.add_argument("--budget", action="store_true")
    parser.add_argument("--budget-context", action="store_true")
    parser.add_argument("--budget-cycles", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args)
    report["configuration"]["source"] = str(args.source)
    report["configuration"]["output"] = (
        None if args.output is None else str(args.output)
    )
    text = json.dumps(report, indent=2, allow_nan=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()

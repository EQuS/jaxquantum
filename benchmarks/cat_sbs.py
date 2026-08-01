"""End-to-end benchmark for the external cat-sBs lifetime model."""

from __future__ import annotations

import argparse
from functools import lru_cache
import importlib.util
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(
    0,
    os.environ.get(
        "JAXQUANTUM_ROOT",
        str(Path(__file__).resolve().parents[1]),
    ),
)

import jax
import numpy as np

import jaxquantum as jqt
import jaxquantum.circuits as jqtc

try:
    import psutil
except ImportError:  # pragma: no cover - optional host-memory detail
    psutil = None


def _load_model(source: Path):
    spec = importlib.util.spec_from_file_location("cat_sbs_benchmark_model", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def _identity_channel(dimension):
    return jqtc.Channel(
        dimension,
        lambda rho, params: rho,
        kraus=[jqt.identity(dimension)],
    )


def _enable_direct_channels(model, include_storage=True, include_qubit=False):
    original_apply = model._apply_kraus
    original_apply_qubit = model._apply_qubit_kraus
    original_completeness = model.completeness_error

    @lru_cache(maxsize=None)
    def loss(dimension, duration, lifetime, max_loss):
        if lifetime is None:
            return _identity_channel(dimension)
        probability = 1 - np.exp(-duration / lifetime)
        return jqtc.Amp_Damp(dimension, probability, max_loss)

    @lru_cache(maxsize=None)
    def dephasing(dimension, duration, lifetime, order):
        if lifetime is None:
            return _identity_channel(dimension)
        return jqtc.Dephasing_Ch(
            dimension,
            2 * duration / lifetime,
            order,
        )

    @lru_cache(maxsize=None)
    def qubit_relaxation(duration, lifetime):
        if lifetime is None:
            return _identity_channel(2)
        return jqtc.Thermal_Ch_Qb(1 - np.exp(-duration / lifetime), 0)

    @lru_cache(maxsize=None)
    def qubit_dephasing(duration, lifetime):
        if lifetime is None:
            return _identity_channel(2)
        probability = (1 - np.exp(-duration / lifetime)) / 2
        return jqtc.Dephasing_Ch_Qb(probability)

    def apply(channel, rhos):
        if isinstance(channel, jqtc.Gate):
            return jqtc.apply_channel(channel, rhos)
        return original_apply(channel, rhos)

    def apply_qubit(channel, rhos):
        if isinstance(channel, jqtc.Gate):
            return jqtc.apply_channel(channel, rhos, axes=(1, 3))
        return original_apply_qubit(channel, rhos)

    def completeness(channel):
        if isinstance(channel, jqtc.Gate):
            channel = channel.KM.data
        return original_completeness(channel)

    model.completeness_error = completeness
    if include_storage:
        model._resonator_loss = loss
        model._resonator_dephasing = dephasing
        model._apply_kraus = apply
    if include_qubit:
        model._qubit_relaxation = qubit_relaxation
        model._qubit_dephasing = qubit_dephasing
        model._apply_qubit_kraus = apply_qubit


def _enable_channel_cache(model):
    for name in (
        "_qubit_relaxation",
        "_qubit_dephasing",
        "_resonator_loss",
        "_resonator_dephasing",
    ):
        setattr(model, name, lru_cache(maxsize=None)(getattr(model, name)))


def _ready(result):
    _block(vars(result))
    return result


def _block(tree):
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _profile_kernel(function, *args, iterations):
    jax.clear_caches()
    start = time.perf_counter()
    lowered = jax.jit(function).lower(*args)
    lowering = time.perf_counter() - start
    start = time.perf_counter()
    compiled = lowered.compile()
    compilation = time.perf_counter() - start
    start = time.perf_counter()
    _block(compiled(*args))
    first = time.perf_counter() - start
    _block(compiled(*args))
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        _block(compiled(*args))
        samples.append(time.perf_counter() - start)

    memory = compiled.memory_analysis()
    hlo = lowered.as_text()
    return {
        "timings_s": {
            "lowering": lowering,
            "compilation": compilation,
            "first_execution": first,
            "cold_total": lowering + compilation + first,
            "warm_median": statistics.median(samples),
            "warm_min": min(samples),
            "warm_max": max(samples),
        },
        "memory_bytes": {
            key: getattr(memory, key, None)
            for key in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "alias_size_in_bytes",
                "temp_size_in_bytes",
                "peak_memory_in_bytes",
                "generated_code_size_in_bytes",
            )
        },
        "hlo": {
            "characters": len(hlo),
            "lines": hlo.count("\n") + 1,
        },
    }


def _peak_rss(call):
    if psutil is None:
        return call(), None

    process = psutil.Process()
    start = process.memory_info().rss
    peak = start
    stop = threading.Event()

    def sample():
        nonlocal peak
        while not stop.wait(0.001):
            peak = max(peak, process.memory_info().rss)

    thread = threading.Thread(target=sample)
    thread.start()
    try:
        result = call()
    finally:
        stop.set()
        thread.join()
    end = process.memory_info().rss
    return result, {
        "start": start,
        "end": end,
        "peak": peak,
        "temporary": max(0, peak - start),
    }


def _result_summary(result):
    return {
        "bit_rate": result.bit_rate,
        "phase_rate": result.phase_rate,
        "bit_lifetime": result.bit_lifetime,
        "phase_lifetime": result.phase_lifetime,
        "trace_error": result.trace_error,
        "tail_population": result.tail_population,
        "sbs_tp_error": result.sbs_tp_error,
        "loss_tp_error": result.loss_tp_error,
        "bit_trace": result.bit_trace,
        "phase_trace": result.phase_trace,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="path to cat_sbs.py")
    parser.add_argument("--nbar", type=float, default=4.0)
    parser.add_argument("--squeezing", type=float, default=0.0)
    parser.add_argument("--cycles", type=int, default=480)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--delta", type=float, default=0.6)
    parser.add_argument("--ratio", type=float, default=3.125)
    parser.add_argument("--cold-iterations", type=int, default=3)
    parser.add_argument("--warm-iterations", type=int, default=5)
    parser.add_argument("--direct-storage", action="store_true")
    parser.add_argument("--direct-qubit", action="store_true")
    parser.add_argument("--direct-channels", action="store_true")
    parser.add_argument("--cache-channels", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    model = _load_model(args.source.resolve())
    if args.direct_storage or args.direct_qubit or args.direct_channels:
        _enable_channel_cache(model)
        _enable_direct_channels(
            model,
            include_storage=args.direct_storage or args.direct_channels,
            include_qubit=args.direct_qubit or args.direct_channels,
        )
    elif args.cache_channels:
        _enable_channel_cache(model)
    params = model.device_parameters()
    call_kwargs = {
        "cycles": args.cycles,
        "N": args.dimension,
        "params": params,
        "delta": args.delta,
        "ratio": args.ratio,
    }

    def simulate():
        return _ready(
            model.simulate_device_lifetimes(
                args.nbar,
                args.squeezing,
                **call_kwargs,
            )
        )

    def setup():
        alpha = model.alpha_from_nbar(args.nbar, args.squeezing)
        rhos = model.initial_states(args.dimension, alpha, args.squeezing)
        observables = model.logical_observables(args.dimension)
        ops = model._device_ops(
            args.dimension,
            alpha,
            args.squeezing,
            params,
            delta=args.delta,
            ratio=args.ratio,
        )
        _block((rhos, observables, ops))
        return rhos, observables, ops

    cold_samples = []
    result = None
    for _ in range(args.cold_iterations):
        jax.clear_caches()
        start = time.perf_counter()
        result = simulate()
        cold_samples.append(time.perf_counter() - start)

    warm_samples = []
    for _ in range(args.warm_iterations):
        start = time.perf_counter()
        result = simulate()
        warm_samples.append(time.perf_counter() - start)

    jax.clear_caches()
    _, cold_rss = _peak_rss(simulate)
    _, warm_rss = _peak_rss(simulate)

    jax.clear_caches()
    start = time.perf_counter()
    rhos, observables, ops = setup()
    setup_cold = time.perf_counter() - start
    setup_samples = []
    for _ in range(args.warm_iterations):
        start = time.perf_counter()
        setup()
        setup_samples.append(time.perf_counter() - start)

    def evolve(rhos, observables, ops):
        return model._device_evolve_observables(
            rhos,
            observables,
            ops,
            ops,
            args.cycles,
        )

    kernel = _profile_kernel(
        evolve,
        rhos,
        observables,
        ops,
        iterations=args.warm_iterations,
    )
    payload = {
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxquantum": jqt.__version__,
        "jaxquantum_source": str(Path(jqt.__file__).resolve()),
        "cat_sbs_source": str(args.source.resolve()),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key not in {"source", "output"}
        },
        "end_to_end": {
            "cold_median_s": statistics.median(cold_samples),
            "cold_min_s": min(cold_samples),
            "cold_max_s": max(cold_samples),
            "warm_median_s": statistics.median(warm_samples),
            "warm_min_s": min(warm_samples),
            "warm_max_s": max(warm_samples),
            "cold_samples_s": cold_samples,
            "warm_samples_s": warm_samples,
            "host_rss_bytes": {"cold": cold_rss, "warm": warm_rss},
        },
        "setup": {
            "cold_s": setup_cold,
            "warm_median_s": statistics.median(setup_samples),
            "warm_min_s": min(setup_samples),
            "warm_max_s": max(setup_samples),
        },
        "kernel": kernel,
        "result": _result_summary(result),
    }
    output = json.dumps(
        payload,
        indent=2,
        default=lambda value: np.asarray(value).tolist(),
    )
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

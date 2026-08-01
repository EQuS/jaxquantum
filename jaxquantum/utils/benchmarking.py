"""Synchronized JAX compilation, execution, and memory measurements."""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Callable, Mapping
from typing import Any

import jax
import numpy as np


_MEMORY_FIELDS = (
    "argument_size_in_bytes",
    "output_size_in_bytes",
    "alias_size_in_bytes",
    "temp_size_in_bytes",
    "peak_memory_in_bytes",
    "generated_code_size_in_bytes",
    "host_argument_size_in_bytes",
    "host_output_size_in_bytes",
    "host_alias_size_in_bytes",
    "host_temp_size_in_bytes",
)


def block_until_ready(tree) -> None:
    """Synchronize every array leaf in a PyTree."""
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def lower_jax_function(
    function: Callable,
    *args,
    jit_kwargs: Mapping[str, Any] | None = None,
    call_kwargs: Mapping[str, Any] | None = None,
):
    """Lower a function with the supplied JIT and call arguments."""
    return jax.jit(function, **dict(jit_kwargs or {})).lower(
        *args,
        **dict(call_kwargs or {}),
    )


def jax_memory_stats(compiled) -> dict[str, int | None]:
    """Return XLA's compiled buffer-size estimates."""
    memory = compiled.memory_analysis()
    if memory is None:
        return dict.fromkeys(_MEMORY_FIELDS)
    return {field: getattr(memory, field, None) for field in _MEMORY_FIELDS}


def jax_device_memory_stats() -> dict[str, dict[str, int] | None]:
    """Return allocator statistics reported by each JAX device."""
    output = {}
    for device in jax.devices():
        stats = device.memory_stats()
        output[str(device)] = (
            None
            if stats is None
            else {
                key: int(value)
                for key, value in stats.items()
                if isinstance(value, (int, np.integer))
            }
        )
    return output


def jax_hlo(
    function: Callable,
    *args,
    jit_kwargs: Mapping[str, Any] | None = None,
    call_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Return lowered StableHLO text for a function call."""
    return lower_jax_function(
        function,
        *args,
        jit_kwargs=jit_kwargs,
        call_kwargs=call_kwargs,
    ).as_text()


def _percentiles(samples):
    if len(samples) < 2:
        return samples[0], samples[0]
    deciles = statistics.quantiles(samples, n=10, method="inclusive")
    return deciles[0], deciles[-1]


def _scalar_costs(costs):
    if not costs:
        return {}
    output = {}
    for key, value in costs.items():
        try:
            output[key] = value.item()
        except AttributeError:
            output[key] = value
    return output


def _benchmark_once(
    function: Callable,
    args,
    iterations,
    warmup,
    clear_caches,
    include_hlo,
    jit_kwargs,
    call_kwargs,
):
    if clear_caches:
        jax.clear_caches()
    device_memory_before = jax_device_memory_stats()

    start = time.perf_counter()
    lowered = lower_jax_function(
        function,
        *args,
        jit_kwargs=jit_kwargs,
        call_kwargs=call_kwargs,
    )
    lowering_s = time.perf_counter() - start

    start = time.perf_counter()
    compiled = lowered.compile()
    compilation_s = time.perf_counter() - start

    start = time.perf_counter()
    first_output = compiled(*args, **call_kwargs)
    block_until_ready(first_output)
    first_execution_s = time.perf_counter() - start

    for _ in range(warmup):
        block_until_ready(compiled(*args, **call_kwargs))

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        block_until_ready(compiled(*args, **call_kwargs))
        samples.append(time.perf_counter() - start)

    p10, p90 = _percentiles(samples)
    hlo = lowered.as_text()
    report = {
        "function": getattr(function, "__qualname__", repr(function)),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "timings_s": {
            "lowering": lowering_s,
            "compilation": compilation_s,
            "first_execution": first_execution_s,
            "cold_total": lowering_s + compilation_s + first_execution_s,
            "warm_median": statistics.median(samples),
            "warm_min": min(samples),
            "warm_max": max(samples),
            "warm_p10": p10,
            "warm_p90": p90,
        },
        "memory_bytes": jax_memory_stats(compiled),
        "device_memory_stats": {
            "before": device_memory_before,
            "after": jax_device_memory_stats(),
        },
        "cost_analysis": _scalar_costs(compiled.cost_analysis()),
        "hlo": {
            "characters": len(hlo),
            "lines": hlo.count("\n") + 1,
        },
        "iterations": iterations,
        "warmup": warmup,
    }
    if include_hlo:
        report["hlo"]["text"] = hlo
    return report, first_output


def _cast_precision(tree, dtype, complex_dtype):
    def cast(leaf):
        leaf_dtype = getattr(leaf, "dtype", None)
        if leaf_dtype is None:
            return leaf
        try:
            if np.issubdtype(leaf_dtype, np.complexfloating):
                return leaf.astype(complex_dtype)
            if np.issubdtype(leaf_dtype, np.floating):
                return leaf.astype(dtype)
        except TypeError:
            pass
        return leaf

    return jax.tree.map(cast, tree)


def _accuracy_stats(reference, candidate):
    if jax.tree.structure(reference) != jax.tree.structure(candidate):
        raise ValueError("precision outputs have different PyTree structures")
    reference_leaves = jax.tree.leaves(reference)
    candidate_leaves = jax.tree.leaves(candidate)

    max_absolute = 0.0
    max_relative = 0.0
    squared_error = 0.0
    squared_reference = 0.0
    elements = 0
    for reference_leaf, candidate_leaf in zip(
        reference_leaves,
        candidate_leaves,
    ):
        reference_array = np.asarray(jax.device_get(reference_leaf))
        candidate_array = np.asarray(jax.device_get(candidate_leaf))
        if reference_array.shape != candidate_array.shape:
            raise ValueError("precision outputs have different shapes")
        if not np.issubdtype(reference_array.dtype, np.inexact):
            continue
        difference = np.abs(reference_array - candidate_array)
        magnitude = np.abs(reference_array)
        if difference.size:
            max_absolute = max(max_absolute, float(np.max(difference)))
            threshold = np.finfo(reference_array.real.dtype).tiny
            relative = difference / np.maximum(magnitude, threshold)
            max_relative = max(max_relative, float(np.max(relative)))
            squared_error += float(np.sum(difference**2))
            squared_reference += float(np.sum(magnitude**2))
            elements += difference.size

    relative_l2 = (
        np.sqrt(squared_error / squared_reference)
        if squared_reference
        else np.sqrt(squared_error)
    )
    return {
        "max_absolute_error": max_absolute,
        "max_relative_error": max_relative,
        "relative_l2_error": float(relative_l2),
        "elements_compared": elements,
    }


def _ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _difference(left, right):
    if left is None or right is None:
        return None
    return left - right


def benchmark_precision(
    function: Callable,
    *args,
    iterations: int = 25,
    warmup: int = 1,
    clear_caches: bool = True,
    include_hlo: bool = False,
    jit_kwargs: Mapping[str, Any] | None = None,
    call_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare double and single precision performance and output accuracy."""
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    original_x64 = jax.config.x64_enabled
    reports = {}
    outputs = {}
    try:
        for name, enabled, real_dtype, complex_dtype in (
            ("double", True, np.float64, np.complex128),
            ("single", False, np.float32, np.complex64),
        ):
            jax.config.update("jax_enable_x64", enabled)
            precision_args = _cast_precision(args, real_dtype, complex_dtype)
            precision_kwargs = _cast_precision(
                dict(call_kwargs or {}),
                real_dtype,
                complex_dtype,
            )
            reports[name], outputs[name] = _benchmark_once(
                function,
                precision_args,
                iterations,
                warmup,
                clear_caches,
                include_hlo,
                jit_kwargs,
                precision_kwargs,
            )
    finally:
        jax.config.update("jax_enable_x64", original_x64)
        if clear_caches:
            jax.clear_caches()

    double_timing = reports["double"]["timings_s"]
    single_timing = reports["single"]["timings_s"]
    double_memory = reports["double"]["memory_bytes"]
    single_memory = reports["single"]["memory_bytes"]
    return {
        "double": reports["double"],
        "single": reports["single"],
        "accuracy": _accuracy_stats(outputs["double"], outputs["single"]),
        "single_vs_double": {
            "cold_speedup": _ratio(
                double_timing["cold_total"],
                single_timing["cold_total"],
            ),
            "warm_speedup": _ratio(
                double_timing["warm_median"],
                single_timing["warm_median"],
            ),
            "temporary_memory_ratio": _ratio(
                double_memory["temp_size_in_bytes"],
                single_memory["temp_size_in_bytes"],
            ),
            "temporary_bytes_saved": (
                _difference(
                    double_memory["temp_size_in_bytes"],
                    single_memory["temp_size_in_bytes"],
                )
            ),
            "peak_memory_ratio": _ratio(
                double_memory["peak_memory_in_bytes"],
                single_memory["peak_memory_in_bytes"],
            ),
            "peak_bytes_saved": (
                _difference(
                    double_memory["peak_memory_in_bytes"],
                    single_memory["peak_memory_in_bytes"],
                )
            ),
        },
    }


def benchmark_jax_function(
    function: Callable,
    *args,
    iterations: int = 25,
    warmup: int = 1,
    clear_caches: bool = True,
    include_hlo: bool = False,
    compare_precision: bool = False,
    jit_kwargs: Mapping[str, Any] | None = None,
    call_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect synchronized JAX timing, memory, HLO, cost, and precision stats."""
    if compare_precision:
        return benchmark_precision(
            function,
            *args,
            iterations=iterations,
            warmup=warmup,
            clear_caches=clear_caches,
            include_hlo=include_hlo,
            jit_kwargs=jit_kwargs,
            call_kwargs=call_kwargs,
        )
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    return _benchmark_once(
        function,
        args,
        iterations,
        warmup,
        clear_caches,
        include_hlo,
        jit_kwargs,
        dict(call_kwargs or {}),
    )[0]

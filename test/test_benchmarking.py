import json

import jax
import jax.numpy as jnp
import pytest

import jaxquantum as jqt


def test_benchmark_jax_function_reports_synchronized_stats():
    operator = jqt.Qarray.create(jnp.eye(8))
    report = jqt.benchmark_jax_function(
        lambda value: (value @ value).data,
        operator,
        iterations=3,
        warmup=0,
        include_hlo=True,
    )

    assert report["timings_s"]["cold_total"] >= 0
    assert report["timings_s"]["warm_median"] >= 0
    assert report["memory_bytes"]["argument_size_in_bytes"] > 0
    assert set(report["device_memory_stats"]) == {"before", "after"}
    assert report["hlo"]["characters"] == len(report["hlo"]["text"])
    assert "flops" in report["cost_analysis"]
    json.dumps(report)


def test_benchmark_jax_function_validates_iterations():
    with pytest.raises(ValueError, match="iterations"):
        jqt.benchmark_jax_function(lambda value: value, jnp.ones(2), iterations=0)


def test_jax_hlo_accepts_call_kwargs():
    hlo = jqt.jax_hlo(
        lambda value, scale=1.0: scale * value,
        jnp.ones(2),
        call_kwargs={"scale": jnp.asarray(2.0)},
    )
    assert "stablehlo.multiply" in hlo


def test_memory_stats_handles_unsupported_backend():
    class Compiled:
        def memory_analysis(self):
            return None

    stats = jqt.jax_memory_stats(Compiled())
    assert all(value is None for value in stats.values())


def test_precision_benchmark_reports_accuracy_speed_and_memory():
    original_x64 = jax.config.x64_enabled
    matrix = jnp.linspace(0.1, 0.9, 256).reshape(16, 16)
    report = jqt.benchmark_jax_function(
        lambda value: jnp.linalg.matrix_power(value, 3),
        matrix,
        iterations=2,
        warmup=0,
        compare_precision=True,
    )

    assert report["accuracy"]["elements_compared"] == matrix.size
    assert report["accuracy"]["relative_l2_error"] < 1e-3
    assert report["single_vs_double"]["warm_speedup"] > 0
    assert (
        report["double"]["memory_bytes"]["argument_size_in_bytes"]
        == 2 * report["single"]["memory_bytes"]["argument_size_in_bytes"]
    )
    assert jax.config.x64_enabled is original_x64
    json.dumps(report)


@pytest.mark.parametrize("initial_x64", [False, True])
def test_precision_benchmark_restores_setting_after_failure(initial_x64):
    original_x64 = jax.config.x64_enabled

    def fail_in_single(value):
        if value.dtype == jnp.float32:
            raise RuntimeError("single precision failure")
        return value

    try:
        jax.config.update("jax_enable_x64", initial_x64)
        with pytest.raises(RuntimeError, match="single precision"):
            jqt.benchmark_precision(
                fail_in_single,
                jnp.ones(2),
                iterations=1,
                warmup=0,
            )
        assert jax.config.x64_enabled is initial_x64
    finally:
        jax.config.update("jax_enable_x64", original_x64)

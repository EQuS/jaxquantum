import itertools

import jax
import jax.numpy as jnp
import pytest

from jaxquantum.devices.analysis import run_jax_sweep


def test_run_jax_sweep_cartesian_product_is_jittable():
    first = jnp.array([1.0, 2.0])
    second = jnp.array([0.1, 0.2, 0.3])

    def run(values):
        return run_jax_sweep(
            {"offset": 2.0},
            {"first": values, "second": second},
            lambda params: params["offset"] + params["first"] * params["second"],
        )

    actual = jax.jit(run)(first)
    expected = jnp.array(
        [2.0 + left * right for left, right in itertools.product(first, second)]
    )
    assert jnp.allclose(actual, expected)


def test_run_jax_sweep_parallel_supports_pytree_results_and_gradients():
    values = jnp.array([1.0, 2.0, 3.0])

    def total(scale):
        result = run_jax_sweep(
            {},
            {"x": values, "y": values + 1},
            lambda params, factor: {
                "value": factor * params["x"] + params["y"],
            },
            fixed_kwargs={"factor": scale},
            is_parallel=True,
        )
        return jnp.sum(result["value"])

    assert jnp.allclose(jax.grad(total)(2.0), jnp.sum(values))


def test_run_jax_sweep_validates_inputs():
    with pytest.raises(ValueError, match="must not be empty"):
        run_jax_sweep({}, {}, lambda params: params)
    with pytest.raises(ValueError, match="equal lengths"):
        run_jax_sweep(
            {},
            {"x": jnp.ones(2), "y": jnp.ones(3)},
            lambda params: params["x"],
            is_parallel=True,
        )

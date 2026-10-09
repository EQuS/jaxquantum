"""Manual tests for two physical GPUs in one JAX process.

Run explicitly with ``pytest -q test/manual_multi_gpu/test_two_gpu_sharding.py``
inside a two-GPU allocation. Pytest excludes this directory from default
discovery so CI never needs GPUs for these tests.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import jaxquantum as jqt


def _two_local_gpus():
    if jax.process_count() != 1:
        pytest.skip("this test uses two GPUs in one JAX process")
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        pytest.skip("no GPU backend is available")
    if len(devices) < 2:
        pytest.skip("two physical GPUs are required")
    return devices[:2]


@pytest.fixture(autouse=True)
def _clear_sharding():
    jqt.clear_default_sharding()
    yield
    jqt.clear_default_sharding()


def _assert_split_between(array, devices):
    shards = array.addressable_shards
    assert {s.device for s in shards} == set(devices)
    assert len({str(s.index) for s in shards}) == 2


def test_jitted_qarray_action_uses_both_gpus():
    devices = _two_local_gpus()
    jqt.set_device_mesh(shape=(2,), axis_names=("mp",))
    h = 0.15 * (jqt.destroy(8) + jqt.create(8))
    psi = jqt.basis(8, 1)

    @jax.jit
    def act(op, state):
        return op @ state

    out = act(h, psi)
    out.data.block_until_ready()
    _assert_split_between(out.data, devices)

    expected = np.zeros(8, dtype=np.complex64)
    expected[0] = 0.15
    expected[2] = 0.15 * np.sqrt(2)
    np.testing.assert_allclose(np.asarray(out.data), expected, rtol=1e-5, atol=1e-6)


def test_sharded_sesolve_matches_exact_phase():
    devices = _two_local_gpus()
    jqt.set_device_mesh(shape=(2,), axis_names=("mp",))
    h = 0.05 * jqt.num(16)
    psi0 = jqt.basis(16, 1)
    times = jnp.linspace(0.0, 1.0, 4)

    states = jqt.sesolve(h, psi0, times)
    states.data.block_until_ready()
    _assert_split_between(states.data, devices)

    expected = np.zeros((len(times), 16), dtype=np.complex64)
    expected[:, 1] = np.exp(-0.05j * np.asarray(times))
    np.testing.assert_allclose(np.asarray(states.data), expected, rtol=1e-5, atol=1e-5)

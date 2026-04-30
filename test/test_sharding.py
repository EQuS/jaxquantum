"""Sharding tests.

Spoofs two CPU devices via ``XLA_FLAGS`` so the tests run on a single laptop
without GPUs/TPUs. The flag must be set *before* JAX is imported anywhere,
so this file imports `jax` itself and is loaded as its own pytest module.
"""

import os

# Must be set before any JAX import. Pytest collects modules independently,
# so this file forks JAX to two-device mode without affecting the rest of
# the suite (other test files have already imported JAX via jaxquantum).
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
).strip()

import sys
import pathlib

# Match the sys.path convention used by the other test files.
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

import jaxquantum as jqt


# ---------------------------------------------------------------------------
# Fixture: ensure two devices are available; otherwise skip the whole module.
# ---------------------------------------------------------------------------

if len(jax.devices()) < 2:
    pytest.skip(
        "Sharding tests require >=2 devices "
        f"(got {len(jax.devices())}). Make sure XLA_FLAGS is set before the "
        "JAX import — re-run with `pytest test/test_sharding.py` standalone.",
        allow_module_level=True,
    )


needs_8 = pytest.mark.skipif(
    len(jax.devices()) < 8,
    reason="2D mesh tests need >=8 devices",
)


def _padded_spec(arr) -> tuple:
    """Return ``arr.sharding.spec`` padded with trailing ``None``s up to ``arr.ndim``.

    ``PartitionSpec`` truncates trailing ``None``s in its string form (so
    ``P('dp', None, None)`` displays as ``P('dp',)``). Tests need the
    padded view to assert per-axis bindings without ``IndexError``.
    """
    raw = tuple(arr.sharding.spec)
    return raw + (None,) * (arr.ndim - len(raw))


@pytest.fixture(autouse=True)
def _reset_sharding():
    """Always start each test with no default sharding configured."""
    jqt.clear_default_sharding()
    yield
    jqt.clear_default_sharding()


# ---------------------------------------------------------------------------
# Settings round-trip
# ---------------------------------------------------------------------------

def test_default_off():
    """No sharding configured -> arrays live on the default device."""
    assert jqt.get_default_sharding() is None
    a = jqt.identity(4)
    # No NamedSharding annotation when default is off.
    assert not isinstance(getattr(a.data, "sharding", None), NamedSharding)


def test_set_and_clear_default():
    jqt.set_device_mesh(shape=(2,), axis_names=("dp",))
    assert jqt.get_default_sharding() is not None
    jqt.clear_default_sharding()
    assert jqt.get_default_sharding() is None


def test_set_device_mesh_validates_lengths():
    with pytest.raises(ValueError):
        jqt.set_device_mesh(shape=(2,), axis_names=("dp", "mp"))


# ---------------------------------------------------------------------------
# Dense path: factories, arithmetic, kron all carry sharding through
# ---------------------------------------------------------------------------

def test_dense_factory_is_sharded():
    jqt.set_device_mesh(shape=(2,), axis_names=("dp",))
    a = jqt.identity(10)
    devices = a.data.sharding.device_set
    assert len(devices) == 2


def test_dense_arithmetic_preserves_sharding():
    jqt.set_device_mesh(shape=(2,), axis_names=("dp",))
    s = jqt.destroy(8) + jqt.create(8)
    assert len(s.data.sharding.device_set) == 2


def test_dense_kron_reshards_output():
    """`kron` changes shape, so XLA needs the explicit re-shard from `_make`."""
    jqt.set_device_mesh(shape=(2,), axis_names=("dp",))
    # Both input sizes must be divisible by the 2-device mesh.
    out = jqt.tensor(jqt.identity(4), jqt.identity(2))
    assert out.data.shape == (8, 8)
    assert len(out.data.sharding.device_set) == 2


# ---------------------------------------------------------------------------
# SparseDIA path
# ---------------------------------------------------------------------------

def test_sparse_dia_factory_is_sharded():
    jqt.set_device_mesh(shape=(2,), axis_names=("dp",))
    d = jqt.destroy(10, implementation=jqt.QarrayImplType.SPARSE_DIA)
    diags = d.data.diags
    assert len(diags.sharding.device_set) == 2


def test_sparse_dia_arithmetic_preserves_sharding():
    jqt.set_device_mesh(shape=(2,), axis_names=("dp",))
    a = jqt.destroy(8, implementation=jqt.QarrayImplType.SPARSE_DIA)
    n = (a.dag() @ a)
    assert len(n.data.diags.sharding.device_set) == 2


# ---------------------------------------------------------------------------
# Validation: shape must be divisible by mesh axis size
# ---------------------------------------------------------------------------

def test_validate_shape_mismatch_raises():
    """Explicit PartitionSpec mismatch raises a friendly ValueError.

    The rank-adaptive default falls back to full replication when no axis
    is divisible (so kets, scalars, etc. don't blow up); validation only
    fires when the user has committed to a specific PartitionSpec.
    """
    jqt.set_device_mesh(
        shape=(2,), axis_names=("dp",),
        partition_spec=PartitionSpec("dp", None),
    )
    with pytest.raises(ValueError, match="not divisible"):
        jqt.identity(5)


def test_adaptive_default_replicates_kets():
    """Kets are (N, 1) — last axis is 1, can't be sharded. Adaptive default
    must shard the leading (N) axis instead, not error."""
    jqt.set_device_mesh(shape=(2,), axis_names=("dp",))
    # basis(N=4, k=0) builds a column vector of shape (4, 1).
    psi = jqt.basis(4, 0)
    assert psi.data.shape == (4, 1)
    # Should not raise; sharding device set should include both devices
    # (axis 0 is sharded along 'dp' since it's divisible by 2).
    spec = psi.data.sharding.spec
    assert spec[0] == "dp"
    assert spec[1] is None


# ---------------------------------------------------------------------------
# BCOO is rejected when sharding is on
# ---------------------------------------------------------------------------

def test_bcoo_under_sharding_raises():
    jqt.set_device_mesh(shape=(2,), axis_names=("dp",))
    with pytest.raises(NotImplementedError, match="SparseBCOO"):
        jqt.Qarray.create(
            jnp.eye(4),
            implementation=jqt.QarrayImplType.SPARSE_BCOO,
        )


# ---------------------------------------------------------------------------
# JIT compatibility: _maybe_shard uses with_sharding_constraint, not device_put
# ---------------------------------------------------------------------------

def test_jit_with_sharded_qarray():
    jqt.set_device_mesh(shape=(2,), axis_names=("dp",))
    a = jqt.identity(8)

    @jax.jit
    def square(x):
        return x @ x

    out = square(a)
    assert len(out.data.sharding.device_set) == 2


# ---------------------------------------------------------------------------
# Eigendecomposition propagates sharding through XLA
# ---------------------------------------------------------------------------

def test_eigh_on_sharded_dense():
    import jax.scipy as jsp
    jqt.set_device_mesh(shape=(2,), axis_names=("dp",))
    H = (jqt.destroy(8) + jqt.create(8)).data
    w, v = jsp.linalg.eigh(H)
    assert len(w.sharding.device_set) == 2 or w.sharding.is_fully_replicated


# ---------------------------------------------------------------------------
# Data-parallel: name-aware default shards leading batch dim under 'dp'
# ---------------------------------------------------------------------------

def test_dp_shards_batch_dim():
    """Batched (B, N, N) with axis_names=('dp',) shards axis 0, not the matrix dims."""
    jqt.set_device_mesh(shape=(2,), axis_names=("dp",))
    a = jqt.destroy(4)
    H_batch_data = jnp.stack([(a.dag() @ a).data, (a + a.dag()).data])  # (2, 4, 4)
    H = jqt.Qarray.create(H_batch_data, bdims=(2,))
    spec = _padded_spec(H.data)
    assert spec == ("dp", None, None)


def test_dp_falls_through_to_matrix_when_no_batch():
    """A bare (N, N) matrix with axis_names=('dp',) still gets sharded (matrix-row)."""
    jqt.set_device_mesh(shape=(2,), axis_names=("dp",))
    a = jqt.identity(8)
    spec = a.data.sharding.spec
    assert spec[0] == "dp"
    assert spec[1] is None


# ---------------------------------------------------------------------------
# Model-parallel: 'mp' axis prefers matrix dim even when a batch is present
# ---------------------------------------------------------------------------

def test_mp_shards_matrix_dim():
    """Bare (N, N) with axis_names=('mp',) shards the matrix-row dim."""
    jqt.set_device_mesh(shape=(2,), axis_names=("mp",))
    a = jqt.identity(8)
    spec = a.data.sharding.spec
    assert spec[0] == "mp"
    assert spec[1] is None


def test_mp_skips_batch_dim_when_present():
    """A batched (B, N, N) with axis_names=('mp',) prefers the matrix-row over batch."""
    jqt.set_device_mesh(shape=(2,), axis_names=("mp",))
    a = jqt.destroy(4)
    H_batch_data = jnp.stack([(a.dag() @ a).data, (a + a.dag()).data])  # (2, 4, 4)
    H = jqt.Qarray.create(H_batch_data, bdims=(2,))
    spec = _padded_spec(H.data)
    assert spec == (None, "mp", None)


# ---------------------------------------------------------------------------
# 2D mesh: both modes simultaneously
# ---------------------------------------------------------------------------

@needs_8
def test_2d_mesh_dense_uses_both_axes():
    """No batch dim → both mesh axes fall through to matrix dims."""
    jqt.set_device_mesh(shape=(2, 4), axis_names=("dp", "mp"))
    a = jqt.identity(8)
    spec = _padded_spec(a.data)
    # Both mesh axes consumed; order depends on priority ordering.
    assert {spec[0], spec[1]} == {"dp", "mp"}


@needs_8
def test_2d_mesh_batched_uses_both_modes():
    """Batched (B, N, N): 'dp' → batch axis, 'mp' → matrix-row."""
    jqt.set_device_mesh(shape=(2, 4), axis_names=("dp", "mp"))
    a = jqt.destroy(8)
    H_batch_data = jnp.stack([(a.dag() @ a).data, (a + a.dag()).data])  # (2, 8, 8)
    H = jqt.Qarray.create(H_batch_data, bdims=(2,))
    spec = _padded_spec(H.data)
    assert spec == ("dp", "mp", None)


@needs_8
def test_2d_mesh_ket_replicates_unused_axis():
    """Ket (N, 1) on (2, 4) mesh: only one mesh axis can bind; the other is unused."""
    jqt.set_device_mesh(shape=(2, 4), axis_names=("dp", "mp"))
    psi = jqt.basis(8, 0)
    spec = _padded_spec(psi.data)
    assert sum(1 for s in spec if s is not None) == 1


@needs_8
def test_2d_mesh_batched_ket():
    """Batched ket (B, N, 1): 'dp' → batch, 'mp' → matrix-row."""
    jqt.set_device_mesh(shape=(2, 4), axis_names=("dp", "mp"))
    psi_batch_data = jnp.stack([jqt.basis(8, 0).data, jqt.basis(8, 1).data])  # (2, 8, 1)
    psi = jqt.Qarray.create(psi_batch_data, bdims=(2,))
    spec = _padded_spec(psi.data)
    assert spec == ("dp", "mp", None)


@needs_8
def test_2d_mesh_matmul_smoke():
    """End-to-end: H @ psi with both sharded across a 2D mesh runs without error."""
    jqt.set_device_mesh(shape=(2, 4), axis_names=("dp", "mp"))
    H = jqt.destroy(8) + jqt.create(8)
    psi = jqt.basis(8, 0)
    out = H @ psi
    assert out.data.shape == (8, 1)


# ---------------------------------------------------------------------------
# End-to-end scientific workflows
# ---------------------------------------------------------------------------

def test_dp_e2e_parameter_sweep():
    """Data-parallel parameter sweep — vary the Kerr coefficient across a batch."""
    jqt.set_device_mesh(shape=(2,), axis_names=("dp",))
    N = 8
    a = jqt.destroy(N)
    ad = jqt.create(N)
    Ks = jnp.array([0.005, 0.01, 0.02, 0.05])

    H_data = jax.vmap(lambda K: K * (ad @ ad @ a @ a).data)(Ks)  # (4, N, N)
    H = jqt.Qarray.create(H_data, bdims=(4,))
    assert H.data.sharding.spec[0] == "dp"

    # Batched eigvalsh; sharding survives the trace.
    w = jax.vmap(jnp.linalg.eigvalsh)(H.data)
    assert w.shape == (4, N)
    assert len(w.sharding.device_set) == 2


def test_mp_e2e_large_system_eigh():
    """Model-parallel single-system eigendecomposition: a large (N, N) Hermitian."""
    jqt.set_device_mesh(shape=(2,), axis_names=("mp",))
    N = 32
    a = jqt.destroy(N)
    ad = jqt.create(N)
    H = (ad @ a) + 0.1 * (a + ad) + 0.01 * (ad @ ad @ a @ a)
    assert H.data.sharding.spec[0] == "mp"

    w, v = jnp.linalg.eigh(H.data)
    assert w.shape == (N,)
    assert v.shape == (N, N)
    assert len(v.sharding.device_set) == 2


def test_mp_e2e_sesolve():
    """Model-parallel sesolve over a Kerr Hamiltonian (large single system)."""
    jqt.set_device_mesh(shape=(2,), axis_names=("mp",))
    N = 16
    a = jqt.destroy(N)
    ad = jqt.create(N)
    H = 0.01 * (ad @ ad @ a @ a)
    psi0 = jqt.basis(N, 1)
    ts = jnp.linspace(0.0, 1.0, 6)
    states = jqt.sesolve(H, psi0, ts)
    assert states.data.shape == (6, N, 1)
    assert len(states.data.sharding.device_set) == 2


@needs_8
def test_2d_mesh_e2e_batched_sesolve():
    """2D mesh end-to-end: a parameter sweep where each system is large enough
    to also benefit from intra-system sharding."""
    jqt.set_device_mesh(shape=(2, 4), axis_names=("dp", "mp"))
    N = 16
    a = jqt.destroy(N)
    ad = jqt.create(N)
    Ks = jnp.array([0.005, 0.02])

    H_data = jax.vmap(lambda K: (K * (ad @ ad @ a @ a)).data)(Ks)  # (2, N, N)
    H = jqt.Qarray.create(H_data, bdims=(2,))
    psi0_data = jnp.broadcast_to(jqt.basis(N, 1).data, (2, N, 1))
    psi0 = jqt.Qarray.create(psi0_data, bdims=(2,))

    spec_H = H.data.sharding.spec
    spec_psi = psi0.data.sharding.spec
    assert spec_H[0] == "dp" and spec_H[1] == "mp"
    assert spec_psi[0] == "dp" and spec_psi[1] == "mp"

    ts = jnp.linspace(0.0, 1.0, 4)
    states = jax.vmap(
        lambda H_, psi_: jqt.sesolve(
            jqt.Qarray.create(H_), jqt.Qarray.create(psi_), ts
        ).data
    )(H.data, psi0.data)
    assert states.shape == (2, 4, N, 1)
    assert len(states.sharding.device_set) == 8

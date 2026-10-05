"""Tests for 1-D (N,) ket/bra storage and the ``qtype`` create parameter.

These cover the migration from storing kets/bras as (N,1)/(1,N) matrices to
storing them as (N,) vectors with the ket/bra/oper distinction held in the
``_qdims`` metadata.
"""

import jax
import jax.numpy as jnp
import pytest

import jaxquantum as jqt
from jaxquantum.core.qarray import Qtypes, QarrayImplType

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# create(): storage convention, legacy inputs, and the qtype parameter
# ---------------------------------------------------------------------------

def test_create_1d_defaults_to_ket():
    a = jqt.Qarray.create(jnp.array([1.0, 0, 0, 0]))
    assert a.qtype == Qtypes.ket
    assert a.data.shape == (4,)
    assert a.dims == ((4,), (1,))


def test_create_legacy_column_is_ket():
    a = jqt.Qarray.create(jnp.array([[1.0], [0], [0], [0]]))
    assert a.qtype == Qtypes.ket
    assert a.data.shape == (4,)


def test_create_legacy_row_is_bra():
    a = jqt.Qarray.create(jnp.array([[1.0, 0, 0, 0]]))
    assert a.qtype == Qtypes.bra
    assert a.data.shape == (4,)


def test_create_qtype_text_ket_bra():
    k = jqt.Qarray.create(jnp.array([1.0, 0, 0, 0]), qtype="ket")
    b = jqt.Qarray.create(jnp.array([1.0, 0, 0, 0]), qtype="bra")
    assert k.qtype == Qtypes.ket and k.data.shape == (4,)
    assert b.qtype == Qtypes.bra and b.data.shape == (4,)


def test_create_qtype_on_legacy_column():
    # (N,1) + qtype="ket" behaves like a plain (N,) ket.
    a = jqt.Qarray.create(jnp.array([[1.0], [0], [0]]), qtype="ket")
    assert a.qtype == Qtypes.ket and a.data.shape == (3,)


def test_create_square_oper_default():
    a = jqt.Qarray.create(jnp.eye(4))
    assert a.qtype == Qtypes.oper and a.data.shape == (4, 4)


def test_create_square_batched_ket_with_qtype_and_bdims():
    # A square (N, N) array is an operator by default; with qtype + bdims it is
    # a batch of N kets.
    data = jnp.eye(3)
    a = jqt.Qarray.create(data, qtype="ket", bdims=(3,))
    assert a.qtype == Qtypes.ket
    assert a.bdims == (3,)
    assert a.data.shape == (3, 3)


def test_create_multimode_ket_flat_dims():
    a = jqt.Qarray.create(jnp.ones(6) / jnp.sqrt(6), dims=[2, 3], qtype="ket")
    assert a.qtype == Qtypes.ket
    assert a.dims == ((2, 3), (1, 1))
    assert a.data.shape == (6,)


def test_create_incompatible_qtype_raises():
    # 1-D data cannot be an operator.
    with pytest.raises(ValueError):
        jqt.Qarray.create(jnp.array([1.0, 0, 0, 0]), qtype="oper")
    # dims claim 12 elements but data has 5.
    with pytest.raises(ValueError):
        jqt.Qarray.create(jnp.arange(5.0), dims=[3, 4], qtype="ket")
    # qtype conflicts with an explicit ket dims tuple.
    with pytest.raises(ValueError):
        jqt.Qarray.create(jnp.eye(4), dims=((4,), (1,)), qtype="oper")


# ---------------------------------------------------------------------------
# Invariant: stored vectors never carry a trailing singleton
# ---------------------------------------------------------------------------

def _assert_vec_storage(q):
    """A ket/bra must store its space on a single trailing axis."""
    assert q.qtype in (Qtypes.ket, Qtypes.bra)
    assert q.data.shape == q.bdims + (q.data.shape[-1],)
    assert q.data.shape[-1] != 1 or q.dims[0] == (1,) or q.dims[1] == (1,)


def test_invariant_no_trailing_singleton_on_common_ops():
    N = 5
    a = jqt.destroy(N)
    psi = jqt.basis(N, 1)
    _assert_vec_storage(psi)
    _assert_vec_storage(psi.dag())
    _assert_vec_storage(a @ psi)
    _assert_vec_storage((psi + jqt.basis(N, 2)).unit())
    _assert_vec_storage(jqt.coherent(N, 0.5))
    # batched
    kets = jqt.Qarray.from_list([jqt.basis(N, i) for i in range(3)])
    assert kets.data.shape == (3, N)
    _assert_vec_storage(a @ kets)
    _assert_vec_storage(kets.dag())


# ---------------------------------------------------------------------------
# Matmul: all five qtype combinations, batched and unbatched, values checked
# ---------------------------------------------------------------------------

def test_matmul_oper_ket():
    N = 4
    a = jqt.destroy(N)
    psi = jqt.basis(N, 2)
    out = a @ psi
    assert out.qtype == Qtypes.ket and out.data.shape == (N,)
    assert jnp.allclose(out.data, jnp.sqrt(2.0) * jqt.basis(N, 1).data)


def test_matmul_bra_oper():
    N = 4
    a = jqt.destroy(N)
    out = jqt.basis(N, 1).dag() @ a
    assert out.qtype == Qtypes.bra and out.data.shape == (N,)
    # <1| a = <2| * sqrt(2)
    assert jnp.allclose(out.data, jnp.sqrt(2.0) * jqt.basis(N, 2).dag().data)


def test_matmul_ket_bra_outer():
    N = 3
    op = jqt.basis(N, 0) @ jqt.basis(N, 1).dag()
    assert op.qtype == Qtypes.oper and op.data.shape == (N, N)
    expected = jnp.zeros((N, N), dtype=jnp.complex128).at[0, 1].set(1.0)
    assert jnp.allclose(op.data, expected)


def test_matmul_bra_ket_inner():
    N = 4
    ip = jqt.basis(N, 1).dag() @ jqt.basis(N, 1)
    assert jnp.allclose(ip.data.reshape(-1)[0], 1.0)
    ip0 = jqt.basis(N, 1).dag() @ jqt.basis(N, 2)
    assert jnp.allclose(ip0.data.reshape(-1)[0], 0.0)


def test_matmul_oper_oper():
    N = 4
    a, ad = jqt.destroy(N), jqt.create(N)
    n = ad @ a
    assert n.qtype == Qtypes.oper and n.data.shape == (N, N)
    assert jnp.allclose(n.data, jnp.diag(jnp.arange(N).astype(jnp.complex128)))


# ---------------------------------------------------------------------------
# Batched matrix x vector arithmetic (the explicit requirement)
# ---------------------------------------------------------------------------

def test_batched_oper_at_batched_ket():
    N = 4
    a, ad = jqt.destroy(N), jqt.create(N)
    H = jqt.Qarray.from_list([a + ad, a @ ad, ad @ a])  # (3, N, N)
    kets = jqt.Qarray.from_list([jqt.basis(N, i) for i in range(3)])  # (3, N)
    out = H @ kets
    assert out.qtype == Qtypes.ket and out.data.shape == (3, N)
    ref = jnp.stack([
        ((a + ad) @ jqt.basis(N, 0)).data,
        ((a @ ad) @ jqt.basis(N, 1)).data,
        ((ad @ a) @ jqt.basis(N, 2)).data,
    ])
    assert jnp.allclose(out.data, ref)


def test_broadcast_single_oper_at_batched_ket():
    N = 4
    a, ad = jqt.destroy(N), jqt.create(N)
    H = a + ad
    kets = jqt.Qarray.from_list([jqt.basis(N, i) for i in range(3)])
    out = H @ kets
    assert out.data.shape == (3, N)
    ref = jnp.stack([(H @ jqt.basis(N, i)).data for i in range(3)])
    assert jnp.allclose(out.data, ref)


def test_batched_oper_at_single_ket():
    N = 4
    a, ad = jqt.destroy(N), jqt.create(N)
    H = jqt.Qarray.from_list([a + ad, a @ ad])  # (2, N, N)
    psi = jqt.basis(N, 1)
    out = H @ psi
    assert out.data.shape == (2, N)
    ref = jnp.stack([((a + ad) @ psi).data, ((a @ ad) @ psi).data])
    assert jnp.allclose(out.data, ref)


def test_batched_ket_outer_product():
    N = 3
    kets = jqt.Qarray.from_list([jqt.basis(N, 0), jqt.basis(N, 1)])
    dms = kets @ kets.dag()
    assert dms.qtype == Qtypes.oper and dms.data.shape == (2, N, N)
    assert jnp.allclose(dms.data[0], jqt.basis(N, 0).to_dm().data)
    assert jnp.allclose(dms.data[1], jqt.basis(N, 1).to_dm().data)


def test_batched_inner_product():
    N = 4
    a = jqt.Qarray.from_list([jqt.basis(N, 0), jqt.basis(N, 1)])
    b = jqt.Qarray.from_list([jqt.basis(N, 0), jqt.basis(N, 2)])
    ip = a.dag() @ b
    vals = ip.data.reshape(2, -1)[:, 0]
    assert jnp.allclose(vals, jnp.array([1.0, 0.0]))


def test_batched_add_sub_scalar_mul():
    N = 3
    kets = jqt.Qarray.from_list([jqt.basis(N, 0), jqt.basis(N, 1)])
    s = kets + kets
    assert s.data.shape == (2, N) and jnp.allclose(s.data, 2 * kets.data)
    d = kets - kets
    assert d.data.shape == (2, N) and jnp.allclose(d.data, 0.0)
    m = jnp.array([2.0, 3.0]) * kets
    assert m.data.shape == (2, N)
    assert jnp.allclose(m.data[0], 2.0 * kets.data[0])
    assert jnp.allclose(m.data[1], 3.0 * kets.data[1])


def test_oper_plus_scalar_stays_oper():
    N = 3
    a = jqt.num(N)
    out = a + 2.0
    assert out.qtype == Qtypes.oper
    assert jnp.allclose(out.data, jnp.diag(jnp.arange(N).astype(jnp.complex128)) + 2.0 * jnp.eye(N))


# ---------------------------------------------------------------------------
# from_list with (N,) and (N,1) sourced kets + qtype kwarg
# ---------------------------------------------------------------------------

def test_from_list_of_kets_from_1d_and_legacy():
    k1 = jqt.Qarray.create(jnp.array([1.0, 0, 0]))            # (N,)
    k2 = jqt.Qarray.create(jnp.array([[0.0], [1.0], [0.0]]))  # legacy (N,1)
    batched = jqt.Qarray.from_list([k1, k2])
    assert batched.qtype == Qtypes.ket
    assert batched.bdims == (2,)
    assert batched.data.shape == (2, 3)
    assert jnp.allclose(batched.data[0], k1.data)
    assert jnp.allclose(batched.data[1], k2.data)


def test_from_list_qtype_kwarg():
    k1 = jqt.basis(4, 0)
    k2 = jqt.basis(4, 1)
    batched = jqt.Qarray.from_list([k1, k2], qtype="ket")
    assert batched.qtype == Qtypes.ket and batched.data.shape == (2, 4)


# ---------------------------------------------------------------------------
# Round-trips: norm/unit/dag/ket2dm/overlap/fidelity/tensor/eigenstates/solvers
# ---------------------------------------------------------------------------

def test_norm_and_unit_batched():
    N = 5
    kets = jqt.Qarray.from_list([jqt.basis(N, 0), 3.0 * jqt.basis(N, 1)])
    norms = kets.norm()
    assert norms.shape == (2,)
    assert jnp.allclose(norms, jnp.array([1.0, 3.0]))
    u = kets.unit()
    assert jnp.allclose(u.norm(), jnp.ones(2))


def test_eigenstates_are_kets():
    N = 4
    a, ad = jqt.destroy(N), jqt.create(N)
    H = ad @ a
    evals, evecs = jqt.eigenstates(H)
    assert evecs.qtype == Qtypes.ket
    assert evecs.data.shape == (N, N)  # batch of N kets of dim N
    assert evecs.bdims == (N,)


def test_sesolve_round_trip_shapes():
    N = 4
    H = jqt.create(N) @ jqt.destroy(N)
    ts = jnp.linspace(0.0, 1.0, 5)
    res = jqt.sesolve(H, jqt.basis(N, 1), ts)
    assert res.qtype == Qtypes.ket and res.data.shape == (5, N)


def test_propagator_round_trip_shapes():
    N = 4
    H = jqt.create(N) @ jqt.destroy(N)
    ts = jnp.linspace(0.0, 1.0, 3)
    U = jqt.propagator(lambda t: H, ts)
    assert U.qtype == Qtypes.oper and U.data.shape == (3, N, N)


def test_tensor_of_kets_and_opers():
    k = jqt.basis(2, 0) ^ jqt.basis(3, 1)
    assert k.qtype == Qtypes.ket and k.data.shape == (6,)
    o = jqt.identity(2) ^ jqt.num(3)
    assert o.qtype == Qtypes.oper and o.data.shape == (6, 6)


def test_overlap_and_fidelity():
    N = 6
    assert jnp.allclose(jqt.overlap(jqt.basis(N, 0), jqt.basis(N, 0)), 1.0)
    assert jnp.allclose(jqt.overlap(jqt.basis(N, 0), jqt.basis(N, 1)), 0.0)
    rho = jqt.coherent(N, 0.5).to_dm()
    assert jnp.allclose(jqt.fidelity(rho, rho, force_positivity=True), 1.0, atol=1e-6)


def test_qutip_round_trip_values():
    # Compare against QuTiP to ensure values (not just shapes) are correct.
    qt = pytest.importorskip("qutip")
    N = 8
    psi = jqt.coherent(N, 1.0 + 0.3j)
    a = jqt.destroy(N)
    out = (a @ psi)
    qt_psi = jqt.jqt2qt(psi)
    qt_out = qt.destroy(N) * qt_psi
    assert jnp.allclose(out.data, jnp.array(qt_out.full()).reshape(-1), atol=1e-8)

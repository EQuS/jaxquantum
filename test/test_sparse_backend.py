"""Tests for the sparse backend (Option A migration).

Coverage map
------------
SparseImpl._eye()           → TestSparseEye
SparseImpl.trace()          → TestSparseTrace (non-batched + batched)
SparseImpl.keep_only_diag() → TestSparseKeepDiag
SparseImpl.l2_norm_batched()→ TestSparseL2Norm
SparseImpl.tidy_up()        → TestSparseTidyUp
Qarray.from_sparse()        → TestFromSparse
Qarray.__eq__()  (sparse)   → TestSparseEquality
Qarray.__add__/__sub__
  scalar+identity            → TestSparseScalarAddSub
tr()                        → TestTraceFn
keep_only_diag_elements()   → TestKeepDiagFn
norm()                      → TestNormFn
operators with impl param   → TestSparseOperators
"""

import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jaxquantum as jqt
import jax.numpy as jnp
from jax.experimental import sparse
from jaxquantum.core.qarray import QarrayImplType, SparseImpl, DenseImpl, Qarray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def todense(arr):
    """Return a dense JAX array regardless of whether input is BCOO or dense."""
    return arr.todense() if hasattr(arr, "todense") else arr


def make_sparse_op(data=None):
    if data is None:
        data = jnp.array([[1.0, 0.0, 0.0],
                          [0.0, 2.0, 0.0],
                          [0.0, 0.0, 3.0]])
    return jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)


def make_dense_op(data=None):
    if data is None:
        data = jnp.array([[1.0, 0.0, 0.0],
                          [0.0, 2.0, 0.0],
                          [0.0, 0.0, 3.0]])
    return jqt.Qarray.create(data)


# ===========================================================================
# SparseImpl._eye_data()
# ===========================================================================

class TestSparseEye:
    def test_identity_values(self):
        data = SparseImpl._eye_data(4, jnp.float64)
        dense = data.todense()
        assert jnp.allclose(dense, jnp.eye(4))

    def test_shape(self):
        data = SparseImpl._eye_data(5, jnp.float64)
        assert data.shape == (5, 5)

    def test_is_sparse_impl(self):
        data = SparseImpl._eye_data(3, jnp.float64)
        assert isinstance(data, sparse.BCOO)

    def test_stores_n_entries(self):
        """The BCOO should store exactly N non-zero entries (diagonal only)."""
        n = 6
        data = SparseImpl._eye_data(n, jnp.float64)
        assert data.data.shape[0] == n


# ===========================================================================
# SparseImpl.trace()
# ===========================================================================

class TestSparseTrace:
    # ---- non-batched ----

    def test_trace_diagonal(self):
        data = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        assert jnp.allclose(impl.trace(), 6.0)

    def test_trace_off_diagonal_ignored(self):
        data = jnp.array([[1.0, 99.0], [99.0, 2.0]])
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        assert jnp.allclose(impl.trace(), 3.0)

    def test_trace_complex(self):
        data = jnp.array([[1+2j, 0.0], [0.0, 3+4j]])
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        assert jnp.allclose(impl.trace(), 4+6j)

    def test_trace_zero_matrix(self):
        impl = SparseImpl(sparse.BCOO.fromdense(jnp.zeros((3, 3))))
        assert jnp.allclose(impl.trace(), 0.0)

    def test_trace_identity(self):
        data = SparseImpl._eye_data(5, jnp.float64)
        impl = SparseImpl(data)
        assert jnp.allclose(impl.trace(), 5.0)

    def test_trace_matches_dense(self):
        data = jnp.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]])
        impl_sparse = SparseImpl(sparse.BCOO.fromdense(data))
        assert jnp.allclose(impl_sparse.trace(), jnp.trace(data))

    # ---- batched ----

    def test_trace_batched_1d(self):
        """Batched (B, N, N) → shape (B,) trace."""
        data = jnp.array([
            [[1.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [0.0, 4.0]],
        ])  # shape (2, 2, 2)
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        traces = impl.trace()
        assert traces.shape == (2,)
        assert jnp.allclose(traces[0], 3.0)
        assert jnp.allclose(traces[1], 7.0)

    def test_trace_batched_off_diag_ignored(self):
        data = jnp.array([
            [[1.0, 9.0], [9.0, 2.0]],
            [[3.0, 9.0], [9.0, 4.0]],
        ])
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        traces = impl.trace()
        assert jnp.allclose(traces, jnp.array([3.0, 7.0]))

    def test_trace_batched_complex(self):
        data = jnp.array([
            [[1+1j, 0.0], [0.0, 2+2j]],
        ])
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        traces = impl.trace()
        assert traces.shape == (1,)
        assert jnp.allclose(traces[0], 3+3j)


# ===========================================================================
# SparseImpl.keep_only_diag()
# ===========================================================================

class TestSparseKeepDiagImpl:
    def test_diagonal_preserved(self):
        data = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        result = impl.keep_only_diag()
        assert jnp.allclose(result._data.todense(), data)

    def test_off_diagonal_zeroed(self):
        data = jnp.array([[1.0, 5.0, 6.0],
                          [7.0, 2.0, 8.0],
                          [9.0, 10.0, 3.0]])
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        result = impl.keep_only_diag()
        dense = result._data.todense()
        assert jnp.allclose(jnp.diag(dense), jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(dense - jnp.diag(jnp.diag(dense)), 0.0)

    def test_returns_sparse_impl(self):
        data = jnp.eye(3)
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        assert isinstance(impl.keep_only_diag(), SparseImpl)

    def test_indices_unchanged(self):
        """BCOO indices should not change — only values are masked."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        result = impl.keep_only_diag()
        assert impl._data.indices.shape == result._data.indices.shape
        assert jnp.all(impl._data.indices == result._data.indices)

    def test_complex_diagonal_preserved(self):
        data = jnp.array([[1+2j, 0.0], [0.0, 3+4j]])
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        result = impl.keep_only_diag()
        assert jnp.allclose(result._data.todense(), data)


# ===========================================================================
# SparseImpl.l2_norm_batched()
# ===========================================================================

class TestSparseL2Norm:
    def test_non_batched_ket(self):
        data = jnp.array([[3.0], [4.0]])  # L2 norm = 5
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        assert jnp.allclose(impl.l2_norm_batched(()), 5.0)

    def test_non_batched_zero_ket(self):
        data = jnp.zeros((4, 1))
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        assert jnp.allclose(impl.l2_norm_batched(()), 0.0)

    def test_non_batched_matches_jnp_norm(self):
        data = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        expected = jnp.linalg.norm(data)
        assert jnp.allclose(impl.l2_norm_batched(()), expected)

    def test_batched_1d_ket(self):
        """Batched (B, N, 1) → per-batch norms of shape (B,)."""
        data = jnp.array([[[3.0], [4.0]],   # norm = 5
                          [[0.0], [1.0]]])   # norm = 1
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        norms = impl.l2_norm_batched((2,))
        assert norms.shape == (2,)
        assert jnp.allclose(norms[0], 5.0)
        assert jnp.allclose(norms[1], 1.0)

    def test_batched_complex(self):
        data = jnp.array([[[1+1j], [0.0]]])  # |1+1j|^2 = 2, norm = sqrt(2)
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        norms = impl.l2_norm_batched((1,))
        assert jnp.allclose(norms[0], jnp.sqrt(2.0))

    def test_non_batched_unit_vector(self):
        data = jnp.array([[1.0], [0.0], [0.0]])
        impl = SparseImpl(sparse.BCOO.fromdense(data))
        assert jnp.allclose(impl.l2_norm_batched(()), 1.0)


# ===========================================================================
# SparseImpl.tidy_up()
# ===========================================================================

class TestSparseTidyUp:
    def test_large_values_preserved(self):
        data = jnp.array([[1.0 + 2.0j, 0.0], [0.0, 3.0 + 4.0j]])
        q = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        assert jnp.allclose(todense(q.data), data)

    def test_small_values_zeroed(self):
        atol = jqt.SETTINGS["auto_tidyup_atol"]
        tiny = atol * 0.5
        data = jnp.array([[1.0, tiny], [0.0, 2.0]])
        q = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        dense = todense(q.data)
        assert dense[0, 1] == 0.0
        assert dense[0, 0] == 1.0
        assert dense[1, 1] == 2.0

    def test_real_imag_filtered_independently(self):
        atol = jqt.SETTINGS["auto_tidyup_atol"]
        tiny = atol * 0.5
        big = 1.0
        data = jnp.array([[big + tiny * 1j, 0.0], [0.0, 0.0]], dtype=jnp.complex128)
        q = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        dense = todense(q.data)
        assert jnp.real(dense[0, 0]) == big
        assert jnp.imag(dense[0, 0]) == 0.0

    def test_preserves_sparse_type(self):
        data = jnp.array([[1.0, 0.0], [0.0, 2.0]])
        q = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        assert q.is_sparse

    def test_tidy_up_direct(self):
        raw = sparse.BCOO.fromdense(jnp.array([[1.0, 1e-20], [0.0, 2.0]]))
        impl = SparseImpl(raw)
        tidied = impl.tidy_up(1e-14)
        assert isinstance(tidied, SparseImpl)
        assert tidied._data.todense()[0, 1] == 0.0
        assert tidied._data.todense()[0, 0] == 1.0


# ===========================================================================
# Qarray.from_sparse()  — no densification
# ===========================================================================

class TestFromSparse:
    def test_from_bcoo_directly(self):
        """from_sparse must accept BCOO without calling .todense()."""
        dense_data = jnp.array([[1.0, 0.0], [0.0, 2.0]])
        bcoo = sparse.BCOO.fromdense(dense_data)
        q = Qarray.from_sparse(bcoo)
        assert q.is_sparse
        assert jnp.allclose(q.data.todense(), dense_data)

    def test_from_sparse_dims_inferred(self):
        dense_data = jnp.eye(4)
        bcoo = sparse.BCOO.fromdense(dense_data)
        q = Qarray.from_sparse(bcoo)
        assert q.dims == ((4,), (4,))

    def test_from_sparse_explicit_dims(self):
        dense_data = jnp.eye(4)
        bcoo = sparse.BCOO.fromdense(dense_data)
        q = Qarray.from_sparse(bcoo, dims=[[2, 2], [2, 2]])
        assert q.dims == ((2, 2), (2, 2))

    def test_from_sparse_round_trip(self):
        """Create sparse, extract BCOO, wrap back — data must be identical."""
        data = jnp.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
        q1 = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        q2 = Qarray.from_sparse(q1.data)
        assert q2.is_sparse
        assert jnp.allclose(q2.data.todense(), q1.data.todense())


# ===========================================================================
# Qarray.__eq__()  — sparse comparison without full densification
# ===========================================================================

class TestSparseEquality:
    def test_sparse_equals_itself(self):
        q = make_sparse_op()
        assert q == q

    def test_sparse_equals_copy(self):
        data = jnp.array([[1.0, 0.0], [0.0, 2.0]])
        q1 = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        q2 = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        assert q1 == q2

    def test_sparse_not_equal_different_data(self):
        q1 = jqt.Qarray.create(jnp.eye(3), implementation=QarrayImplType.SPARSE)
        q2 = jqt.Qarray.create(2.0 * jnp.eye(3), implementation=QarrayImplType.SPARSE)
        assert q1 != q2

    def test_sparse_eq_different_dims_false(self):
        q1 = jqt.Qarray.create(jnp.eye(2), implementation=QarrayImplType.SPARSE)
        q2 = jqt.Qarray.create(jnp.eye(3), implementation=QarrayImplType.SPARSE)
        assert q1 != q2

    def test_to_sparse_idempotent(self):
        q = make_sparse_op()
        assert q.to_sparse() == q

    def test_sparse_dense_cross_equality(self):
        """Comparing a sparse Qarray to its dense twin still works (mixed case)."""
        data = jnp.array([[1.0, 0.0], [0.0, 2.0]])
        q_sparse = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        q_dense = jqt.Qarray.create(data)
        assert q_sparse == q_dense


# ===========================================================================
# Qarray.__add__ / __sub__ scalar — no dense N×N intermediate for sparse
# ===========================================================================

class TestSparseScalarAddSub:
    def test_scalar_add_stays_sparse(self):
        q = make_sparse_op()
        assert (q + 1.0).is_sparse

    def test_scalar_radd_stays_sparse(self):
        q = make_sparse_op()
        assert (1.0 + q).is_sparse

    def test_scalar_sub_stays_sparse(self):
        q = make_sparse_op()
        assert (q - 1.0).is_sparse

    def test_scalar_rsub_stays_sparse(self):
        q = make_sparse_op()
        assert (1.0 - q).is_sparse

    def test_scalar_add_correct_value(self):
        q_s = make_sparse_op()
        q_d = make_dense_op()
        scalar = 1.5
        assert jnp.allclose(todense((q_s + scalar).data), (q_d + scalar).data)

    def test_scalar_sub_correct_value(self):
        q_s = make_sparse_op()
        q_d = make_dense_op()
        scalar = 0.5
        assert jnp.allclose(todense((q_s - scalar).data), (q_d - scalar).data)

    def test_zero_add_is_noop(self):
        q = make_sparse_op()
        assert (q + 0) == q

    def test_complex_scalar_add_stays_sparse(self):
        q = make_sparse_op()
        assert (q + (1.0 + 2.0j)).is_sparse

    def test_neg_scalar_sub_stays_sparse(self):
        q = make_sparse_op()
        assert (q - (-2.0)).is_sparse

    def test_scalar_add_large_sparse(self):
        """For large N the _eye path avoids a dense N×N intermediate — just verify correctness."""
        N = 50
        n_s = jqt.num(N, implementation=QarrayImplType.SPARSE)
        n_d = jqt.num(N)
        result_s = todense((n_s + 1.0).data)
        result_d = (n_d + 1.0).data
        assert jnp.allclose(result_s, result_d)


# ===========================================================================
# tr() — no densification for sparse
# ===========================================================================

class TestTraceFn:
    def test_tr_diagonal_sparse(self):
        data = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        q = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        assert jnp.allclose(jqt.tr(q), 6.0)

    def test_tr_matches_dense(self):
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        q_s = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        q_d = jqt.Qarray.create(data)
        assert jnp.allclose(jqt.tr(q_s), jqt.tr(q_d))

    def test_tr_complex_sparse(self):
        data = jnp.array([[1+2j, 0.0], [0.0, 3+4j]])
        q_s = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        q_d = jqt.Qarray.create(data)
        assert jnp.allclose(jqt.tr(q_s), jqt.tr(q_d))

    def test_tr_method_sparse(self):
        data = jnp.diag(jnp.array([2.0, 5.0]))
        q = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        assert jnp.allclose(q.tr(), 7.0)

    def test_tr_identity_sparse(self):
        N = 5
        q = jqt.identity(N, implementation=QarrayImplType.SPARSE)
        assert jnp.allclose(jqt.tr(q), N)

    def test_tr_num_operator(self):
        N = 6
        n_s = jqt.num(N, implementation=QarrayImplType.SPARSE)
        n_d = jqt.num(N)
        assert jnp.allclose(jqt.tr(n_s), jqt.tr(n_d))

    def test_tr_off_diagonal_not_counted(self):
        data = jnp.array([[1.0, 99.0, 99.0],
                          [99.0, 2.0, 99.0],
                          [99.0, 99.0, 3.0]])
        q = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        assert jnp.allclose(jqt.tr(q), 6.0)

    def test_tr_batched_sparse(self):
        """tr() on batched sparse Qarray returns per-batch traces."""
        data = jnp.array([
            [[1.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [0.0, 4.0]],
        ])
        q = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        traces = jqt.tr(q)
        assert jnp.allclose(traces[0], 3.0)
        assert jnp.allclose(traces[1], 7.0)

    def test_trace_alias(self):
        data = jnp.diag(jnp.array([1.0, 2.0]))
        q = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        assert jnp.allclose(q.trace(), 3.0)


# ===========================================================================
# keep_only_diag_elements() — no densification for sparse
# ===========================================================================

class TestKeepDiagFn:
    def test_sparse_stays_sparse(self):
        q = jqt.Qarray.create(jnp.ones((3, 3)), implementation=QarrayImplType.SPARSE)
        result = jqt.keep_only_diag_elements(q)
        assert result.is_sparse

    def test_correct_values(self):
        data = jnp.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]])
        q_s = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        q_d = jqt.Qarray.create(data)
        result_s = todense(jqt.keep_only_diag_elements(q_s).data)
        result_d = jqt.keep_only_diag_elements(q_d).data
        assert jnp.allclose(result_s, result_d)

    def test_off_diag_zeroed(self):
        data = jnp.array([[1.0, 99.0], [99.0, 2.0]])
        q = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        result = todense(jqt.keep_only_diag_elements(q).data)
        assert result[0, 1] == 0.0
        assert result[1, 0] == 0.0
        assert result[0, 0] == 1.0
        assert result[1, 1] == 2.0

    def test_complex_diagonal_preserved(self):
        data = jnp.array([[1+2j, 0.0], [0.0, 3+4j]])
        q = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        result = todense(jqt.keep_only_diag_elements(q).data)
        assert jnp.allclose(result, data)

    def test_num_operator_keeps_diagonal(self):
        N = 5
        n = jqt.num(N, implementation=QarrayImplType.SPARSE)
        result = jqt.keep_only_diag_elements(n)
        assert result.is_sparse
        assert jnp.allclose(todense(result.data), todense(n.data))

    def test_dense_unchanged(self):
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        q = jqt.Qarray.create(data)
        result = jqt.keep_only_diag_elements(q)
        assert result.is_dense
        assert jnp.allclose(result.data, jnp.diag(jnp.array([1.0, 4.0])))


# ===========================================================================
# norm() — sparse paths
# ===========================================================================

class TestNormFn:
    # ---- ket / bra (l2_norm_batched path) ----

    def test_norm_ket_sparse(self):
        ket = jqt.Qarray.create(jnp.array([3.0, 4.0]), implementation=QarrayImplType.SPARSE)
        assert jnp.allclose(jqt.norm(ket), 5.0)

    def test_norm_ket_sparse_matches_dense(self):
        data = jnp.array([1.0, 2.0, 3.0])
        ket_s = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        ket_d = jqt.Qarray.create(data)
        assert jnp.allclose(jqt.norm(ket_s), jqt.norm(ket_d))

    def test_norm_bra_sparse(self):
        data = jnp.array([3.0, 4.0])
        ket = jqt.Qarray.create(data)
        bra_s = ket.dag().to_sparse()
        bra_d = ket.dag()
        assert jnp.allclose(jqt.norm(bra_s), jqt.norm(bra_d))

    def test_norm_basis_sparse(self):
        ket = jqt.basis(5, 2, implementation=QarrayImplType.SPARSE)
        assert jnp.allclose(jqt.norm(ket), 1.0)

    def test_unit_sparse_ket(self):
        ket = jqt.Qarray.create(jnp.array([3.0, 4.0]), implementation=QarrayImplType.SPARSE)
        normalized = jqt.unit(ket)
        assert jnp.allclose(jqt.norm(normalized), 1.0)

    def test_norm_batched_sparse_ket(self):
        """Batched sparse ket: per-element L2 norms."""
        data = jnp.array([[[3.0], [4.0]],   # norm = 5
                          [[0.0], [1.0]]])   # norm = 1
        ket_s = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        ket_d = jqt.Qarray.create(data)
        norms_s = jqt.norm(ket_s)
        norms_d = jqt.norm(ket_d)
        assert jnp.allclose(norms_s, norms_d)

    # ---- operator (trace path) ----

    def test_norm_identity_sparse(self):
        """Identity matrix trace norm = N."""
        N = 4
        eye = jqt.identity(N, implementation=QarrayImplType.SPARSE)
        assert jnp.allclose(jqt.norm(eye), float(N))

    def test_norm_density_matrix_sparse(self):
        """A normalised density matrix has trace (and hence norm) = 1."""
        ket = jqt.basis(4, 1, implementation=QarrayImplType.SPARSE)
        rho = jqt.ket2dm(ket)
        assert rho.is_sparse
        assert jnp.allclose(jqt.norm(rho), 1.0)

    def test_norm_oper_sparse_matches_dense(self):
        data = jnp.diag(jnp.array([0.5, 0.3, 0.2]))  # trace = 1 (density matrix)
        op_s = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        op_d = jqt.Qarray.create(data)
        # Both should equal the trace = 1.0
        assert jnp.allclose(jqt.norm(op_s), jqt.norm(op_d))

    def test_norm_num_operator_sparse(self):
        """num operator: trace norm should equal tr(num) via sparse path."""
        N = 5
        n_s = jqt.num(N, implementation=QarrayImplType.SPARSE)
        n_d = jqt.num(N)
        assert jnp.allclose(jqt.norm(n_s), jqt.norm(n_d))


# ===========================================================================
# operators.py — implementation parameter
# ===========================================================================

class TestSparseOperators:
    def _check(self, sparse_op, dense_op):
        assert sparse_op.is_sparse
        assert dense_op.is_dense
        assert jnp.allclose(todense(sparse_op.data), dense_op.data)

    def test_destroy(self):
        self._check(jqt.destroy(5, implementation=QarrayImplType.SPARSE), jqt.destroy(5))

    def test_destroy_string_impl(self):
        assert jqt.destroy(4, implementation="sparse").is_sparse

    def test_create(self):
        self._check(jqt.create(5, implementation=QarrayImplType.SPARSE), jqt.create(5))

    def test_num(self):
        self._check(jqt.num(6, implementation=QarrayImplType.SPARSE), jqt.num(6))

    def test_identity(self):
        self._check(jqt.identity(4, implementation=QarrayImplType.SPARSE), jqt.identity(4))

    def test_identity_like(self):
        ref = jqt.num(3, implementation=QarrayImplType.SPARSE)
        self._check(
            jqt.identity_like(ref, implementation=QarrayImplType.SPARSE),
            jqt.identity_like(ref),
        )

    def test_basis(self):
        self._check(
            jqt.basis(5, 2, implementation=QarrayImplType.SPARSE),
            jqt.basis(5, 2),
        )

    def test_defaults_still_dense(self):
        assert jqt.destroy(4).is_dense
        assert jqt.create(4).is_dense
        assert jqt.num(4).is_dense
        assert jqt.identity(4).is_dense
        assert jqt.basis(4, 0).is_dense

    def test_ladder_commutation_sparse(self):
        """[a, a†] = I (within truncation) — verify numerically in sparse."""
        N = 6
        a = jqt.destroy(N, implementation=QarrayImplType.SPARSE)
        adag = jqt.create(N, implementation=QarrayImplType.SPARSE)
        comm = a @ adag - adag @ a
        comm_dense = todense(comm.data)
        assert jnp.allclose(comm_dense[:-1, :-1], jnp.eye(N)[:-1, :-1])

    def test_num_eigenvalues_sparse(self):
        N = 5
        n = jqt.num(N, implementation=QarrayImplType.SPARSE)
        evals = n.eigenenergies()
        assert jnp.allclose(evals, jnp.arange(N, dtype=jnp.float64))


# ===========================================================================
# __getitem__ preserves impl type  (bug fix 1b)
# ===========================================================================

class TestGetitem:
    def test_batched_sparse_getitem_stays_sparse(self):
        """Indexing a batched sparse Qarray must return a sparse Qarray."""
        data = jnp.array([jnp.eye(3), 2.0 * jnp.eye(3)])  # shape (2, 3, 3)
        q = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        assert q.is_sparse
        q0 = q[0]
        assert q0.is_sparse, "getitem dropped the sparse implementation"

    def test_batched_dense_getitem_stays_dense(self):
        """Indexing a batched dense Qarray must return a dense Qarray."""
        data = jnp.array([jnp.eye(3), 2.0 * jnp.eye(3)])
        q = jqt.Qarray.create(data)
        assert q.is_dense
        assert q[0].is_dense

    def test_getitem_correct_values(self):
        """The indexed sub-Qarray must have the correct values."""
        data = jnp.array([jnp.eye(3), 2.0 * jnp.eye(3)])
        q = jqt.Qarray.create(data, implementation=QarrayImplType.SPARSE)
        assert jnp.allclose(todense(q[0].data), jnp.eye(3))
        assert jnp.allclose(todense(q[1].data), 2.0 * jnp.eye(3))

    def test_nonbatched_getitem_raises(self):
        """Indexing a non-batched Qarray must raise ValueError."""
        q = jqt.Qarray.create(jnp.eye(3), implementation=QarrayImplType.SPARSE)
        import pytest as _pytest
        with _pytest.raises(ValueError):
            _ = q[0]


# ===========================================================================
# from_list with sparse input  (bug fix 1c)
# ===========================================================================

class TestFromListSparse:
    def test_from_list_sparse_stays_sparse(self):
        """from_list on all-sparse inputs should return a sparse Qarray."""
        kets = [jqt.basis(4, k, implementation=QarrayImplType.SPARSE) for k in range(3)]
        batch = jqt.Qarray.from_list(kets)
        assert batch.is_sparse

    def test_from_list_sparse_correct_values(self):
        """Values stacked from sparse inputs must match dense stacking."""
        kets_s = [jqt.basis(4, k, implementation=QarrayImplType.SPARSE) for k in range(4)]
        kets_d = [jqt.basis(4, k) for k in range(4)]
        batch_s = jqt.Qarray.from_list(kets_s)
        batch_d = jqt.Qarray.from_list(kets_d)
        assert jnp.allclose(todense(batch_s.data), batch_d.data)

    def test_from_list_dense_unchanged(self):
        """from_list on dense inputs still returns a dense Qarray."""
        ops = [jqt.num(3) for _ in range(2)]
        batch = jqt.Qarray.from_list(ops)
        assert batch.is_dense

    def test_from_list_empty_works(self):
        """from_list([]) should not raise."""
        batch = jqt.Qarray.from_list([])
        assert batch is not None


# ===========================================================================
# tensor() with sparse inputs → dense output  (documented behaviour)
# ===========================================================================

class TestTensorSparse:
    def test_tensor_sparse_returns_dense(self):
        """tensor() always produces a dense Qarray, even from sparse inputs."""
        a = jqt.destroy(3, implementation=QarrayImplType.SPARSE)
        b = jqt.create(3, implementation=QarrayImplType.SPARSE)
        result = jqt.tensor(a, b)
        assert result.is_dense, "tensor() should return dense (documented limitation)"

    def test_tensor_sparse_correct_values(self):
        """Values from tensor() with sparse inputs must match the dense counterpart."""
        a_s = jqt.destroy(3, implementation=QarrayImplType.SPARSE)
        b_s = jqt.create(3, implementation=QarrayImplType.SPARSE)
        a_d = jqt.destroy(3)
        b_d = jqt.create(3)
        result_s = jqt.tensor(a_s, b_s)
        result_d = jqt.tensor(a_d, b_d)
        assert jnp.allclose(todense(result_s.data), result_d.data)


# ===========================================================================
# SparseImpl @ DenseImpl → dense, using native BCOO @ dense  (bug fix 1d)
# ===========================================================================

class TestMatmulSparseDense:
    def test_sparse_at_dense_returns_dense(self):
        """sparse_op @ dense_op must return a DenseImpl Qarray."""
        a_s = jqt.destroy(5, implementation=QarrayImplType.SPARSE)
        a_d = jqt.create(5)
        result = a_s @ a_d
        assert result.is_dense

    def test_dense_at_sparse_returns_dense(self):
        """dense_op @ sparse_op must return a DenseImpl Qarray."""
        a_d = jqt.destroy(5)
        a_s = jqt.create(5, implementation=QarrayImplType.SPARSE)
        result = a_d @ a_s
        assert result.is_dense

    def test_sparse_at_dense_correct_values(self):
        """Values of sparse @ dense must match the all-dense product."""
        N = 5
        a_s = jqt.destroy(N, implementation=QarrayImplType.SPARSE)
        adag_d = jqt.create(N)
        ref = jqt.destroy(N) @ jqt.create(N)  # all-dense reference
        result = a_s @ adag_d                  # sparse @ dense
        assert jnp.allclose(todense(result.data), ref.data)

    def test_sparse_at_dense_large(self):
        """Native BCOO @ dense path for larger N."""
        N = 20
        a = jqt.destroy(N, implementation=QarrayImplType.SPARSE)
        b = jqt.create(N)
        result = a @ b
        expected = jqt.destroy(N) @ jqt.create(N)
        assert jnp.allclose(todense(result.data), expected.data)

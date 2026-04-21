"""Tests for the SparseDIA (sparse diagonal) backend.

Coverage:
  - SparseDiaImpl construction and raw data access
  - dag() without densification
  - matmul: SparseDIA @ Dense, SparseDIA @ SparseDIA
  - add / sub: SparseDIA ± SparseDIA
  - scalar mul, neg
  - real, imag, conj
  - to_dense / to_sparse / to_sparse_dia conversions
  - kron: SparseDIA ⊗ SparseDIA
  - tidy_up
  - trace, frobenius_norm
  - _eye_data
  - can_handle_data / dag_data dispatch
  - SparseDiaData arithmetic (__mul__, __rmul__, __matmul__, __rmatmul__)
  - Qarray-level API (create, from_sparse_dia, is_sparse_dia, to_sparse_dia)
  - Operators with implementation="sparse_dia" (all operators)
  - Promotion: SparseDIA + Dense → Dense
  - mesolve with SparseDIA Hamiltonian and collapse operators
"""

import jax.numpy as jnp
import pytest

import jaxquantum as jqt
from jaxquantum.core.qarray import (
    DenseImpl,
    SparseImpl,
    QarrayImplType,
    dag_data,
)
from jaxquantum.core.sparse_dia import SparseDiaData, SparseDiaImpl, _dia_slice


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 6  # small dimension for most tests


def _dense_destroy(n):
    return jnp.diag(jnp.sqrt(jnp.arange(1, n, dtype=jnp.float64)), k=1)


def _dense_create(n):
    return jnp.diag(jnp.sqrt(jnp.arange(1, n, dtype=jnp.float64)), k=-1)


def _dense_num(n):
    return jnp.diag(jnp.arange(n, dtype=jnp.float64))


# ===========================================================================
# SparseDiaImpl construction
# ===========================================================================


class TestConstruction:
    def test_from_dense_array(self):
        a_dense = _dense_destroy(N)
        impl = SparseDiaImpl.from_data(jnp.array(a_dense))
        assert isinstance(impl, SparseDiaImpl)
        assert jnp.allclose(impl.to_dense()._data, a_dense)

    def test_from_sparse_dia_data(self):
        diags = jnp.zeros((1, N))
        diags = diags.at[0, 1:].set(jnp.sqrt(jnp.arange(1, N, dtype=jnp.float64)))
        data = SparseDiaData(offsets=(1,), diags=diags)
        impl = SparseDiaImpl.from_data(data)
        assert impl._offsets == (1,)
        assert jnp.allclose(impl._diags, diags)

    def test_from_diags_factory(self):
        diags = jnp.zeros((1, N))
        vals = jnp.arange(N, dtype=jnp.float64)
        diags = diags.at[0, :].set(vals)
        impl = SparseDiaImpl.from_diags(offsets=(0,), diags=diags)
        assert impl._offsets == (0,)

    def test_offsets_sorted(self):
        """from_diags should sort offsets regardless of input order."""
        diags = jnp.ones((2, N))
        impl = SparseDiaImpl.from_diags(offsets=(1, -1), diags=diags)
        assert impl._offsets == (-1, 1)

    def test_shape(self):
        impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        assert impl.shape() == (N, N)

    def test_memory_efficiency(self):
        """SparseDIA with 1 diagonal stores n values, not n²."""
        impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        assert impl._diags.size == N  # one padded diagonal of length N

    def test_get_data_returns_sparse_dia_data(self):
        impl = SparseDiaImpl.from_data(jnp.array(_dense_num(N)))
        data = impl.get_data()
        assert isinstance(data, SparseDiaData)
        assert data.offsets == impl._offsets


# ===========================================================================
# to_dense / to_sparse / to_sparse_dia
# ===========================================================================


class TestConversions:
    def test_to_dense_destroy(self):
        a_dense = _dense_destroy(N)
        impl = SparseDiaImpl.from_data(jnp.array(a_dense))
        assert jnp.allclose(impl.to_dense()._data, a_dense)

    def test_to_dense_num(self):
        n_dense = _dense_num(N)
        impl = SparseDiaImpl.from_data(jnp.array(n_dense))
        assert jnp.allclose(impl.to_dense()._data, n_dense)

    def test_to_dense_roundtrip(self):
        """Dense → SparseDIA → Dense should be identity."""
        mat = jnp.diag(jnp.array([1.0, 2.0, 3.0])) + jnp.diag(jnp.array([0.5, 0.5]), k=1)
        impl = SparseDiaImpl.from_data(mat)
        assert jnp.allclose(impl.to_dense()._data, mat)

    def test_to_sparse_is_bcoo(self):
        impl = SparseDiaImpl.from_data(jnp.array(_dense_num(N)))
        sparse_impl = impl.to_sparse()
        assert isinstance(sparse_impl, SparseImpl)

    def test_to_sparse_dia_is_self(self):
        impl = SparseDiaImpl.from_data(jnp.array(_dense_num(N)))
        assert impl.to_sparse_dia() is impl

    def test_from_dense_impl_to_sparse_dia(self):
        dense_impl = DenseImpl.from_data(_dense_destroy(N))
        dia_impl = dense_impl.to_sparse_dia()
        assert isinstance(dia_impl, SparseDiaImpl)
        assert jnp.allclose(dia_impl.to_dense()._data, _dense_destroy(N))


# ===========================================================================
# dag
# ===========================================================================


class TestDag:
    def test_dag_destroy_equals_create(self):
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        adag_impl = a_impl.dag()
        assert jnp.allclose(adag_impl.to_dense()._data, _dense_create(N))

    def test_dag_negates_offsets(self):
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        assert a_impl._offsets == (1,)
        adag_impl = a_impl.dag()
        assert adag_impl._offsets == (-1,)

    def test_dag_no_densification(self):
        """dag() result should be SparseDIA, not Dense."""
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        adag_impl = a_impl.dag()
        assert isinstance(adag_impl, SparseDiaImpl)

    def test_dag_num_is_self(self):
        """Number operator is Hermitian: n† = n."""
        n_impl = SparseDiaImpl.from_data(jnp.array(_dense_num(N)))
        ndag_impl = n_impl.dag()
        assert jnp.allclose(ndag_impl.to_dense()._data, _dense_num(N))

    def test_dag_double(self):
        """(A†)† = A."""
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        assert jnp.allclose(a_impl.dag().dag().to_dense()._data, a_impl.to_dense()._data)

    def test_dag_complex(self):
        """Conjugate is applied to values."""
        mat = 1j * jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        impl = SparseDiaImpl.from_data(mat)
        dagd = impl.dag().to_dense()._data
        assert jnp.allclose(dagd, -1j * jnp.diag(jnp.array([1.0, 2.0, 3.0])))


# ===========================================================================
# matmul
# ===========================================================================


class TestMatmul:
    def test_sparsedia_at_dense(self):
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        vec = jnp.eye(N)  # use identity as dense RHS
        result_impl = a_impl.matmul(DenseImpl.from_data(vec))
        assert isinstance(result_impl, DenseImpl)
        assert jnp.allclose(result_impl._data, _dense_destroy(N) @ vec)

    def test_sparsedia_at_sparsedia(self):
        """a† @ a should equal num operator."""
        adag_impl = SparseDiaImpl.from_data(jnp.array(_dense_create(N)))
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        result = adag_impl.matmul(a_impl)
        assert isinstance(result, SparseDiaImpl)
        assert jnp.allclose(result.to_dense()._data, _dense_num(N), atol=1e-10)

    def test_sparsedia_at_sparsedia_a_adag(self):
        """a @ a† matches the reference dense product (truncation sets last element to 0)."""
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        adag_impl = SparseDiaImpl.from_data(jnp.array(_dense_create(N)))
        result = a_impl.matmul(adag_impl)
        expected = _dense_destroy(N) @ _dense_create(N)
        assert jnp.allclose(result.to_dense()._data, expected, atol=1e-10)

    def test_matmul_result_stays_sparsedia(self):
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        result = a_impl.matmul(a_impl.dag())
        assert isinstance(result, SparseDiaImpl)


# ===========================================================================
# add / sub
# ===========================================================================


class TestAddSub:
    def test_add_two_sparsedia(self):
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        adag_impl = SparseDiaImpl.from_data(jnp.array(_dense_create(N)))
        result = a_impl.add(adag_impl)
        assert isinstance(result, SparseDiaImpl)
        expected = _dense_destroy(N) + _dense_create(N)
        assert jnp.allclose(result.to_dense()._data, expected)

    def test_add_union_of_offsets(self):
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        adag_impl = SparseDiaImpl.from_data(jnp.array(_dense_create(N)))
        result = a_impl.add(adag_impl)
        assert set(result._offsets) == {-1, 1}

    def test_sub(self):
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        adag_impl = SparseDiaImpl.from_data(jnp.array(_dense_create(N)))
        result = a_impl.sub(adag_impl)
        expected = _dense_destroy(N) - _dense_create(N)
        assert jnp.allclose(result.to_dense()._data, expected)

    def test_add_same_offset(self):
        """Adding two diagonal operators accumulates values."""
        n_impl = SparseDiaImpl.from_data(jnp.array(_dense_num(N)))
        result = n_impl.add(n_impl)
        assert jnp.allclose(result.to_dense()._data, 2 * _dense_num(N))


# ===========================================================================
# scalar mul
# ===========================================================================


class TestMul:
    def test_scalar_mul(self):
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        result = a_impl.mul(3.0)
        assert isinstance(result, SparseDiaImpl)
        assert jnp.allclose(result.to_dense()._data, 3.0 * _dense_destroy(N))

    def test_scalar_mul_preserves_offsets(self):
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        result = a_impl.mul(2.5)
        assert result._offsets == a_impl._offsets

    def test_neg(self):
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        neg = a_impl.neg()
        assert isinstance(neg, SparseDiaImpl)
        assert jnp.allclose(neg.to_dense()._data, -_dense_destroy(N))


# ===========================================================================
# real / imag / conj
# ===========================================================================


class TestRealImagConj:
    def test_real(self):
        mat = (1 + 1j) * jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        impl = SparseDiaImpl.from_data(mat)
        result = impl.real().to_dense()._data
        assert jnp.allclose(result, jnp.diag(jnp.array([1.0, 2.0, 3.0])))

    def test_imag(self):
        mat = (1 + 2j) * jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        impl = SparseDiaImpl.from_data(mat)
        result = impl.imag().to_dense()._data
        assert jnp.allclose(result, 2 * jnp.diag(jnp.array([1.0, 2.0, 3.0])))

    def test_conj(self):
        mat = 1j * jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        impl = SparseDiaImpl.from_data(mat)
        result = impl.conj().to_dense()._data
        assert jnp.allclose(result, -1j * jnp.diag(jnp.array([1.0, 2.0, 3.0])))


# ===========================================================================
# SparseDiaData arithmetic
# ===========================================================================


class TestSparseDiaDataArithmetic:
    def test_mul(self):
        data = SparseDiaData(offsets=(0,), diags=jnp.ones((1, 4)))
        result = data * 3.0
        assert isinstance(result, SparseDiaData)
        assert jnp.allclose(result.diags, 3.0 * jnp.ones((1, 4)))

    def test_rmul(self):
        data = SparseDiaData(offsets=(0,), diags=jnp.ones((1, 4)))
        result = 2.0 * data
        assert isinstance(result, SparseDiaData)
        assert jnp.allclose(result.diags, 2.0 * jnp.ones((1, 4)))

    def test_matmul_dense(self):
        """SparseDiaData @ dense array should give the correct product."""
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        raw = a_impl.get_data()
        vec = jnp.eye(N)
        result = raw @ vec
        assert jnp.allclose(result, _dense_destroy(N), atol=1e-10)

    def test_rmatmul_dense(self):
        """dense array @ SparseDiaData should give the correct product."""
        n_impl = SparseDiaImpl.from_data(jnp.array(_dense_num(N)))
        raw = n_impl.get_data()
        vec = jnp.eye(N)
        result = vec @ raw
        assert jnp.allclose(result, _dense_num(N), atol=1e-10)

    def test_shape_property(self):
        data = SparseDiaData(offsets=(0,), diags=jnp.ones((1, 5)))
        assert data.shape == (5, 5)

    def test_dtype_property(self):
        data = SparseDiaData(offsets=(0,), diags=jnp.ones((1, 4), dtype=jnp.float64))
        assert data.dtype == jnp.float64


# ===========================================================================
# kron
# ===========================================================================


class TestKron:
    def test_kron_identity_identity(self):
        eye_impl = SparseDiaImpl.from_data(jnp.eye(3))
        result = eye_impl.kron(eye_impl)
        assert isinstance(result, SparseDiaImpl)
        assert jnp.allclose(result.to_dense()._data, jnp.eye(9))

    def test_kron_destroy_identity(self):
        """a ⊗ I should give a Kronecker product."""
        a3 = SparseDiaImpl.from_data(jnp.array(_dense_destroy(3)))
        eye2 = SparseDiaImpl.from_data(jnp.eye(2))
        result = a3.kron(eye2)
        expected = jnp.kron(_dense_destroy(3), jnp.eye(2))
        assert jnp.allclose(result.to_dense()._data, expected, atol=1e-10)

    def test_kron_identity_destroy(self):
        eye3 = SparseDiaImpl.from_data(jnp.eye(3))
        a2 = SparseDiaImpl.from_data(jnp.array(_dense_destroy(2)))
        result = eye3.kron(a2)
        expected = jnp.kron(jnp.eye(3), _dense_destroy(2))
        assert jnp.allclose(result.to_dense()._data, expected, atol=1e-10)

    def test_kron_stays_sparsedia(self):
        a3 = SparseDiaImpl.from_data(jnp.array(_dense_destroy(3)))
        eye2 = SparseDiaImpl.from_data(jnp.eye(2))
        result = a3.kron(eye2)
        assert isinstance(result, SparseDiaImpl)

    def test_kron_shape(self):
        a3 = SparseDiaImpl.from_data(jnp.array(_dense_destroy(3)))
        eye2 = SparseDiaImpl.from_data(jnp.eye(2))
        result = a3.kron(eye2)
        assert result.shape() == (6, 6)


# ===========================================================================
# tidy_up
# ===========================================================================


class TestTidyUp:
    def test_tidy_up_zeros_small_values(self):
        diags = jnp.array([[1e-10, 1.0, 2.0, 3.0, 4.0, 5.0]])
        impl = SparseDiaImpl(_offsets=(0,), _diags=diags)
        tidied = impl.tidy_up(atol=1e-8)
        assert tidied._diags[0, 0] == 0.0
        assert tidied._diags[0, 1] == 1.0

    def test_tidy_up_preserves_format(self):
        impl = SparseDiaImpl.from_data(jnp.array(_dense_num(N)))
        tidied = impl.tidy_up(atol=1e-14)
        assert isinstance(tidied, SparseDiaImpl)
        assert tidied._offsets == impl._offsets


# ===========================================================================
# trace and frobenius_norm
# ===========================================================================


class TestTraceFrobenius:
    def test_trace_num(self):
        n_impl = SparseDiaImpl.from_data(jnp.array(_dense_num(N)))
        expected = float(sum(range(N)))
        assert jnp.allclose(n_impl.trace(), expected)

    def test_trace_off_diagonal(self):
        """Off-diagonal operator has zero trace."""
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        assert jnp.allclose(a_impl.trace(), 0.0)

    def test_frobenius_norm(self):
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        dense = _dense_destroy(N)
        expected = jnp.sqrt(jnp.sum(jnp.abs(dense) ** 2))
        assert jnp.allclose(a_impl.frobenius_norm(), expected)


# ===========================================================================
# _eye_data
# ===========================================================================


class TestEyeData:
    def test_eye_data_shape(self):
        data = SparseDiaImpl._eye_data(5, jnp.float64)
        assert data.shape == (5, 5)

    def test_eye_data_values(self):
        data = SparseDiaImpl._eye_data(4, jnp.float64)
        assert jnp.allclose(data, jnp.eye(4))


# ===========================================================================
# can_handle_data / dag_data dispatch
# ===========================================================================


class TestDispatch:
    def test_can_handle_data_true_for_sparsediadata(self):
        data = SparseDiaData(offsets=(0,), diags=jnp.ones((1, 4)))
        assert SparseDiaImpl.can_handle_data(data) is True

    def test_can_handle_data_false_for_dense(self):
        assert SparseDiaImpl.can_handle_data(jnp.ones((4, 4))) is False

    def test_dense_can_handle_excludes_sparsediadata(self):
        data = SparseDiaData(offsets=(0,), diags=jnp.ones((1, 4)))
        assert DenseImpl.can_handle_data(data) is False

    def test_dag_data_dispatch_sparsedia(self):
        """dag_data should route SparseDiaData to SparseDiaImpl.dag_data."""
        a_impl = SparseDiaImpl.from_data(jnp.array(_dense_destroy(N)))
        raw = a_impl.get_data()
        result_raw = dag_data(raw)
        assert isinstance(result_raw, SparseDiaData)
        result_impl = SparseDiaImpl(_offsets=result_raw.offsets, _diags=result_raw.diags)
        assert jnp.allclose(result_impl.to_dense()._data, _dense_create(N))


# ===========================================================================
# Qarray-level API
# ===========================================================================


class TestQarrayAPI:
    def test_create_with_sparse_dia(self):
        a = jqt.Qarray.create(jnp.array(_dense_destroy(N)), implementation="sparse_dia")
        assert a.is_sparse_dia
        assert not a.is_dense
        assert not a.is_sparse

    def test_create_with_enum(self):
        a = jqt.Qarray.create(
            jnp.array(_dense_destroy(N)),
            implementation=QarrayImplType.SPARSE_DIA,
        )
        assert a.is_sparse_dia

    def test_from_sparse_dia(self):
        a = jqt.Qarray.from_sparse_dia(jnp.array(_dense_destroy(N)))
        assert a.is_sparse_dia

    def test_to_sparse_dia_from_dense(self):
        a_dense = jqt.Qarray.create(jnp.array(_dense_destroy(N)))
        a_dia = a_dense.to_sparse_dia()
        assert a_dia.is_sparse_dia
        assert jnp.allclose(a_dia.to_dense().data, a_dense.data)

    def test_to_sparse_dia_idempotent(self):
        a = jqt.Qarray.from_sparse_dia(jnp.array(_dense_destroy(N)))
        assert a.to_sparse_dia() is a

    def test_qarray_matmul_sparsedia_dense(self):
        a = jqt.destroy(N, implementation=QarrayImplType.SPARSE_DIA)
        adag = jqt.create(N, implementation=QarrayImplType.DENSE)
        result = a @ adag
        # SparseDIA @ Dense → Dense
        assert result.is_dense
        # Reference: dense @ dense (truncation makes last diagonal element 0, not N)
        expected = jqt.destroy(N) @ jqt.create(N)
        assert jnp.allclose(result.data, expected.data, atol=1e-10)

    def test_scalar_add_works(self):
        """Scalar add uses _eye_data internally; verify it works for SparseDIA."""
        n_op = jqt.Qarray.create(jnp.array(_dense_num(N)), implementation="sparse_dia")
        result = n_op + 1.0
        expected = _dense_num(N) + jnp.eye(N)
        assert jnp.allclose(result.to_dense().data, expected, atol=1e-10)


# ===========================================================================
# Operators with implementation="sparse_dia"
# ===========================================================================


class TestOperators:
    def test_destroy_sparse_dia(self):
        a = jqt.destroy(N, implementation=QarrayImplType.SPARSE_DIA)
        assert a.is_sparse_dia
        assert jnp.allclose(a.to_dense().data, _dense_destroy(N))

    def test_create_sparse_dia(self):
        adag = jqt.create(N, implementation=QarrayImplType.SPARSE_DIA)
        assert adag.is_sparse_dia
        assert jnp.allclose(adag.to_dense().data, _dense_create(N))

    def test_num_sparse_dia(self):
        n = jqt.num(N, implementation=QarrayImplType.SPARSE_DIA)
        assert n.is_sparse_dia
        assert jnp.allclose(n.to_dense().data, _dense_num(N))

    def test_identity_sparse_dia(self):
        eye = jqt.identity(N, implementation=QarrayImplType.SPARSE_DIA)
        assert eye.is_sparse_dia
        assert jnp.allclose(eye.to_dense().data, jnp.eye(N))

    def test_destroy_memory(self):
        """destroy(N) as SparseDIA should store 1 diagonal of N elements, not N²."""
        a = jqt.destroy(N, implementation=QarrayImplType.SPARSE_DIA)
        assert a._impl._diags.size == N

    def test_sigmax_sparse_dia(self):
        sx = jqt.sigmax(implementation=QarrayImplType.SPARSE_DIA)
        assert sx.is_sparse_dia
        expected = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        assert jnp.allclose(sx.to_dense().data, expected)

    def test_sigmay_sparse_dia(self):
        sy = jqt.sigmay(implementation=QarrayImplType.SPARSE_DIA)
        assert sy.is_sparse_dia
        expected = jnp.array([[0.0, -1.0j], [1.0j, 0.0]])
        assert jnp.allclose(sy.to_dense().data, expected)

    def test_sigmaz_sparse_dia(self):
        sz = jqt.sigmaz(implementation=QarrayImplType.SPARSE_DIA)
        assert sz.is_sparse_dia
        expected = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        assert jnp.allclose(sz.to_dense().data, expected)

    def test_sigmap_sparse_dia(self):
        sp = jqt.sigmap(implementation=QarrayImplType.SPARSE_DIA)
        assert sp.is_sparse_dia
        expected = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        assert jnp.allclose(sp.to_dense().data, expected)

    def test_sigmam_sparse_dia(self):
        sm = jqt.sigmam(implementation=QarrayImplType.SPARSE_DIA)
        assert sm.is_sparse_dia
        expected = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        assert jnp.allclose(sm.to_dense().data, expected)

    def test_hadamard_sparse_dia(self):
        h = jqt.hadamard(implementation=QarrayImplType.SPARSE_DIA)
        assert h.is_sparse_dia
        expected = jnp.array([[1, 1], [1, -1]]) / jnp.sqrt(2.0)
        assert jnp.allclose(h.to_dense().data, expected)

    def test_sigmax_no_leading_zeros_wrong(self):
        """sigmax direct construction uses correct padding (not swapped rows)."""
        sx_dia = jqt.sigmax(implementation=QarrayImplType.SPARSE_DIA)
        sx_dense = jqt.sigmax(implementation=QarrayImplType.DENSE)
        assert jnp.allclose(sx_dia.to_dense().data, sx_dense.data)


# ===========================================================================
# Promotion
# ===========================================================================


class TestPromotion:
    def test_sparsedia_plus_dense_gives_dense(self):
        a_dia = jqt.destroy(N, implementation=QarrayImplType.SPARSE_DIA)
        adag_dense = jqt.create(N, implementation=QarrayImplType.DENSE)
        result = a_dia + adag_dense
        assert result.is_dense

    def test_sparsedia_plus_sparsedia_gives_sparsedia(self):
        a = jqt.destroy(N, implementation=QarrayImplType.SPARSE_DIA)
        adag = jqt.create(N, implementation=QarrayImplType.SPARSE_DIA)
        result = a + adag
        assert result.is_sparse_dia

    def test_promotion_order(self):
        assert SparseDiaImpl.PROMOTION_ORDER == 0
        assert SparseImpl.PROMOTION_ORDER == 1
        assert DenseImpl.PROMOTION_ORDER == 2


# ===========================================================================
# mesolve with SparseDIA
# ===========================================================================


class TestMesolve:
    def test_mesolve_sparsedia_hamiltonian(self):
        """mesolve with a SparseDIA Hamiltonian should match dense result."""
        N_modes = 4
        omega = 1.0
        H_dense = jqt.num(N_modes) * omega
        H_dia = jqt.num(N_modes, implementation=QarrayImplType.SPARSE_DIA) * omega
        opts = jqt.SolverOptions.create(progress_meter=False)

        rho0 = jqt.basis(N_modes, 1).to_dm()
        tlist = jnp.linspace(0, 1.0, 20)

        result_dense = jqt.mesolve(H_dense, rho0, tlist, solver_options=opts)
        result_dia = jqt.mesolve(H_dia, rho0, tlist, solver_options=opts)

        assert jnp.allclose(
            result_dense.data, result_dia.data, atol=1e-6
        ), "mesolve with SparseDIA H differs from dense reference"

    def test_mesolve_sparsedia_c_ops(self):
        """mesolve with SparseDIA collapse operators should match dense result."""
        N_modes = 4
        kappa = 0.1
        H = jqt.num(N_modes)
        opts = jqt.SolverOptions.create(progress_meter=False)

        c_dense = jqt.Qarray.from_list([jqt.destroy(N_modes) * jnp.sqrt(kappa)])
        c_dia = jqt.Qarray.from_list(
            [jqt.destroy(N_modes, implementation=QarrayImplType.SPARSE_DIA) * jnp.sqrt(kappa)]
        )

        rho0 = jqt.basis(N_modes, 2).to_dm()
        tlist = jnp.linspace(0, 2.0, 30)

        result_dense = jqt.mesolve(H, rho0, tlist, c_ops=c_dense, solver_options=opts)
        result_dia = jqt.mesolve(H, rho0, tlist, c_ops=c_dia, solver_options=opts)

        assert jnp.allclose(
            result_dense.data, result_dia.data, atol=1e-5
        ), "mesolve with SparseDIA c_ops differs from dense reference"


# ===========================================================================
# Batched SparseDIA (from_list + operations)
# ===========================================================================


class TestBatchedSparseDIA:
    def _make_batched(self):
        """Return a batched SparseDIA from [destroy, create] and the dense reference."""
        a = jqt.destroy(N, implementation=QarrayImplType.SPARSE_DIA)
        adag = jqt.create(N, implementation=QarrayImplType.SPARSE_DIA)
        batched = jqt.Qarray.from_list([a, adag])
        return batched, a, adag

    def test_from_list_stays_sparsedia(self):
        batched, _, _ = self._make_batched()
        assert batched.is_sparse_dia

    def test_from_list_union_offsets(self):
        """destroy (offset +1) and create (offset -1) → union is (-1, 1)."""
        batched, _, _ = self._make_batched()
        assert batched._impl._offsets == (-1, 1)

    def test_from_list_no_densification(self):
        """Batched array stores n_ops × n_union_diags × N, not n_ops × N²."""
        batched, _, _ = self._make_batched()
        # 2 operators, 2 union diagonals, N elements each
        assert batched._impl._diags.size == 2 * 2 * N

    def test_from_list_correct_values(self):
        """Each batch slice should equal its original operator."""
        batched, a, adag = self._make_batched()
        # Slice out first and second operators via Qarray indexing
        slice0 = batched[0]
        slice1 = batched[1]
        assert jnp.allclose(slice0.to_dense().data, a.to_dense().data, atol=1e-10)
        assert jnp.allclose(slice1.to_dense().data, adag.to_dense().data, atol=1e-10)

    def test_batched_at_single_sparsedia(self):
        """batched @ single_sparsedia → batched SparseDIA with correct values."""
        batched, a, adag = self._make_batched()
        n_op = jqt.num(N, implementation=QarrayImplType.SPARSE_DIA)
        result = batched @ n_op
        assert result.is_sparse_dia
        assert result.bdims == (2,)
        # a @ num and adag @ num
        expected0 = (a @ n_op).to_dense().data
        expected1 = (adag @ n_op).to_dense().data
        assert jnp.allclose(result[0].to_dense().data, expected0, atol=1e-10)
        assert jnp.allclose(result[1].to_dense().data, expected1, atol=1e-10)

    def test_batched_at_single_dense(self):
        """batched_sparsedia @ single_dense → batched Dense with correct values."""
        batched, a, adag = self._make_batched()
        n_op = jqt.num(N, implementation=QarrayImplType.DENSE)
        result = batched @ n_op
        assert result.is_dense
        assert result.bdims == (2,)
        expected0 = (a @ n_op).to_dense().data
        expected1 = (adag @ n_op).to_dense().data
        assert jnp.allclose(result[0].to_dense().data, expected0, atol=1e-10)
        assert jnp.allclose(result[1].to_dense().data, expected1, atol=1e-10)

    def test_batched_add_single(self):
        """batched + single → batched, bdims preserved, values correct."""
        batched, a, adag = self._make_batched()
        eye = jqt.identity(N, implementation=QarrayImplType.SPARSE_DIA)
        result = batched + eye
        assert result.bdims == (2,)
        expected0 = (a + eye).to_dense().data
        expected1 = (adag + eye).to_dense().data
        assert jnp.allclose(result[0].to_dense().data, expected0, atol=1e-10)
        assert jnp.allclose(result[1].to_dense().data, expected1, atol=1e-10)

    def test_batched_dag(self):
        """batched.dag() preserves bdims and conjugate-transposes each slice."""
        batched, a, adag = self._make_batched()
        result = batched.dag()
        assert result.bdims == (2,)
        # dag of destroy is create and vice versa
        assert jnp.allclose(result[0].to_dense().data, adag.to_dense().data, atol=1e-10)
        assert jnp.allclose(result[1].to_dense().data, a.to_dense().data, atol=1e-10)

    def test_batched_scalar_mul(self):
        """Scalar multiplication preserves bdims and scales each slice."""
        batched, a, adag = self._make_batched()
        result = 2.0 * batched
        assert result.bdims == (2,)
        assert jnp.allclose(result[0].to_dense().data, 2.0 * a.to_dense().data, atol=1e-10)
        assert jnp.allclose(result[1].to_dense().data, 2.0 * adag.to_dense().data, atol=1e-10)

    def test_mesolve_c_ops_batched_sparsedia(self):
        """mesolve with batched SparseDIA c_ops (destroy + dephasing) matches dense."""
        N_modes = 4
        kappa = 0.1
        gamma = 0.05
        H = jqt.num(N_modes)
        opts = jqt.SolverOptions.create(progress_meter=False)
        rho0 = jqt.basis(N_modes, 2).to_dm()
        tlist = jnp.linspace(0, 2.0, 30)

        # Two collapse operators with different diagonal structures
        c_dense = jqt.Qarray.from_list([
            jqt.destroy(N_modes) * jnp.sqrt(kappa),
            jqt.num(N_modes) * jnp.sqrt(gamma),
        ])
        c_dia = jqt.Qarray.from_list([
            jqt.destroy(N_modes, implementation=QarrayImplType.SPARSE_DIA) * jnp.sqrt(kappa),
            jqt.num(N_modes, implementation=QarrayImplType.SPARSE_DIA) * jnp.sqrt(gamma),
        ])
        assert c_dia.is_sparse_dia

        result_dense = jqt.mesolve(H, rho0, tlist, c_ops=c_dense, solver_options=opts)
        result_dia = jqt.mesolve(H, rho0, tlist, c_ops=c_dia, solver_options=opts)

        assert jnp.allclose(
            result_dense.data, result_dia.data, atol=1e-5
        ), "mesolve with batched SparseDIA c_ops differs from dense reference"


# ===========================================================================
# _dia_slice helper
# ===========================================================================


class TestDiaSlice:
    def test_positive_offset(self):
        assert _dia_slice(2) == slice(2, None)

    def test_negative_offset(self):
        assert _dia_slice(-2) == slice(None, -2)

    def test_zero_offset(self):
        assert _dia_slice(0) == slice(0, None)

    def test_complement_positive(self):
        """_dia_slice(-k) for k>0 gives the complementary slice."""
        assert _dia_slice(-3) == slice(None, -3)

    def test_complement_negative(self):
        assert _dia_slice(3) == slice(3, None)


# ===========================================================================
# powm — integer matrix power staying SparseDIA
# ===========================================================================


class TestPowm:
    def test_powm_zero_is_identity(self):
        a = jqt.destroy(N, implementation=QarrayImplType.SPARSE_DIA)
        result = a._impl.powm(0)
        assert isinstance(result, SparseDiaImpl)
        assert jnp.allclose(result.to_dense()._data, jnp.eye(N), atol=1e-10)

    def test_powm_one_is_self(self):
        a = jqt.destroy(N, implementation=QarrayImplType.SPARSE_DIA)
        result = a._impl.powm(1)
        assert jnp.allclose(result.to_dense()._data, a.to_dense().data, atol=1e-10)

    def test_powm_two_matches_dense(self):
        a = jqt.destroy(N, implementation=QarrayImplType.SPARSE_DIA)
        result = a._impl.powm(2)
        expected = _dense_destroy(N) @ _dense_destroy(N)
        assert jnp.allclose(result.to_dense()._data, expected, atol=1e-10)

    def test_powm_stays_sparsedia(self):
        a = jqt.destroy(N, implementation=QarrayImplType.SPARSE_DIA)
        for n in (0, 1, 2, 3, 4):
            result = a._impl.powm(n)
            assert isinstance(result, SparseDiaImpl), f"powm({n}) not SparseDIA"

    def test_powm_via_qarray(self):
        """Qarray.powm dispatches to SparseDiaImpl.powm and stays SparseDIA."""
        a = jqt.destroy(N, implementation=QarrayImplType.SPARSE_DIA)
        result = a.powm(3)
        assert result.is_sparse_dia
        expected = jqt.destroy(N).powm(3)
        assert jnp.allclose(result.to_dense().data, expected.to_dense().data, atol=1e-10)

    def test_powm_negative_raises(self):
        a = jqt.destroy(N, implementation=QarrayImplType.SPARSE_DIA)
        with pytest.raises(ValueError):
            a._impl.powm(-1)

    def test_powm_sparsedia_out_of_range_offsets_pruned(self):
        """a^k for large k: output offsets should be pruned to valid range."""
        a = jqt.destroy(N, implementation=QarrayImplType.SPARSE_DIA)
        result = a._impl.powm(N - 1)  # a^(N-1) is nearly nilpotent
        for k in result._offsets:
            assert abs(k) < N, f"out-of-range offset {k} found in powm result"

"""cuQuantum backend parity tests; run with ``pytest -o addopts=`` on a GPU host."""

# ruff: noqa: E402 -- JAX x64 must be enabled before importing cuQuantum.

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# cuquantum.densitymat.jax requires jax_enable_x64=True at import time;
# set it before pytest's importorskip probe.
import jax
jax.config.update("jax_enable_x64", True)

import pytest

# Skip the entire file if cuquantum (with the JAX extension) isn't installed.
pytest.importorskip(
    "cuquantum.densitymat.jax",
    reason="cuquantum not installed; skipping cuquantum backend tests",
)

import jax.numpy as jnp

import jaxquantum as jqt
from cuquantum.densitymat.jax import OperatorTerm
from jaxquantum.core.qarray import QarrayImplType
from jaxquantum.core.cuquantum_impl import CuquantumImpl, _cuqnt_dag

# Mark every test in this module so CI's `-m "not cuquantum"` excludes them.
pytestmark = pytest.mark.cuquantum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 3


def _dense(matrix):
    return jnp.asarray(matrix, dtype=jnp.complex128)


def _destroy_mat(n):
    return jnp.diag(jnp.sqrt(jnp.arange(1, n, dtype=jnp.complex128)), k=1)


def _create_mat(n):
    return jnp.diag(jnp.sqrt(jnp.arange(1, n, dtype=jnp.complex128)), k=-1)


# ===========================================================================
# from_data / round-trip
# ===========================================================================

class TestFromData:
    def test_single_site_round_trip(self):
        a = _destroy_mat(N)
        q = jqt.Qarray.create(a, implementation="cuquantum")
        assert q.impl_type == QarrayImplType.CUQUANTUM
        assert jnp.allclose(q.to_dense().data, a)

    def test_shape(self):
        q = jqt.Qarray.create(_destroy_mat(N), implementation="cuquantum")
        assert q.shape == (N, N)

    def test_identity_term(self):
        impl = CuquantumImpl.identity_term(N)
        assert jnp.allclose(impl.to_dense().get_data(), jnp.eye(N))

    def test_get_data_returns_operator_term(self):
        q = jqt.Qarray.create(_destroy_mat(N), implementation="cuquantum")
        d = q._impl.get_data()
        assert isinstance(d, OperatorTerm)
        assert d.dims == (N,)

    def test_operator_term_preserves_mode_dims(self):
        structured = jqt.tensor(
            jqt.destroy(N, implementation="cuquantum"),
            jqt.sigmax(implementation="cuquantum"),
        )
        restored = jqt.Qarray.create(
            structured._impl.get_data(), implementation="cuquantum"
        )
        assert restored.dims == structured.dims
        assert jnp.allclose(restored.to_dense().data, structured.to_dense().data)


# ===========================================================================
# Operator constructors
# ===========================================================================

class TestOperatorConstructors:
    """The CUQUANTUM branches added to operators.py should all match dense."""

    @pytest.mark.parametrize(
        "name",
        ["destroy", "create", "num"],
    )
    def test_creation_matches_dense(self, name):
        cu = getattr(jqt, name)(N, implementation="cuquantum")
        d = getattr(jqt, name)(N)
        assert jnp.allclose(cu.to_dense().data, d.data)

    @pytest.mark.parametrize(
        "name",
        ["sigmax", "sigmay", "sigmaz", "sigmap", "sigmam", "hadamard"],
    )
    def test_pauli_matches_dense(self, name):
        cu = getattr(jqt, name)(implementation="cuquantum")
        d = getattr(jqt, name)()
        assert jnp.allclose(cu.to_dense().data, d.data)

    def test_identity_matches_dense(self):
        cu = jqt.identity(N, implementation="cuquantum")
        assert jnp.allclose(cu.to_dense().data, jnp.eye(N))

    def test_identity_like_two_modes(self):
        H_ref = jqt.tensor(jqt.sigmaz(), jqt.sigmax())
        I_like = jqt.identity_like(H_ref, implementation="cuquantum")
        assert I_like.shape == (4, 4)
        assert jnp.allclose(I_like.to_dense().data, jnp.eye(4))

    def test_scaled_multimode_identity_stays_local(self):
        H = jqt.tensor(
            jqt.destroy(N, implementation="cuquantum"),
            jqt.destroy(N, implementation="cuquantum"),
        )
        scaled = 2 * jqt.identity_like(H, implementation="cuquantum")
        assert scaled._impl.get_data().op_prods[0][0].num_modes == 1
        assert jnp.allclose(scaled.to_dense().data, 2 * jnp.eye(N * N))


# ===========================================================================
# Arithmetic vs dense reference
# ===========================================================================

class TestArithmetic:
    def setup_method(self):
        self.a_mat = _destroy_mat(N)
        self.ad_mat = _create_mat(N)
        self.a = jqt.Qarray.create(self.a_mat, implementation="cuquantum")
        self.ad = jqt.Qarray.create(self.ad_mat, implementation="cuquantum")

    def test_add(self):
        H = self.a + self.ad
        assert jnp.allclose(H.to_dense().data, self.a_mat + self.ad_mat)

    def test_sub(self):
        D = self.a - self.ad
        assert jnp.allclose(D.to_dense().data, self.a_mat - self.ad_mat)

    def test_mul_left(self):
        scaled = 2.5 * self.a
        assert jnp.allclose(scaled.to_dense().data, 2.5 * self.a_mat)

    def test_mul_complex(self):
        scaled = self.a * (1.0 + 2.0j)
        assert jnp.allclose(scaled.to_dense().data, self.a_mat * (1.0 + 2.0j))

    def test_mixed_real_complex_arithmetic(self):
        x = jqt.sigmax(implementation="cuquantum")
        y = jqt.sigmay(implementation="cuquantum")
        x_ref, y_ref = jqt.sigmax().data, jqt.sigmay().data
        for actual, expected in (
            (x + y, x_ref + y_ref),
            (x - y, x_ref - y_ref),
            (x @ y, x_ref @ y_ref),
            (jqt.tensor(x, y), jnp.kron(x_ref, y_ref)),
        ):
            assert jnp.allclose(actual.to_dense().data, expected)

    def test_operator_times_dense_ket(self):
        ket = jqt.basis(N, 1)
        assert jnp.allclose((self.a @ ket).data, self.a_mat @ ket.data)

    def test_matmul_same_mode(self):
        # ad @ a on a 3-level system equals number operator
        n = self.ad @ self.a
        # cpu_device = jax.devices("cpu")[0]
        # a_mat = jax.device_put(_destroy_mat(3).astype(jnp.float64), cpu_device)
        # ad_mat = jax.device_put(_create_mat(3).astype(jnp.float64), cpu_device)
        # import numpy as np
        # import cupy as cp
        # expected = np.asarray(ad_mat) @ np.asarray(a_mat)

        expected = self.ad_mat @ self.a_mat
        assert jnp.allclose(n.to_dense().data, expected)

    def test_dag(self):
        adag = self.a.dag()
        assert jnp.allclose(adag.to_dense().data, self.ad_mat)

    def test_dag_double(self):
        # Double-dag should restore the original.
        assert jnp.allclose(self.a.dag().dag().to_dense().data, self.a_mat)

    def test_composed_adjoint_reverses_factors_and_modes(self):
        a = jqt.destroy(3, implementation="cuquantum")
        n = jqt.num(3, implementation="cuquantum")
        y = jqt.sigmay(implementation="cuquantum")
        product = jqt.tensor(a @ n, y)
        dagged = CuquantumImpl.from_data(_cuqnt_dag(product._impl.to_operator_term()))

        assert jnp.allclose(
            dagged.to_dense()._data,
            product.to_dense().data.conj().T,
        )


# ===========================================================================
# Kronecker / tensor product with mode bookkeeping
# ===========================================================================

class TestKron:
    def setup_method(self):
        self.a_mat = _destroy_mat(N)
        self.ad_mat = _create_mat(N)
        self.a = jqt.Qarray.create(self.a_mat, implementation="cuquantum")
        self.ad = jqt.Qarray.create(self.ad_mat, implementation="cuquantum")
        self.I = jqt.identity(N, implementation="cuquantum")

    def test_a_kron_I(self):
        out = jqt.tensor(self.a, self.I)
        expected = jnp.kron(self.a_mat, jnp.eye(N))
        assert jnp.allclose(out.to_dense().data, expected)

    def test_I_kron_ada(self):
        out = jqt.tensor(self.I, self.ad @ self.a)
        expected = jnp.kron(jnp.eye(N), self.ad_mat @ self.a_mat)
        assert jnp.allclose(out.to_dense().data, expected)

    def test_modeshift_matmul(self):
        # (a⊗I) @ (I⊗a) should equal a⊗a
        left = jqt.tensor(self.a, self.I)
        right = jqt.tensor(self.I, self.a)
        prod = left @ right
        expected = jnp.kron(self.a_mat, self.a_mat)
        assert jnp.allclose(prod.to_dense().data, expected)

    def test_three_mode(self):
        out = jqt.tensor(self.a, self.I, self.ad)
        expected = jnp.kron(jnp.kron(self.a_mat, jnp.eye(N)), self.ad_mat)
        assert jnp.allclose(out.to_dense().data, expected)


# ===========================================================================
# Solver parity (requires GPU)
# ===========================================================================

class TestSolverParity:
    """Compare cuquantum mesolve/sesolve against the dense backend.

    These need a CUDA device and are excluded from the default test run.
    """

    def test_time_dependent_hamiltonian(self):
        tlist = jnp.linspace(0, 0.5, 7)
        psi0 = jqt.basis(2, 0)
        options = jqt.SolverOptions(progress_meter=None)
        def dense_h(t):
            return jnp.sin(t) * jqt.sigmax()

        def cu_h(t):
            return jnp.sin(t) * jqt.sigmax(implementation="cuquantum")

        for solve, initial, dense_kwargs, cu_kwargs in (
            (jqt.sesolve, psi0, {}, {}),
            (
                jqt.mesolve, psi0.to_dm(),
                {"c_ops": jqt.Qarray.from_list([0.1 * jqt.sigmam()])},
                {"c_ops": [0.1 * jqt.sigmam(implementation="cuquantum")]},
            ),
        ):
            expected = solve(dense_h, initial, tlist, solver_options=options, **dense_kwargs)
            actual = solve(cu_h, initial, tlist, solver_options=options, **cu_kwargs)
            assert jnp.allclose(actual.data, expected.data, atol=1e-6)

    def test_sesolve_single_qubit_drive(self):
        # σx-driven qubit: ψ̇ = -i σx ψ, ψ(0) = |0>
        H_dense = 0.5 * jqt.sigmax()
        H_cu = 0.5 * jqt.sigmax(implementation="cuquantum")
        psi0 = jqt.basis(2, 0)
        tlist = jnp.linspace(0, 1.0, 11)
        opts = jqt.SolverOptions(progress_meter=None)

        ref = jqt.sesolve(H_dense, psi0, tlist, solver_options=opts)
        cu = jqt.sesolve(H_cu, psi0, tlist, solver_options=opts)

        assert jnp.allclose(cu.data, ref.data, atol=1e-6)

    def test_mesolve_amplitude_decay(self):
        # Single-qubit amplitude damping with σx drive.
        gamma = 0.05
        H_dense = 0.5 * jqt.sigmax()
        H_cu = 0.5 * jqt.sigmax(implementation="cuquantum")
        L_dense = jnp.sqrt(gamma) * jqt.sigmam()
        L_cu = jnp.sqrt(gamma) * jqt.sigmam(implementation="cuquantum")

        rho0 = jqt.basis(2, 0).to_dm()
        tlist = jnp.linspace(0, 1.0, 11)
        opts = jqt.SolverOptions(progress_meter=None)

        ref = jqt.mesolve(
            H_dense, rho0, tlist,
            c_ops=jqt.Qarray.from_list([L_dense]),
            solver_options=opts,
        )
        # cuquantum c_ops must be passed as a Python list — Qarray.from_list
        # densifies cuquantum impls (no batched OperatorTerm exists).
        cu = jqt.mesolve(
            H_cu, rho0, tlist,
            c_ops=[L_cu],
            solver_options=opts,
        )

        assert jnp.allclose(cu.data, ref.data, atol=1e-5)

    def test_mesolve_two_mode_collapse(self):
        h_dense = jqt.tensor(jqt.sigmaz(), jqt.sigmaz())
        h_cu = jqt.tensor(
            jqt.sigmaz(implementation="cuquantum"),
            jqt.sigmaz(implementation="cuquantum"),
        )
        l_dense = jqt.tensor(jqt.sigmam(), jqt.sigmaz())
        l_cu = jqt.tensor(
            jqt.sigmam(implementation="cuquantum"),
            jqt.sigmaz(implementation="cuquantum"),
        )
        rho0 = jqt.tensor(jqt.basis(2, 1), jqt.basis(2, 0)).to_dm()
        tlist = jnp.linspace(0, 0.2, 5)
        opts = jqt.SolverOptions(progress_meter=None)

        ref = jqt.mesolve(
            h_dense, rho0, tlist,
            c_ops=jqt.Qarray.from_list([l_dense]), solver_options=opts,
        )
        cu = jqt.mesolve(h_cu, rho0, tlist, c_ops=[l_cu], solver_options=opts)

        assert jnp.allclose(cu.data, ref.data, atol=1e-5)

    def test_mesolve_composed_single_mode_collapse(self):
        h_dense = jqt.num(3)
        h_cu = jqt.num(3, implementation="cuquantum")
        l_dense = jqt.destroy(3) @ jqt.num(3)
        l_cu = (
            jqt.destroy(3, implementation="cuquantum")
            @ jqt.num(3, implementation="cuquantum")
        )
        rho0 = jqt.basis(3, 2).to_dm()
        tlist = jnp.linspace(0, 0.2, 5)
        opts = jqt.SolverOptions(progress_meter=None)

        ref = jqt.mesolve(
            h_dense, rho0, tlist,
            c_ops=jqt.Qarray.from_list([l_dense]), solver_options=opts,
        )
        cu = jqt.mesolve(h_cu, rho0, tlist, c_ops=[l_cu], solver_options=opts)

        assert jnp.allclose(cu.data, ref.data, atol=1e-5)

    def test_mesolve_complex_h_real_collapse_matches_dense(self):
        gamma = 0.05
        L_dense = jnp.sqrt(gamma) * jqt.sigmam()
        L_cu = jnp.sqrt(gamma) * jqt.sigmam(implementation="cuquantum")
        H_dense = 0.5 * jqt.sigmay()
        H_cu = 0.5 * jqt.sigmay(implementation="cuquantum")
        rho0 = jqt.basis(2, 0).to_dm()
        tlist = jnp.linspace(0, 1.0, 11)
        opts = jqt.SolverOptions(progress_meter=None)

        ref = jqt.mesolve(
            H_dense, rho0, tlist,
            c_ops=jqt.Qarray.from_list([L_dense]),
            solver_options=opts,
        )
        cu = jqt.mesolve(H_cu, rho0, tlist, c_ops=[L_cu], solver_options=opts)

        assert jnp.allclose(cu.data, ref.data, atol=1e-5)

    def test_mesolve_real_h_complex_collapse_matches_dense(self):
        H_dense = 0.5 * jqt.sigmax()
        H_cu = 0.5 * jqt.sigmax(implementation="cuquantum")
        L_dense = 0.2 * jqt.sigmay()
        L_cu = 0.2 * jqt.sigmay(implementation="cuquantum")
        rho0 = jqt.basis(2, 0).to_dm()
        tlist = jnp.linspace(0, 0.5, 7)
        opts = jqt.SolverOptions(progress_meter=None)

        ref = jqt.mesolve(
            H_dense, rho0, tlist,
            c_ops=jqt.Qarray.from_list([L_dense]), solver_options=opts,
        )
        cu = jqt.mesolve(H_cu, rho0, tlist, c_ops=[L_cu], solver_options=opts)

        assert jnp.allclose(cu.data, ref.data, atol=1e-5)

    def test_mesolve_summed_collapse_keeps_cross_terms(self):
        H_dense = jqt.sigmax()
        H_cu = jqt.sigmax(implementation="cuquantum")
        L_dense = jqt.sigmam() + 0.2 * jqt.sigmaz()
        L_cu = (
            jqt.sigmam(implementation="cuquantum")
            + 0.2 * jqt.sigmaz(implementation="cuquantum")
        )
        rho0 = jqt.Qarray.create(
            jnp.array([[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]], dtype=jnp.complex128)
        )
        tlist = jnp.linspace(0, 0.2, 5)
        opts = jqt.SolverOptions(progress_meter=None)

        ref = jqt.mesolve(
            H_dense, rho0, tlist,
            c_ops=jqt.Qarray.from_list([L_dense]), solver_options=opts,
        )
        cu = jqt.mesolve(H_cu, rho0, tlist, c_ops=[L_cu], solver_options=opts)

        assert jnp.allclose(cu.data, ref.data, atol=1e-5)


# ===========================================================================
# Operator construction with implementation="cuquantum"
# ===========================================================================


class TestOperatorImplementationFlagCuquantum:
    """Composite constructors preserve the requested backend."""

    def test_displace_round_trips_to_cuquantum(self):
        # ``expm`` densifies internally; the result must still come back as
        # CUQUANTUM thanks to ``Qarray.to_backend``.
        ref = jqt.displace(N, 0.4)
        op = jqt.displace(N, 0.4, implementation="cuquantum")
        assert op.impl_type == QarrayImplType.CUQUANTUM
        assert jnp.allclose(op.to_dense().data, ref.to_dense().data)

    def test_squeeze_round_trips_to_cuquantum(self):
        ref = jqt.squeeze(N, 0.3)
        op = jqt.squeeze(N, 0.3, implementation="cuquantum")
        assert op.impl_type == QarrayImplType.CUQUANTUM
        assert jnp.allclose(op.to_dense().data, ref.to_dense().data)

    def test_qubit_rotation_cuquantum(self):
        ref = jqt.qubit_rotation(0.7, 1, 0, 0)
        op = jqt.qubit_rotation(0.7, 1, 0, 0, implementation="cuquantum")
        assert op.impl_type == QarrayImplType.CUQUANTUM
        assert jnp.allclose(op.to_dense().data, ref.to_dense().data)

    def test_thermal_dm_cuquantum(self):
        ref = jqt.thermal_dm(N, 0.5)
        op = jqt.thermal_dm(N, 0.5, implementation="cuquantum")
        assert op.impl_type == QarrayImplType.CUQUANTUM
        assert jnp.allclose(op.to_dense().data, ref.to_dense().data)

    def test_basis_falls_back_to_dense_for_cuquantum(self):
        # Kets aren't representable as cuquantum OperatorTerms — fall back.
        ket = jqt.basis(N, 1, implementation="cuquantum")
        assert ket.impl_type == QarrayImplType.DENSE

    def test_coherent_falls_back_to_dense_for_cuquantum(self):
        ket = jqt.coherent(N, 0.3, implementation="cuquantum")
        assert ket.impl_type == QarrayImplType.DENSE
        assert jnp.allclose(ket.data, jqt.coherent(N, 0.3).to_dense().data)

    def test_identity_like_cuquantum(self):
        a = jqt.tensor(jqt.destroy(N, implementation="cuquantum"),
                       jqt.identity(N, implementation="cuquantum"))
        eye = jqt.identity_like(a, implementation="cuquantum")
        assert eye.impl_type == QarrayImplType.CUQUANTUM
        assert jnp.allclose(eye.to_dense().data, jnp.eye(N * N))

    def test_default_backend_uses_structured_constructor(self):
        previous = jqt.SETTINGS["default_backend"]
        rotation_dense = jqt.qubit_rotation(0.7, 1, 0, 0, implementation="dense")
        try:
            jqt.SETTINGS["default_backend"] = "cuquantum"
            assert len(jqt.identity(N)._impl.get_data().op_prods) == 0
            assert jqt.displace(N, 0.4).impl_type == QarrayImplType.CUQUANTUM
            rotation = jqt.qubit_rotation(0.7, 1, 0, 0)
            assert rotation.impl_type == QarrayImplType.CUQUANTUM
            assert jnp.allclose(rotation.to_dense().data, rotation_dense.data)
        finally:
            jqt.SETTINGS["default_backend"] = previous


class TestToBackendCuquantum:
    """``to_backend('cuquantum')`` from non-cuquantum sources goes through
    dense (no direct ``to_cuquantum`` method exists)."""

    def test_dense_to_cuquantum(self):
        q = jqt.destroy(N)
        cu = q.to_backend("cuquantum")
        assert cu.impl_type == QarrayImplType.CUQUANTUM
        assert jnp.allclose(cu.to_dense().data, q.data)

    def test_sparse_dia_to_cuquantum(self):
        q = jqt.destroy(N, implementation="sparse_dia")
        cu = q.to_backend("cuquantum")
        assert cu.impl_type == QarrayImplType.CUQUANTUM
        assert jnp.allclose(cu.to_dense().data, q.to_dense().data)

    def test_cuquantum_to_dense(self):
        q = jqt.destroy(N, implementation="cuquantum")
        d = q.to_backend("dense")
        assert d.impl_type == QarrayImplType.DENSE
        assert jnp.allclose(d.data, q.to_dense().data)

    def test_noop_when_already_cuquantum(self):
        q = jqt.destroy(N, implementation="cuquantum")
        assert q.to_backend("cuquantum") is q

    def test_multimode_dense_conversion_fails_clearly(self):
        dense = jqt.tensor(jqt.destroy(2), jqt.sigmax())
        with pytest.raises(ValueError, match="build local operators first"):
            dense.to_backend("cuquantum")
        with pytest.raises(ValueError, match="build local operators first"):
            jqt.Qarray.create(dense.data, dims=dense.dims, implementation="cuquantum")

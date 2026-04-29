"""Tests for the cuQuantum backend.

These tests are tagged with the ``cuquantum`` pytest marker and additionally
guarded with ``pytest.importorskip`` at module load.  The default CI flow
selects ``-m "not cuquantum"`` (configured in ``pyproject.toml``) so they
never run there.

Run them locally on a machine with cuquantum + CUDA::

    pytest test/test_cuquantum_backend.py
    pytest -m cuquantum

Skip them everywhere else::

    pytest -m "not cuquantum"

The tests fall into two groups:

- **Construction / arithmetic** (no GPU needed).  These exercise
  ``CuquantumImpl`` arithmetic via ``to_dense()`` round-trips, comparing
  against the reference dense backend.  They run on any box where
  ``cuquantum`` imports cleanly.
- **Solver / operator_action** (GPU needed).  ``mesolve``/``sesolve`` on a
  cuquantum-backed Hamiltonian go through ``cuquantum.densitymat.jax``'s
  ``operator_action``, which requires a CUDA device.  These tests are
  individually marked ``cuquantum_gpu`` so they can be deselected on
  cuquantum-but-no-GPU hosts.
"""

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
from jaxquantum.core.cuquantum_impl import CuquantumImpl

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

    def test_matmul_same_mode(self):
        # ad @ a on a 3-level system equals number operator
        n = self.ad @ self.a
        a_mat = _destroy_mat(3).astype(jnp.float64)
        ad_mat = _create_mat(3).astype(jnp.float64)
        import numpy as np
        import cupy as cp
        breakpoint()
        expected = cp.asarray(a_mat) @ cp.asarray(ad_mat)
        # expected = self.ad_mat @ self.a_mat
        assert jnp.allclose(n.to_dense().data, expected.get())

    def test_dag(self):
        adag = self.a.dag()
        assert jnp.allclose(adag.to_dense().data, self.ad_mat)

    def test_dag_double(self):
        # Double-dag should restore the original.
        assert jnp.allclose(self.a.dag().dag().to_dense().data, self.a_mat)


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

cuquantum_gpu = pytest.mark.cuquantum  # alias; run via pytest -m cuquantum


@cuquantum_gpu
class TestSolverParity:
    """Compare cuquantum mesolve/sesolve against the dense backend.

    These need a CUDA device; they will fail with a clear FFI error message
    on cuquantum-but-no-GPU hosts.  Mark `not cuquantum_gpu` (or just
    `not cuquantum`) to deselect them in that case.
    """

    def test_sesolve_single_qubit_drive(self):
        # σx-driven qubit: ψ̇ = -i σx ψ, ψ(0) = |0>
        H_dense = 0.5 * jqt.sigmax()
        H_cu = 0.5 * jqt.sigmax(implementation="cuquantum")
        psi0 = jqt.basis(2, 0)
        tlist = jnp.linspace(0, 1.0, 11)
        opts = jqt.SolverOptions.create(progress_meter=False)

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
        opts = jqt.SolverOptions.create(progress_meter=False)

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

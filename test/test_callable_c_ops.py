"""Tests for callable time-dependent collapse operators.

These exercise the JAXQuantum + Diffrax solver, not an independent RK4 driver.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import jaxquantum as jqt


def _initial_density():
    return jqt.basis(2, 1).to_dm()


def _collapses(t, gamma0, amp, tau):
    rate = gamma0 + amp * jnp.exp(-t / tau)
    # destroy(2) = |0><1|, so an initial |1> must decay toward |0>.
    # JAXQuantum's sigmam() instead has matrix |1><0| in this basis.
    return jqt.Qarray.from_list([jnp.sqrt(rate) * jqt.destroy(2)])


def _analytic(t, gamma0, amp, tau):
    return jnp.exp(-gamma0 * t - amp * tau * (1 - jnp.exp(-t / tau)))


def _populations(gamma0, amp, tau):
    ts = jnp.linspace(0, 5, 51)
    result = jqt.mesolve(
        0.0,
        _initial_density(),
        ts,
        c_ops=lambda t: _collapses(t, gamma0, amp, tau),
        solver_options=jqt.SolverOptions(progress_meter=None),
    )
    return jnp.real(result.to_dense().data[..., 1, 1])


def test_callable_matches_nonstationary_decay():
    ts = jnp.linspace(0, 5, 51)
    p = _populations(0.2, 0.3, 1.5)
    np.testing.assert_allclose(
        p, _analytic(ts, 0.2, 0.3, 1.5), atol=2e-5, rtol=2e-5
    )


def test_callable_static_equivalence():
    ts = jnp.linspace(0, 5, 51)
    opts = jqt.SolverOptions(progress_meter=None)
    static = _collapses(0.0, 0.2, 0.0, 1.5)
    reference = jqt.mesolve(
        0.0, _initial_density(), ts, c_ops=static, solver_options=opts
    )
    dynamic = jqt.mesolve(
        0.0, _initial_density(), ts, c_ops=lambda t: static, solver_options=opts
    )
    np.testing.assert_allclose(
        dynamic.to_dense().data, reference.to_dense().data, atol=2e-5
    )
    # Constant callable/static equivalence must be nontrivial, not 1 == 1.
    np.testing.assert_allclose(
        jnp.real(reference.to_dense().data[..., 1, 1]),
        jnp.exp(-0.2 * ts),
        atol=2e-5,
        rtol=2e-5,
    )


def test_callable_jit_grad_and_batch():
    def final_population(a):
        return _populations(0.2, a, 1.5)[-1]

    value = jax.jit(final_population)(0.3)
    np.testing.assert_allclose(
        value, _analytic(5.0, 0.2, 0.3, 1.5), atol=2e-5, rtol=2e-5
    )
    amps = jnp.array([0.0, 0.1, 0.3])
    values = jax.jit(jax.vmap(final_population))(amps)
    np.testing.assert_allclose(
        values, _analytic(5.0, 0.2, amps, 1.5), atol=2e-5, rtol=2e-5
    )
    grad = jax.jit(jax.grad(final_population))(0.3)
    target_grad = (
        -1.5 * (1 - jnp.exp(-5.0 / 1.5)) * _analytic(5.0, 0.2, 0.3, 1.5)
    )
    np.testing.assert_allclose(
        grad, target_grad, atol=3e-5, rtol=2e-4
    )


def test_callable_pure_dephasing():
    """Nonstationary Z dephasing must suppress only the coherence."""
    ts = jnp.linspace(0.0, 4.0, 41)
    ground, excited = jqt.basis(2, 0), jqt.basis(2, 1)
    rho0 = (ground + excited).unit().to_dm()
    gamma0, amp, tau = 0.12, 0.22, 1.7

    def collapse(t):
        rate = gamma0 + amp * jnp.exp(-t / tau)
        return jqt.Qarray.from_list(
            [jnp.sqrt(rate / 2.0) * jqt.sigmaz()]
        )

    result = jqt.mesolve(
        0.0,
        rho0,
        ts,
        c_ops=collapse,
        solver_options=jqt.SolverOptions(progress_meter=None),
    ).to_dense().data
    integrated = gamma0 * ts + amp * tau * (1 - jnp.exp(-ts / tau))
    np.testing.assert_allclose(
        jnp.real(result[..., 0, 1]),
        0.5 * jnp.exp(-integrated),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        jnp.real(result[..., 1, 1]),
        0.5,
        rtol=2e-5,
        atol=2e-5,
    )


def test_callable_multiple_relaxation_and_dephasing():
    """Two dynamic collapse operators must combine their dissipators."""
    ts = jnp.linspace(0.0, 4.0, 41)
    ground, excited = jqt.basis(2, 0), jqt.basis(2, 1)
    rho0 = (ground + excited).unit().to_dm()
    t1, tp = 1.25, 2.0

    def collapse(t):
        gamma_1 = 0.12 + 0.20 * jnp.exp(-t / t1)
        gamma_phi = 0.06 + 0.10 * jnp.exp(-t / tp)
        return jqt.Qarray.from_list(
            [
                jnp.sqrt(gamma_1) * jqt.destroy(2),
                jnp.sqrt(gamma_phi / 2.0) * jqt.sigmaz(),
            ]
        )

    result = jqt.mesolve(
        0.0,
        rho0,
        ts,
        c_ops=collapse,
        solver_options=jqt.SolverOptions(progress_meter=None),
    ).to_dense().data

    integrated_1 = 0.12 * ts + 0.20 * t1 * (1 - jnp.exp(-ts / t1))
    integrated_phi = 0.06 * ts + 0.10 * tp * (1 - jnp.exp(-ts / tp))
    np.testing.assert_allclose(
        jnp.real(result[..., 1, 1]),
        0.5 * jnp.exp(-integrated_1),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        jnp.real(result[..., 0, 1]),
        0.5 * jnp.exp(-(0.5 * integrated_1 + integrated_phi)),
        rtol=2e-5,
        atol=2e-5,
    )


def test_callable_batched_operator_stack():
    """A single solve supports distinct rates along the operator batch axis."""
    times = jnp.linspace(0.0, 3.0, 31)
    base = jnp.array([0.12, 0.24, 0.35])
    amplitude = jnp.array([0.07, 0.11, 0.21])
    tau = 1.4
    lowering = jqt.destroy(2)

    def collapse(t):
        rates = base + amplitude * jnp.exp(-t / tau)
        return jqt.Qarray.from_list([jnp.sqrt(rates) * lowering])

    result = jqt.mesolve(
        0.0,
        _initial_density(),
        times,
        c_ops=collapse,
        solver_options=jqt.SolverOptions(progress_meter=None),
    ).to_dense().data

    assert result.shape == (len(times), 3, 2, 2)
    integral = (
        times[:, None] * base[None, :]
        + tau * amplitude[None, :]
        * (1 - jnp.exp(-times[:, None] / tau))
    )
    np.testing.assert_allclose(
        jnp.real(result[..., 1, 1]),
        jnp.exp(-integral),
        rtol=2e-5,
        atol=2e-5,
    )


def test_callable_sparse_operator_matches_dense():
    """A fixed BCOO sparsity pattern supports a time-dependent rate."""
    times = jnp.linspace(0.0, 2.0, 21)
    dense_lowering = jqt.destroy(2)
    sparse_lowering = dense_lowering.to_sparse_bcoo()
    # Pre-stack the BCOO operator before JAX traces the time-dependent callback.
    # Qarray.from_list() densifies and re-sparsifies BCOO inputs, which needs
    # a concrete nse (number of stored elements) and is not trace-safe.
    sparse_stack = jqt.Qarray.from_list([sparse_lowering])
    options = jqt.SolverOptions(progress_meter=None)

    def dense_c_ops(t):
        rate = 0.15 + 0.22 * jnp.exp(-t / 1.1)
        return jqt.Qarray.from_list([jnp.sqrt(rate) * dense_lowering])

    def sparse_c_ops(t):
        rate = 0.15 + 0.22 * jnp.exp(-t / 1.1)
        # Scalar multiplication updates stored values without changing indices.
        return jnp.sqrt(rate) * sparse_stack

    dense = jqt.mesolve(
        0.0, _initial_density(), times,
        c_ops=dense_c_ops, solver_options=options,
    ).to_dense().data
    sparse = jqt.mesolve(
        0.0, _initial_density(), times,
        c_ops=sparse_c_ops, solver_options=options,
    ).to_dense().data
    np.testing.assert_allclose(sparse, dense, atol=2e-5, rtol=2e-5)


def test_callable_rejects_non_qarray_result():
    """Invalid dynamic return types should fail with an actionable error."""
    with pytest.raises(TypeError, match=r"c_ops\(t\).*Qarray"):
        jqt.mesolve(
            0.0, _initial_density(), jnp.array([0.0, 1.0]),
            c_ops=lambda t: jnp.ones((1, 2, 2)),
        )


def test_callable_rejects_incompatible_hilbert_space():
    """The collapse operator and density matrix must have compatible sizes."""
    with pytest.raises(ValueError, match="Hilbert-space dimensions"):
        jqt.mesolve(
            0.0, _initial_density(), jnp.array([0.0, 1.0]),
            c_ops=lambda t: jqt.Qarray.from_list([jqt.destroy(3)]),
        )

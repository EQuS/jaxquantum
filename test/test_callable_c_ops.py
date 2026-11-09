"""Tests for callable time-dependent collapse operators.

These exercise the JAXQuantum + Diffrax solver, not an independent RK4 driver.
"""
import jax
import jax.numpy as jnp
import numpy as np
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

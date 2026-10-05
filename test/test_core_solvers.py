import os
import sys

import diffrax
import jax
import jax.numpy as jnp
import pytest
from jax import jit

# Add the jaxquantum directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jaxquantum as jqt
from jaxquantum.core import solvers


# sesolve ====
def test_sesolve():
    omega_q = 5.0 #GHzz
    Omega = .1
    g_state = jqt.basis(2,0) ^ jqt.basis(2,0)

    ts = jnp.linspace(0,5*jnp.pi/Omega,101)

    sz0 = jqt.sigmaz() ^ jqt.identity(N=2)

    @jit
    def Ht(t):
        H0 = omega_q/2.0*((jqt.sigmaz()^jqt.identity(N=2)) + (jqt.identity(N=2)^jqt.sigmaz()))
        H1 = Omega*jnp.cos((omega_q)*t)*((jqt.sigmax()^jqt.identity(N=2)) + (jqt.identity(N=2)^jqt.sigmax()))
        return H0 + H1

    states = jqt.sesolve(Ht, g_state, ts) 
    szt = jnp.real(jqt.overlap(states, sz0))

    test_time = ts[34]
    sim_szt =  szt[34]
    cal_szt = jnp.cos(Omega*test_time)
    assert jnp.isclose(sim_szt, cal_szt, atol=1e-4), f"Expected {cal_szt}, got {sim_szt}"

def test_sesolve_batch():
    omega_q = 5.0 #GHzz
    Omega = jnp.array([.1,.2])
    g_state = jqt.basis(2,0) ^ jqt.basis(2,0)

    ts = jnp.linspace(0,5*jnp.pi/Omega[0],101)

    sz0 = jqt.sigmaz() ^ jqt.identity(N=2)

    @jit
    def Ht(t):
        H0 = omega_q/2.0*((jqt.sigmaz()^jqt.identity(N=2)) + (jqt.identity(N=2)^jqt.sigmaz()))
        H1 = Omega*jnp.cos((omega_q)*t)*((jqt.sigmax()^jqt.identity(N=2)) + (jqt.identity(N=2)^jqt.sigmax()))
        return H0 + H1

    states = jqt.sesolve(Ht, g_state, ts) 
    szt = jnp.real(jqt.overlap(states, sz0))

    for j in range(2):
        test_time = ts[34]
        sim_szt =  szt[34,j]
        cal_szt = jnp.cos(Omega[j]*test_time)
        assert jnp.isclose(sim_szt, cal_szt, atol=1e-3), f"Expected {cal_szt}, got {sim_szt}"


def test_sesolve_edge_cases():
    # constant H0

    Omega = .1
    g_state = jqt.basis(2,0) ^ jqt.basis(2,0)
    ts = jnp.linspace(0,5*jnp.pi/Omega,101)
    sz0 = jqt.sigmaz() ^ jqt.identity(N=2)
    H0 = Omega/2*jqt.sigmax() ^ jqt.identity(N=2)
    states = jqt.sesolve(H0, g_state, ts)

    szt = jnp.real(jqt.overlap(states, sz0))

    test_time = ts[50]
    test_szt = szt[50]
    cal_szt = jnp.cos(Omega*test_time)
    assert jnp.isclose(test_szt, cal_szt, atol=1e-5)

    # valueerror if initial state is dm
    with pytest.raises(ValueError):
        jqt.sesolve(H0, g_state.to_dm(), ts)


def test_sesolve_save_final_only():
    """An empty saveat_tlist saves only the final state, and is jit-safe."""
    N = 4
    omega = 1.0
    psi0 = jqt.basis(N, 1)
    ts = jnp.linspace(0, 1.0, 50)
    opts = jqt.SolverOptions(progress_meter=None)
    H = omega * jqt.identity(N)

    full = jqt.sesolve(H, psi0, ts, solver_options=opts)
    final_only = jqt.sesolve(
        H, psi0, ts, saveat_tlist=jnp.array([]), solver_options=opts
    )

    # only a single (final) time slice is saved, and it matches the full solve
    assert full.data.shape[0] == len(ts)
    assert final_only.data.shape[0] == 1
    assert jnp.allclose(final_only.data[-1], full.data[-1], atol=1e-6)

    # jit-safe: keying on the (static) length of a traced saveat_tlist does not
    # raise a TracerBoolConversionError.
    @jit
    def run(saveat):
        return jqt.sesolve(H, psi0, ts, saveat_tlist=saveat, solver_options=opts).data

    jitted_final = run(jnp.array([]))
    assert jitted_final.shape[0] == 1
    assert jnp.allclose(jitted_final[-1], full.data[-1], atol=1e-6)


def test_native_diffrax_solver_options():
    """Native Diffrax solver, controller, and step size pass through."""
    N = 4
    omega = 1.0
    psi0 = jqt.basis(N, 1)
    ts = jnp.linspace(0, 1.0, 50)
    H = omega * jqt.identity(N)

    ref = jqt.sesolve(
        H, psi0, ts,
        solver_options=jqt.SolverOptions(progress_meter=None),
    )

    opts = jqt.SolverOptions(
        solver=diffrax.Dopri5(),
        stepsize_controller=diffrax.PIDController(rtol=1e-9, atol=1e-11),
        dt0=1e-3,
    )
    states = jqt.sesolve(H, psi0, ts, solver_options=opts)
    assert jnp.allclose(states.data, ref.data, atol=1e-6)

def test_solver_option_defaults():
    options = jqt.SolverOptions()
    assert isinstance(options.solver, diffrax.Tsit5)
    assert isinstance(options.stepsize_controller, diffrax.PIDController)
    assert options.max_steps == 100_000
    assert options.dt0 == "tlist"
    assert options.adjoint is None
    assert options.progress_meter == "default"


def test_progress_meter_options():
    assert solvers._resolve_progress_meter(None) is None
    assert solvers._resolve_progress_meter(False) is None
    assert isinstance(
        solvers._resolve_progress_meter("default"), solvers.CustomProgressMeter
    )
    assert isinstance(
        solvers._resolve_progress_meter(True), solvers.CustomProgressMeter
    )

    meter = diffrax.TextProgressMeter()
    assert solvers._resolve_progress_meter(meter) is meter

    with pytest.raises(ValueError, match="progress_meter"):
        solvers._resolve_progress_meter("text")
    with pytest.raises(TypeError, match="AbstractProgressMeter"):
        solvers._resolve_progress_meter(object())


def test_optional_diffrax_options_are_forwarded(monkeypatch):
    """None preserves Diffrax defaults; explicit native options pass through."""
    calls = []

    def fake_diffeqsolve(*args, **kwargs):
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(diffrax, "diffeqsolve", fake_diffeqsolve)
    times = jnp.array([0.0, 1.0])
    rhs = lambda t, y, args: y

    solvers.solve(
        rhs,
        jnp.ones(1),
        times,
        solver_options=jqt.SolverOptions(progress_meter=None),
    )
    assert calls[-1]["dt0"] == 1.0
    for name in (
        "adjoint",
        "event",
        "progress_meter",
        "solver_state",
        "controller_state",
        "made_jump",
    ):
        assert name not in calls[-1]

    solvers.solve(rhs, jnp.ones(1), times)
    assert isinstance(calls[-1]["progress_meter"], solvers.CustomProgressMeter)

    solvers.solve(
        rhs,
        jnp.ones(1),
        times,
        solver_options=jqt.SolverOptions(dt0=None),
    )
    assert calls[-1]["dt0"] is None

    with pytest.raises(ValueError, match="dt0"):
        solvers.solve(
            rhs,
            jnp.ones(1),
            times,
            solver_options=jqt.SolverOptions(dt0="invalid"),
        )

    adjoint = diffrax.ForwardMode()
    event = diffrax.Event(lambda t, y, args, **kwargs: False)
    meter = diffrax.TextProgressMeter()
    solver_state = {"state": jnp.array(1.0)}
    controller_state = {"state": jnp.array(2.0)}
    options = jqt.SolverOptions(
        adjoint=adjoint,
        event=event,
        progress_meter=meter,
        solver_state=solver_state,
        controller_state=controller_state,
        made_jump=False,
        max_steps=17,
        throw=False,
    )
    solvers.solve(rhs, jnp.ones(1), times, solver_options=options)
    call = calls[-1]
    assert call["adjoint"] is adjoint
    assert call["event"] is event
    assert call["progress_meter"] is meter
    assert call["solver_state"] is solver_state
    assert call["controller_state"] is controller_state
    assert call["made_jump"] is False
    assert call["max_steps"] == 17
    assert call["throw"] is False


def test_legacy_solver_options_warn_and_match_native_options():
    times = jnp.linspace(0.0, 1.0, 11)
    psi0 = jqt.basis(2, 0)
    hamiltonian = 0.2 * jqt.sigmax()
    native = jqt.SolverOptions(
        solver=diffrax.Dopri5(),
        stepsize_controller=diffrax.PIDController(rtol=1e-9, atol=1e-11),
    )

    with pytest.warns(FutureWarning, match=r"SolverOptions\.create"):
        created = jqt.SolverOptions.create(
            progress_meter=False,
            solver="Dopri5",
            stepsize_controller_kwargs={"rtol": 1e-9, "atol": 1e-11},
        )
    assert jnp.allclose(
        jqt.sesolve(hamiltonian, psi0, times, solver_options=created).data,
        jqt.sesolve(hamiltonian, psi0, times, solver_options=native).data,
    )

    legacy = jqt.SolverOptions(
        progress_meter=False,
        solver="Dopri5",
        stepsize_controller="PIDController",
        stepsize_controller_kwargs={"rtol": 1e-9, "atol": 1e-11},
    )
    with pytest.warns(FutureWarning, match="native objects"):
        result = jqt.sesolve(hamiltonian, psi0, times, solver_options=legacy)
    expected = jqt.sesolve(
        hamiltonian, psi0, times, solver_options=native
    ).data
    assert jnp.allclose(result.data, expected)


def test_sesolve_result_saveat_and_gradients():
    times = jnp.linspace(0.0, 1.0, 21)
    psi0 = jqt.basis(2, 0)

    result = jqt.sesolve_result(0.2 * jqt.sigmax(), psi0, times)
    states = jqt.sesolve(0.2 * jqt.sigmax(), psi0, times)
    assert isinstance(result, diffrax.Solution)
    assert jnp.allclose(result.ys, states.data)
    assert "num_steps" in result.stats

    final = jqt.sesolve_result(
        0.2 * jqt.sigmax(),
        psi0,
        times,
        solver_options=jqt.SolverOptions(saveat=diffrax.SaveAt(t1=True)),
    )
    assert final.ys.shape[0] == 1

    with pytest.raises(ValueError, match="not both"):
        jqt.sesolve_result(
            0.2 * jqt.sigmax(),
            psi0,
            times,
            saveat_tlist=times[-2:],
            solver_options=jqt.SolverOptions(saveat=diffrax.SaveAt(t1=True)),
        )

    def population(scale, adjoint=None):
        options = jqt.SolverOptions(
            adjoint=adjoint,
            saveat=diffrax.SaveAt(t1=True),
        )
        solution = jqt.sesolve_result(
            scale * jqt.sigmax(), psi0, times, solver_options=options
        )
        return jnp.abs(solution.ys[-1, 1]) ** 2

    assert jnp.isfinite(jax.grad(population)(0.2))
    assert jnp.isfinite(jax.jacfwd(population)(0.2, diffrax.ForwardMode()))


def test_mesolve_result_matches_states():
    times = jnp.linspace(0.0, 1.0, 11)
    rho0 = jqt.basis(3, 1).to_dm()
    collapse = jqt.Qarray.from_list([0.2 * jqt.destroy(3)])
    result = jqt.mesolve_result(jqt.num(3), rho0, times, c_ops=collapse)
    states = jqt.mesolve(jqt.num(3), rho0, times, c_ops=collapse)
    assert isinstance(result, diffrax.Solution)
    assert jnp.allclose(result.ys, states.data)


def test_propagator_forwards_solver_options():
    times = jnp.linspace(0.0, 1.0, 11)
    result = jqt.propagator(
        lambda t: 0.2 * jqt.sigmax(),
        times,
        solver_options=jqt.SolverOptions(saveat=diffrax.SaveAt(t1=True)),
    )
    assert result.data.shape == (1, 2, 2)

# ====

# mesolve ====

def test_mesolve_batch():
    N = 100
    a = jqt.destroy(N)
    n = a.dag() @ a

    omega_a = 2.0*jnp.pi*5.0
    H0 = omega_a*n

    kappa = 2*jnp.pi*jnp.array([1,2])
    batched_loss_op = jnp.sqrt(kappa)*a
    c_ops = jqt.Qarray.from_list([batched_loss_op])

    initial_state = (jqt.displace(N, 0.1) @ jqt.basis(N,0)).to_dm() # initial state

    ts = jnp.linspace(0, 4*2*jnp.pi/omega_a, 101) # Time points

    solver_options = jqt.SolverOptions()
    states = jqt.mesolve(
        H0, initial_state, ts, c_ops=c_ops, solver_options=solver_options) # solve
        
    n_exp = jnp.real(jqt.overlap(n, states))


    for j in range(2):
        test_time = ts[50]
        test_nt = n_exp[50,j]
        expected_nt = jnp.exp(-kappa[j]*test_time) * jnp.abs(jqt.overlap(n, initial_state))  # Expectation value of n at time t
        assert jnp.isclose(test_nt, expected_nt, atol=1e-8), f"Expected {expected_nt}, got {test_nt}"

def test_mesolve():
    N = 100

    omega_a = 2.0*jnp.pi*5.0
    kappa = 2*jnp.pi*1.0 # Batching to explore two different kappa values!
    initial_state = jqt.displace(N, 0.1) @ jqt.basis(N,0)
    initial_state_dm = initial_state.to_dm()
    ts = jnp.linspace(0, 4*2*jnp.pi/omega_a, 101)

    a = jqt.destroy(N)
    n = a.dag() @ a

    c_ops = jqt.Qarray.from_list([jnp.sqrt(kappa)*a])

    @jit
    def Ht(t):
        H0 = omega_a*n
        return H0

    solver_options = jqt.SolverOptions()
    states = jqt.mesolve(Ht, initial_state_dm, ts, c_ops=c_ops, solver_options=solver_options) 
    nt = jnp.real(jqt.overlap(n, states))

    test_time = ts[50]
    test_nt = nt[50]
    expected_nt = jnp.exp(-kappa*test_time) * jnp.abs(jqt.overlap(n, initial_state_dm))  # Expectation value of n at time t
    assert jnp.isclose(test_nt, expected_nt, atol=1e-8), f"Expected {expected_nt}, got {test_nt}"


def test_mesolve_edge_cases():
    # constant H0

    Omega = .1
    g_state = jqt.basis(2,0) ^ jqt.basis(2,0)
    ts = jnp.linspace(0,5*jnp.pi/Omega,101)
    sz0 = jqt.sigmaz() ^ jqt.identity(N=2)
    H0 = Omega/2*jqt.sigmax() ^ jqt.identity(N=2)
    c_ops = jqt.Qarray.from_list([])
    states = jqt.mesolve(H0, g_state, ts, c_ops=c_ops)

    szt = jnp.real(jqt.overlap(states, sz0))

    test_time = ts[50]
    test_szt = szt[50]
    cal_szt = jnp.cos(Omega*test_time)
    assert jnp.isclose(test_szt, cal_szt, atol=1e-5)


def test_sesolve_scalar_H():
    """sesolve with a scalar H = omega should behave like H = omega * I."""
    N = 4
    omega = 2.0
    psi0 = jqt.basis(N, 1)
    ts = jnp.linspace(0, 1.0, 50)

    states_scalar = jqt.sesolve(omega, psi0, ts)

    H_ref = omega * jqt.identity(N)
    states_ref = jqt.sesolve(H_ref, psi0, ts)

    assert jnp.allclose(
        jnp.abs(states_scalar.data) ** 2,
        jnp.abs(states_ref.data) ** 2,
        atol=1e-6,
    ), "sesolve scalar H populations differ from omega*I reference"


def test_mesolve_scalar_H():
    """mesolve with a scalar H should match H = omega * I."""
    N = 4
    omega = 1.0
    kappa = 0.2
    rho0 = jqt.basis(N, 2).to_dm()
    ts = jnp.linspace(0, 2.0, 40)
    c_ops = jqt.Qarray.from_list([jqt.destroy(N) * jnp.sqrt(kappa)])
    opts = jqt.SolverOptions(progress_meter=None)

    result_scalar = jqt.mesolve(omega, rho0, ts, c_ops=c_ops, solver_options=opts)

    H_ref = omega * jqt.identity(N)
    result_ref = jqt.mesolve(H_ref, rho0, ts, c_ops=c_ops, solver_options=opts)

    assert jnp.allclose(
        result_scalar.data, result_ref.data, atol=1e-6
    ), "mesolve scalar H differs from omega*I reference"


# ====


# sparse / sparse_dia initial states ====

def test_sesolve_sparse_ket():
    """sesolve with a BCOO sparse ket should match the dense result."""
    N = 4
    omega = 1.0
    psi0 = jqt.basis(N, 1)
    psi0_sparse = psi0.to_sparse_bcoo()
    ts = jnp.linspace(0, 1.0, 20)
    opts = jqt.SolverOptions(progress_meter=None)

    ref = jqt.sesolve(omega * jqt.identity(N), psi0, ts, solver_options=opts)
    result = jqt.sesolve(omega * jqt.identity(N), psi0_sparse, ts, solver_options=opts)

    assert jnp.allclose(result.data, ref.data, atol=1e-6), \
        "sesolve with sparse ket differs from dense reference"


def test_mesolve_sparse_dm():
    """mesolve with a BCOO sparse density matrix should match the dense result."""
    N = 4
    kappa = 0.2
    rho0 = jqt.basis(N, 1).to_dm()
    rho0_sparse = rho0.to_sparse_bcoo()
    ts = jnp.linspace(0, 1.0, 20)
    H = jqt.num(N)
    c_ops = jqt.Qarray.from_list([jqt.destroy(N) * jnp.sqrt(kappa)])
    opts = jqt.SolverOptions(progress_meter=None)

    ref = jqt.mesolve(H, rho0, ts, c_ops=c_ops, solver_options=opts)
    result = jqt.mesolve(H, rho0_sparse, ts, c_ops=c_ops, solver_options=opts)

    assert jnp.allclose(result.data, ref.data, atol=1e-6), \
        "mesolve with sparse dm differs from dense reference"


def test_mesolve_sparse_dia_dm():
    """mesolve with a SparseDIA density matrix should match the dense result."""
    N = 4
    kappa = 0.2
    rho0 = jqt.basis(N, 1).to_dm()
    rho0_dia = rho0.to_sparse_dia()
    ts = jnp.linspace(0, 1.0, 20)
    H = jqt.num(N)
    c_ops = jqt.Qarray.from_list([jqt.destroy(N) * jnp.sqrt(kappa)])
    opts = jqt.SolverOptions(progress_meter=None)

    ref = jqt.mesolve(H, rho0, ts, c_ops=c_ops, solver_options=opts)
    result = jqt.mesolve(H, rho0_dia, ts, c_ops=c_ops, solver_options=opts)

    assert jnp.allclose(result.data, ref.data, atol=1e-6), \
        "mesolve with SparseDIA dm differs from dense reference"

# ====


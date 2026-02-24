import pytest

from jax import jit
import sys
import os

# Add the jaxquantum directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jaxquantum as jqt
import jax.numpy as jnp


# sesolve ====
def test_sesolve():
    omega_q = 5.0 #GHzz
    Omega = .1
    g_state = jqt.basis(2,0) ^ jqt.basis(2,0)
    g_state_dm = g_state.to_dm()

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
    g_state_dm = g_state.to_dm()

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

    omega_q = 5.0 #GHzz
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

# ====

# mesolve ====

def test_mesolve_batch():
    N = 100
    a = jqt.destroy(N); n = a.dag() @ a

    omega_a = 2.0*jnp.pi*5.0; H0 = omega_a*n # Hamiltonian

    kappa = 2*jnp.pi*jnp.array([1,2]); batched_loss_op = jnp.sqrt(kappa)*a; 
    c_ops = jqt.Qarray.from_list([batched_loss_op]) # collapse operators

    initial_state = (jqt.displace(N, 0.1) @ jqt.basis(N,0)).to_dm() # initial state

    ts = jnp.linspace(0, 4*2*jnp.pi/omega_a, 101) # Time points

    solver_options = jqt.SolverOptions.create(progress_meter=True) 
    states = jqt.mesolve(
        H0, initial_state, ts, c_ops=c_ops, solver_options=solver_options) # solve
        
    n_exp = jnp.real(jqt.overlap(n, states)); a_exp = jqt.overlap(a, states) # expectation values


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

    solver_options = jqt.SolverOptions.create(progress_meter=True)
    states = jqt.mesolve(Ht, initial_state_dm, ts, c_ops=c_ops, solver_options=solver_options) 
    nt = jnp.real(jqt.overlap(n, states))
    a_real = jnp.real(jqt.overlap(a, states))
    a_imag = jnp.imag(jqt.overlap(a, states))

    test_time = ts[50]
    test_nt = nt[50]
    expected_nt = jnp.exp(-kappa*test_time) * jnp.abs(jqt.overlap(n, initial_state_dm))  # Expectation value of n at time t
    assert jnp.isclose(test_nt, expected_nt, atol=1e-8), f"Expected {expected_nt}, got {test_nt}"


def test_mesolve_edge_cases():
    # constant H0

    omega_q = 5.0 #GHzz
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


# ====


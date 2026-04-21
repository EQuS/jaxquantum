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
    opts = jqt.SolverOptions.create(progress_meter=False)

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
    psi0_sparse = psi0.to_sparse()
    ts = jnp.linspace(0, 1.0, 20)
    opts = jqt.SolverOptions.create(progress_meter=False)

    ref = jqt.sesolve(omega * jqt.identity(N), psi0, ts, solver_options=opts)
    result = jqt.sesolve(omega * jqt.identity(N), psi0_sparse, ts, solver_options=opts)

    assert jnp.allclose(result.data, ref.data, atol=1e-6), \
        "sesolve with sparse ket differs from dense reference"


def test_mesolve_sparse_dm():
    """mesolve with a BCOO sparse density matrix should match the dense result."""
    N = 4
    kappa = 0.2
    rho0 = jqt.basis(N, 1).to_dm()
    rho0_sparse = rho0.to_sparse()
    ts = jnp.linspace(0, 1.0, 20)
    H = jqt.num(N)
    c_ops = jqt.Qarray.from_list([jqt.destroy(N) * jnp.sqrt(kappa)])
    opts = jqt.SolverOptions.create(progress_meter=False)

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
    opts = jqt.SolverOptions.create(progress_meter=False)

    ref = jqt.mesolve(H, rho0, ts, c_ops=c_ops, solver_options=opts)
    result = jqt.mesolve(H, rho0_dia, ts, c_ops=c_ops, solver_options=opts)

    assert jnp.allclose(result.data, ref.data, atol=1e-6), \
        "mesolve with SparseDIA dm differs from dense reference"

# ====


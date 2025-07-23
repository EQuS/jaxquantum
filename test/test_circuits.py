import pytest

import sys
import os

# Add the jaxquantum directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jaxquantum as jqt
import jaxquantum.circuits as jqtc
import jax.numpy as jnp



def test_unitary_simulation():
    N = 10
    beta = 2
    reg = jqtc.Register([2,N])
    cirq = jqtc.Circuit.create(reg, layers=[])
    cirq.append(jqtc.X(),0)
    cirq.append(jqtc.CD(N, beta),[0,1])
    initial_state = jqt.basis(2,0) ^ jqt.basis(N,0)
    
    
    # Unitary simulation
    res = jqtc.simulate(cirq, initial_state)

    a = jqt.identity(2) ^ jqt.destroy(N)
    q = (a + a.dag())/2

    assert jnp.abs(beta/2 + jqt.overlap(res[-1][-1], q)) < 1e-6, "Overlap with q should be close to beta/2"


    # Kraus map simulation
    res = jqtc.simulate(cirq, initial_state, mode="kraus")

    a = jqt.identity(2) ^ jqt.destroy(N)
    q = (a + a.dag())/2

    assert jnp.abs(beta/2 + jqt.overlap(res[-1][-1], q)) < 1e-6, "Overlap with q should be close to beta/2"
    

def test_hamiltonian_simulation():
    N = 10
    alpha = 1
    reg = jqtc.Register([2,N])
    cirq = jqtc.Circuit.create(reg, layers=[])
    cirq.append(jqtc.X(),0)
    cirq.append(jqtc.D(N, alpha, ts=jnp.linspace(0,100,101)),1, default_simulate_mode="hamiltonian")
    initial_state = jqt.basis(2,0) ^ jqt.basis(N,0)
    res = jqtc.simulate(cirq, initial_state)

    a = jqt.identity(2) ^ jqt.destroy(N)
    q = (a + a.dag())/2

    assert jnp.abs(alpha - jqt.overlap(res[-1][-1], q)) < 1e-7
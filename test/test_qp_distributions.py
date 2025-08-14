#%% md
# Unit Tests for qp_distributions

#%%
import pytest
import jax.numpy as jnp
from jaxquantum import basis
from jaxquantum.core.qarray import Qarray
from jaxquantum.core.qp_distributions import (wigner, qfunc,
                                              _wigner_clenshaw,
                                              _wig_laguerre_val)
from jax.scipy.special import factorial


#%%
def test_wigner_coherent_state():
    """Test wigner function with a coherent state"""
    # Create a simple coherent state (|α⟩)
    N = 20  # Truncation dimension
    alpha = 1.0  # Coherent state parameter
    
    # Create coherent state vector
    n = jnp.arange(N)
    state_vec = jnp.exp(-0.5 * abs(alpha)**2) * alpha**n / jnp.sqrt(factorial(n))
    state = Qarray.create(state_vec)
    
    xvec = jnp.linspace(-3, 3, 50)
    yvec = jnp.linspace(-3, 3, 50)
    
    W = wigner(state, xvec, yvec)
    
    # Test properties of the Wigner function
    assert W.shape == (50, 50)  # Check output shape
    assert jnp.all(jnp.isreal(W))  # Wigner function should be real
    assert (jnp.abs(jnp.sum(W) * (xvec[1]-xvec[0]) * (yvec[1]-yvec[0]) - 1.0)
            < 1e-4)  # Normalization

#%%
def test_husimi_coherent_state():
    """Test husimi Q-function with a coherent state"""
    N = 20  # Truncation dimension
    alpha = 1.0  # Coherent state parameter
    
    # Create coherent state vector
    n = jnp.arange(N)
    state_vec = jnp.exp(-0.5 * abs(alpha)**2) * alpha**n / jnp.sqrt(factorial(n))
    state = Qarray.create(state_vec)
    
    xvec = jnp.linspace(-3, 3, 50)
    yvec = jnp.linspace(-3, 3, 50)
    
    Q = qfunc(state, xvec, yvec)
    
    # Test properties of the Husimi Q-function
    assert Q.shape == (50, 50)  # Check output shape
    assert jnp.all(jnp.isreal(Q))  # Q-function should be real
    assert jnp.all(Q >= 0)  # Q-function should be non-negative
    assert (jnp.abs(jnp.sum(Q) * (xvec[1]-xvec[0]) * (yvec[1]-yvec[0]) - 1.0)
            < 1e-2)  # Normalization

#%%
def test_wigner_density_matrix():
    """Test wigner function with a density matrix input"""
    # Create a simple pure state density matrix
    state_vec = jnp.array([1.0, 0.0])
    state = Qarray.create(state_vec)
    rho = state @ state.dag()  # Create density matrix
    
    xvec = jnp.linspace(-3, 3, 50)
    yvec = jnp.linspace(-3, 3, 50)
    
    W = wigner(rho, xvec, yvec)
    
    assert W.shape == (50, 50)
    assert jnp.all(jnp.isreal(W))
    assert jnp.abs(jnp.sum(W) * (xvec[1]-xvec[0]) * (yvec[1]-yvec[0]) - 1.0) < 1e-5

#%%
def test_wigner_mixed_state():
    """Test wigner function with a mixed state"""
    # Create a mixed state (ρ = 0.5|0⟩⟨0| + 0.5|1⟩⟨1|)
    basis_0 = Qarray.create(jnp.array([1.0, 0.0]))
    basis_1 = Qarray.create(jnp.array([0.0, 1.0]))
    
    rho = 0.5 * (basis_0 @ basis_0.dag() + basis_1 @ basis_1.dag())
    
    xvec = jnp.linspace(-3, 3, 50)
    yvec = jnp.linspace(-3, 3, 50)
    
    W = wigner(rho, xvec, yvec)
    
    assert W.shape == (50, 50)
    assert jnp.all(jnp.isreal(W))
    assert jnp.abs(jnp.sum(W) * (xvec[1]-xvec[0]) * (yvec[1]-yvec[0]) - 1.0) < 1e-5


#%% md
# Tests with Known Analytical Solutions

#%%
def test_wigner_fock_state_zero():
    """Test Wigner function for Fock state |0⟩ (vacuum state).
    The analytical solution for the Wigner function of |0⟩ is:
    W(x,y) = (1/π) * exp(-(x² + y²))
    """
    N = 20
    state = basis(N, 0)  # |0⟩ state

    # Test at specific points
    xvec = jnp.array([0.0, 1.0, -1.0])
    yvec = jnp.array([0.0, 1.0, -1.0])

    W = wigner(state, xvec, yvec, g=jnp.sqrt(2))

    # Calculate analytical values
    X, Y = jnp.meshgrid(xvec, yvec)
    W_analytical = (1/jnp.pi) * jnp.exp(-(X**2 + Y**2))

    assert jnp.allclose(W, W_analytical, rtol=1e-5)

#%%
def test_husimi_fock_state_zero():
    """Test Husimi Q-function for Fock state |0⟩ (vacuum state).
    The analytical solution for the Q-function of |0⟩ is:
    Q(x,y) = (1/π) * exp(-|α|²), where α = (x + iy)/√2
    """
    N = 20
    state = Qarray.create(jnp.array([1.0] + [0.0] * (N-1)))  # |0⟩ state

    # Test at specific points
    xvec = jnp.array([0.0, 1.0, -1.0])
    yvec = jnp.array([0.0, 1.0, -1.0])

    Q = qfunc(state, xvec, yvec, g=jnp.sqrt(2))

    # Calculate analytical values
    X, Y = jnp.meshgrid(xvec, yvec)
    alpha = (X + 1j*Y)/jnp.sqrt(2)
    Q_analytical = (1/jnp.pi) * jnp.exp(-jnp.abs(alpha)**2)

    assert jnp.allclose(Q, Q_analytical, rtol=1e-5)

#%%
def test_wigner_coherent_state_value():
    """Test Wigner function for a coherent state |α⟩.
    The analytical solution for the Wigner function of |α⟩ is:
    W(x,y) = (1/π) * exp(-(x-x₀)² - (y-y₀)²)
    where x₀ = √2 Re(α) and y₀ = √2 Im(α)
    """
    N = 30  # Need larger dimension for good accuracy with coherent states
    alpha = 1.0 + 0.5j  # Coherent state parameter

    # Create coherent state
    n = jnp.arange(N)
    state_vec = jnp.exp(-0.5 * abs(alpha)**2) * alpha**n / jnp.sqrt(factorial(n))
    state = Qarray.create(state_vec)

    # Test points
    xvec = jnp.array([-1.0, 0.0, 1.0])
    yvec = jnp.array([-1.0, 0.0, 1.0])

    W = wigner(state, xvec, yvec, g=jnp.sqrt(2))

    # Calculate analytical values
    X, Y = jnp.meshgrid(xvec, yvec)
    x0 = jnp.sqrt(2) * alpha.real
    y0 = jnp.sqrt(2) * alpha.imag
    W_analytical = (1/jnp.pi) * jnp.exp(-((X-x0)**2 + (Y-y0)**2))

    assert jnp.allclose(W, W_analytical, rtol=1e-4)

#%%
def test_husimi_coherent_state_value():
    """Test Husimi Q-function for a coherent state |α⟩.
    The analytical solution for the Q-function of |α⟩ is:
    Q(β) = (1/π) * exp(-|β-α|²)
    where β = (x + iy)/√2 is the phase-space point
    """
    N = 30
    alpha = 1.0 + 0.5j

    # Create coherent state
    n = jnp.arange(N)
    state_vec = jnp.exp(-0.5 * abs(alpha)**2) * alpha**n / jnp.sqrt(factorial(n))
    state = Qarray.create(state_vec)

    # Test points
    xvec = jnp.array([-1.0, 0.0, 1.0])
    yvec = jnp.array([-1.0, 0.0, 1.0])

    Q = qfunc(state, xvec, yvec, g=jnp.sqrt(2))

    # Calculate analytical values
    X, Y = jnp.meshgrid(xvec, yvec)
    beta = (X + 1j*Y)/jnp.sqrt(2)
    Q_analytical = (1/jnp.pi) * jnp.exp(-jnp.abs(beta - alpha)**2)

    assert jnp.allclose(Q, Q_analytical, rtol=1e-4)

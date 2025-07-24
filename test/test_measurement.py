#%% md
### Testing measurements.py

#%%
import pytest
import jax.numpy as jnp
from jaxquantum.core.qarray import Qarray
from jaxquantum.core.operators import identity, sigmax, sigmay, sigmaz
from jaxquantum.core.measurements import (
    overlap, 
    fidelity,
    QuantumStateTomography,
    tensor_basis,
    _reconstruct_density_matrix,
    _parametrize_density_matrix,
    _L1_reg,
    _likelihood
)

#%% md
### Test helper functions

#%%
def test_reconstruct_density_matrix():
    # Test with a 2x2 matrix
    dim = 2
    params = jnp.array([1.0, 0.0, 0.5, 2.0])  # Real diagonal and
    # lower triangle
    rho = _reconstruct_density_matrix(params, dim)
    
    # Check properties of density matrix
    assert rho.shape == (2, 2)
    assert jnp.allclose(rho, rho.conj().T)  # Hermitian
    assert jnp.allclose(jnp.trace(rho), 1.0)  # Trace 1
    assert jnp.all(jnp.linalg.eigvals(rho).real >= -1e-10)  # Positive semidefinite

#%%
def test_parametrize_density_matrix():
    # Test with a known density matrix
    dim = 2
    rho = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex128)
    params = _parametrize_density_matrix(rho, dim)
    
    # Reconstruct and verify
    reconstructed = _reconstruct_density_matrix(params, dim)
    assert jnp.allclose(rho, reconstructed, atol=1e-6)

#%%
def test_L1_reg():
    params = jnp.array([1.0, -2.0, 3.0])
    reg = _L1_reg(params)
    assert jnp.allclose(reg, 6.0)  # |1| + |-2| + |3| = 6

#%% md
### Test core measurement functions

#%%
def test_overlap():
    # Test overlap between pure states
    psi = Qarray.create(jnp.array([1.0, 0.0]))
    phi = Qarray.create(jnp.array([1.0/jnp.sqrt(2), 1.0/jnp.sqrt(2)]))
    
    overlap_val = overlap(psi, phi)
    expected = 0.5  # |<ψ|φ>|^2 = |1/√2|^2 = 1/2
    assert jnp.allclose(overlap_val, expected)

#%%
def test_fidelity():
    # Test fidelity between pure states
    psi = Qarray.create(jnp.array([1.0, 0.0]))
    phi = Qarray.create(jnp.array([1.0/jnp.sqrt(2), 1.0/jnp.sqrt(2)]))
    
    fid = fidelity(psi, phi)
    expected = 0.5  # For pure states, fidelity = |<ψ|φ>|^2
    assert jnp.allclose(fid, expected)
    
    # Test fidelity with force_positivity
    fid_pos = fidelity(psi.to_dm(), phi.to_dm(), force_positivity=True)
    assert jnp.allclose(fid_pos, expected)

#%% md
### Test Quantum State Tomography

#%%
def test_quantum_state_tomography_direct():
    # Create a simple test state
    dim = 2
    rho = Qarray.create(jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex128))
    
    # Create measurement basis (Pauli matrices)
    basis = Qarray.from_list([identity(2), sigmax(), sigmay(), sigmaz()])
    
    # Generate measurement results
    results = jnp.array([1.0, 0.0, 0.0, 1.0])
    
    # Perform tomography
    qst = QuantumStateTomography(
        rho_guess=identity(2)/2,
        measurement_basis=basis,
        measurement_results=results
    )
    
    reconstructed = qst.quantum_state_tomography_direct()
    assert jnp.allclose(reconstructed.data, rho.data, atol=1e-6)

#%%
def test_quantum_state_tomography_mle():
    # Create a simple test state
    dim = 2
    rho = Qarray.create(jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex128))
    
    # Create measurement basis
    basis = Qarray.from_list([identity(2), sigmax(), sigmay(), sigmaz()])
    
    # Generate measurement results
    results = jnp.array([1.0, 0.0, 0.0, 1.0])
    
    # Perform tomography
    qst = QuantumStateTomography(
        rho_guess=identity(2)/2,
        measurement_basis=basis,
        measurement_results=results,
        true_rho=rho
    )
    
    result = qst.quantum_state_tomography_mle(epochs=500)
    
    # Check result properties
    assert isinstance(result.rho, Qarray)
    assert len(result.loss_history) == 500
    assert len(result.grads_history) == 500
    assert result.infidelity_history is not None
    
    # Check fidelity between reconstructed and true state
    fid = fidelity(result.rho, rho)
    assert fid > 0.95  # Allow for some numerical imprecision

#%% md
### Test Tensor Basis Construction

#%%
def test_tensor_basis():
    # Create single-qubit Pauli basis
    single_basis = Qarray.from_list([identity(2), sigmax(), sigmay(),
                                     sigmaz()])
    
    # Create two-qubit basis
    two_qubit_basis = tensor_basis(single_basis, 2)
    
    # Check dimensions
    assert two_qubit_basis.bdims == (16,)  # 4^2 = 16 basis elements
    assert two_qubit_basis.data.shape == (16, 4, 4)  # 16 4x4 matrices

#%% md
### Run all tests

#%%

# /Users/phionx/Github/qc/EQuS/bosonic/jax/jaxquantum/test_example.py

import pytest

import sys
import os

# Add the jaxquantum directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jaxquantum as jqt
import jax.numpy as jnp
import qutip as qt

minimum_version_for_tests = "0.2.0"


# Version
# ========================================
def test_conversions():
    a = jqt.basis(4, 3)
    assert jqt.jqt2qt(a) == qt.basis(4, 3)
    assert jqt.jqt2qt(None) is None

    a = jqt.displace(4, jnp.array([[1.0,2,3],[4,5,6]]))
    assert jqt.jqt2qt(a)[1][2] == jqt.jqt2qt(jqt.displace(4,6))

    b = jqt.displace(4,1.2)
    assert jqt.jnp2jqt(b.data) == b
    assert jqt.jnp2jqt(b.data, dims=(4,)) == b

    c = jqt.basis(4,1)
    assert jqt.jnp2jqt(c.data, dims=(4,)) == c

    assert jqt.qt2jqt(c) == c
    assert jqt.qt2jqt(jqt.jqt2qt(c)) == c


# ========================================

# Helpers
# ========================================

def test_overlap():
    a = jqt.basis(3,0)
    b = jqt.basis(3,1)
    n = jqt.num(3)

    assert jqt.overlap(a,b) == 0
    assert jqt.overlap(a,n) == 0
    assert jqt.overlap(n,b) == 1
    assert jqt.overlap(n,a.to_dm()) == 0
    assert jqt.overlap(b.to_dm(), n) == 1


# ========================================

# Utils
# ========================================
def test_device():
    a = {"test1": 1.0, "test2": jnp.array([1,2]), "test3": "wow"}
    jqt.device_put_params(a, non_device_params=["test3"])

def test_comb():
    assert jnp.abs(jqt.comb(5,2) - 10) < 1e-7

def test_iso_transforms():
    a = jqt.displace(3,1)
    assert jnp.all(jqt.real_to_complex_iso_matrix(jqt.complex_to_real_iso_matrix(a.data)) == a.data)
    
    b = jqt.coherent(3,1)
    assert jnp.all(jqt.real_to_complex_iso_vector(jqt.complex_to_real_iso_vector(a.data)) == a.data)

    a_iso = jqt.complex_to_real_iso_matrix(a.data)
    assert jnp.all(jqt.imag_times_iso_matrix(a_iso) == jqt.complex_to_real_iso_matrix((1j*a).data))
    assert jnp.all(jqt.conj_transpose_iso_matrix(a_iso) == jqt.complex_to_real_iso_matrix((a.dag()).data))

    b_iso = jqt.complex_to_real_iso_vector(b.data)
    assert jnp.all(jqt.imag_times_iso_vector(b_iso) == jqt.complex_to_real_iso_vector((1j*b).data))

# ========================================

# Operators

# ========================================

def test_misc_operators():
    assert jnp.max(jnp.abs(jqt.sigmay().data - jnp.array([[0.0, -1.0j], [1.0j, 0.0]]))) < 1e-12

    assert jnp.max(jnp.abs(jqt.hadamard().data - jnp.array([[1, 1], [1, -1]]) / jnp.sqrt(2)))  < 1e-12

    assert jnp.max(jnp.abs(jqt.sigmam().data - jnp.array([[0.0, 0.0], [1.0, 0.0]]))) < 1e-12

    assert jnp.max(jnp.abs(jqt.sigmap().data - jnp.array([[0.0, 1.0], [0.0, 0.0]]))) < 1e-12

    assert jnp.max(jnp.abs((jqt.qubit_rotation(0,0,0,0).data - jqt.identity(2).data))) < 1e-12

    assert jnp.max(jnp.abs(jqt.identity_like(jqt.sigmax()).data - jqt.identity(2).data)) < 1e-12

    state_val = jqt.thermal(3,0.1).data[2,2] 
    calc_val = jnp.exp(-0.1*2)/(jnp.exp(-0.1*0)+jnp.exp(-0.1*1)+jnp.exp(-0.1*2))
    assert jnp.abs(state_val-calc_val) < 1e-7

    assert jqt.thermal(10,jnp.inf) == jqt.basis(10,0)

    assert jqt.basis_like((jqt.identity(2)^jqt.identity(3)), [1,0]) == (jqt.basis(2,1)^jqt.basis(3,0)) 
# ========================================
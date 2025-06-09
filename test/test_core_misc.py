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
def test_helpers_misc():
    a = jqt.basis(3,0)
    assert jqt.isvec(a)
    assert not jqt.isvec(a.to_dm())

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
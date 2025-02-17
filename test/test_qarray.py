# /Users/phionx/Github/qc/EQuS/bosonic/jax/jaxquantum/test_example.py

import pytest

import sys
import os

# Add the jaxquantum directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jaxquantum as jqt
import jax.numpy as jnp

minimum_version_for_tests = "0.2.0"

# Qarray
# ========================================
def test_version():
    high_enough = False 
    ver_tuple = tuple(map(int, jqt.__version__.split('.')))
    test_ver_tuple = tuple(map(int, minimum_version_for_tests.split('.')))
    high_enough = high_enough or (ver_tuple[0] > test_ver_tuple[0])
    high_enough = high_enough or (ver_tuple[0] == test_ver_tuple[0] and ver_tuple[1] > test_ver_tuple[1])
    high_enough = high_enough or (ver_tuple[0] == test_ver_tuple[0] and ver_tuple[1] == test_ver_tuple[1] and ver_tuple[2] >= test_ver_tuple[2])
    assert high_enough


def test_qarray_basic_math_add():
    a = jqt.displace(2,1.0)
    b = jqt.displace(2,1.25)
    c = jqt.displace(2,1.5)

    arr = jqt.QarrayArray.create([a,b])

    # Qarray + Qarray
    assert jnp.max(jnp.abs((a._data + b._data) - (a+b)._data)) < 1e-10
    
    # QarrayArray + Qarray
    assert jnp.max(jnp.abs((arr + c)._data - jqt.QarrayArray.create([a+c,b+c])._data)) < 1e-10
    
    # Qarray + QarrayArray
    assert jnp.max(jnp.abs((c + arr)._data - jqt.QarrayArray.create([a+c,b+c])._data)) < 1e-10

    # QarrayArray + QarrayArray
    assert jnp.max(jnp.abs((arr + arr)._data - jqt.QarrayArray.create([a+a,b+b])._data)) < 1e-10

    # QarrayArray + QarrayArray (of different size)
    with pytest.raises(ValueError):
        jqt.QarrayArray.create([a,b]) + jqt.QarrayArray.create([a,b,c])

# ========================================s
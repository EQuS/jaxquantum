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
    
    scalar = 1.23
    scalar_id_data = scalar*jnp.eye(2)

    arr = jqt.QarrayArray.create([a,b])

    # Scalar + Qarray 
    assert jnp.max(jnp.abs(((scalar+a)._data) - (scalar_id_data + a._data))) < 1e-10

    # Qarray + Scalar
    assert jnp.max(jnp.abs(((a+scalar)._data) - (a._data + scalar_id_data))) < 1e-10

    # Scalar + QarrayArray
    assert jnp.max(jnp.abs(((scalar + arr)._data - (scalar_id_data + arr._data)))) < 1e-10

    # QarrayArray + Scalar
    assert jnp.max(jnp.abs(((arr + scalar)._data - (arr._data + scalar_id_data))) < 1e-10)

    # Array + Qarray
    assert jnp.max(jnp.abs((a._data+b)._data - (a._data + b._data))) < 1e-10

    # Qarray + Array
    assert jnp.max(jnp.abs((a+b._data)._data - (a._data + b._data)) < 1e-10)

    # QarrayArray + Array 
    assert jnp.max(jnp.abs((arr + c._data)._data - (arr._data + c._data))) < 1e-10

    # Array + QarrayArray
    assert jnp.max(jnp.abs((c._data + arr)._data - (c._data + arr._data))) < 1e-10

    # QarrayArray + Qarray
    assert jnp.max(jnp.abs((arr + c)._data - (arr._data + c._data))) < 1e-10
    
    # Qarray + QarrayArray
    assert jnp.max(jnp.abs((c + arr)._data - (c._data + arr._data))) < 1e-10

    # Qarray + Qarray
    assert jnp.max(jnp.abs((a+b)._data - (a._data + b._data))) < 1e-10
    
    # QarrayArray + QarrayArray
    assert jnp.max(jnp.abs((arr + arr)._data - (arr._data + arr._data))) < 1e-10

    # QarrayArray + QarrayArray (of different size)
    with pytest.raises(ValueError):
        jqt.QarrayArray.create([a,b]) + jqt.QarrayArray.create([a,b,c])

def test_qarray_basic_math_sub():
    a = jqt.displace(2,1.0)
    b = jqt.displace(2,1.25)
    c = jqt.displace(2,1.5)
    
    scalar = 1.23
    scalar_id_data = scalar*jnp.eye(2)

    arr = jqt.QarrayArray.create([a,b])

    # Scalar - Qarray 
    assert jnp.max(jnp.abs(((scalar-a)._data) - (scalar_id_data - a._data))) < 1e-10

    # Qarray - Scalar
    assert jnp.max(jnp.abs(((a-scalar)._data) - (a._data - scalar_id_data))) < 1e-10

    # Scalar - QarrayArray
    assert jnp.max(jnp.abs((scalar - arr)._data - (scalar_id_data - arr._data))) < 1e-10

    # QarrayArray - Scalar
    assert jnp.max(jnp.abs((arr - scalar)._data - (arr._data - scalar_id_data))) < 1e-10

    # Array - Qarray
    assert jnp.max(jnp.abs((a._data-b)._data - (a._data - b._data))) < 1e-10

    # Qarray - Array
    assert jnp.max(jnp.abs((a-b._data)._data - (a._data - b._data))) < 1e-10

    # QarrayArray - Array 
    assert jnp.max(jnp.abs((arr - c._data)._data - (arr._data - c._data))) < 1e-10

    # Array - QarrayArray
    assert jnp.max(jnp.abs((c._data - arr)._data - (c._data - arr._data))) < 1e-10

    # QarrayArray - Qarray
    assert jnp.max(jnp.abs((arr - c)._data - (arr._data - c._data))) < 1e-10
    
    # Qarray - QarrayArray
    assert jnp.max(jnp.abs((c - arr)._data - (c._data - arr._data))) < 1e-10

    # Qarray - Qarray
    assert jnp.max(jnp.abs((a-b)._data - (a._data - b._data))) < 1e-10

    # QarrayArray - QarrayArray
    assert jnp.max(jnp.abs((arr - arr)._data - (arr._data - arr._data))) < 1e-10

    # QarrayArray - QarrayArray (of different size)
    with pytest.raises(ValueError):
        jqt.QarrayArray.create([a,b]) - jqt.QarrayArray.create([a,b,c])

# ========================================s
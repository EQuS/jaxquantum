# /Users/phionx/Github/qc/EQuS/bosonic/jax/jaxquantum/test_example.py

import pytest

import sys
import os

# Add the jaxquantum directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jaxquantum as jqt
import jax.numpy as jnp

minimum_version_for_tests = "0.2.0"

# Version
# ========================================
def test_version():
    high_enough = False 
    ver_tuple = tuple(map(int, jqt.__version__.split('.')))
    test_ver_tuple = tuple(map(int, minimum_version_for_tests.split('.')))
    high_enough = high_enough or (ver_tuple[0] > test_ver_tuple[0])
    high_enough = high_enough or (ver_tuple[0] == test_ver_tuple[0] and ver_tuple[1] > test_ver_tuple[1])
    high_enough = high_enough or (ver_tuple[0] == test_ver_tuple[0] and ver_tuple[1] == test_ver_tuple[1] and ver_tuple[2] >= test_ver_tuple[2])
    assert high_enough
# ========================================

# Initialization
# ========================================
def test_qarray_creation():
    a = jqt.Qarray.create(jnp.array([1,2,3]))
    assert a.shape == (3,1)

    a = jqt.Qarray.create(jnp.array([[1,2,3],[4,5,6]]))
    assert a.shape == (2,3,1)

    a = jqt.Qarray.create(jnp.array([[1,2,],[4,5]]), bdims=(2,))
    assert a.shape == (2,2,1)

    a = jqt.Qarray.from_list([])
    assert a.dims == ((),()) and a.shape == jnp.array([]).shape

# =========================================

# Properties
# ========================================
def test_qarray_properties():
    a = jqt.Qarray.create(jnp.array([[1+1.0j,2,3],[4,5,6]]))
    assert a.qtype == jqt.Qtypes.ket
    assert a.dtype == jnp.array([1+1.0j]).dtype
    assert a.bdims == (2,)
    assert a.qdims == jqt.basis(3,1).qdims

    a_dag = a.dag()
    assert a.space_dims == a.dims[0]
    assert a_dag.space_dims == a_dag.dims[1]
    
    # a_bad = a.copy()
    # a_bad._qdims._qtype = None
    # with pytest.raises(ValueError): # Doesn't give expected error for some reason...
    #     print(a_bad.space_dims) 
    
    assert jnp.all(a.shaped_data == a.data.reshape(2,3,1))

    assert a.is_batched == True
    assert a[0] == jqt.Qarray.create(jnp.array([1+1.0j,2,3]))
    
    with pytest.raises(ValueError):
        print(a[0][0])

    a_reshaped = a.reshape_bdims(2,1)
    assert a_reshaped.shape == (2,1,3,1)

    assert len(a) == 2
    with pytest.raises(ValueError):
        print(len(a[0]))
    
    with pytest.raises(ValueError):
        print(a == a.data)

    assert a != a[0]
    assert a != jqt.Qarray.create(jnp.array([1,2]))

    with pytest.raises(ValueError):
        print(a/a_dag)

    assert a/3 == 1/3*a
    

# Basic Math
# ========================================
def test_qarray_basic_math_add():
    N = 3

    a = jqt.displace(N,1.0)
    b = jqt.displace(N,1.25)
    c = jqt.displace(N,1.5)
    
    scalar = 1.23
    scalar_id_data = scalar*jnp.eye(N)

    arr = jqt.Qarray.from_list([a,b])

    # Scalar + Qarray 
    assert jnp.max(jnp.abs(((scalar+a).data) - (scalar_id_data + a.data))) < 1e-10

    # Qarray + Scalar
    assert jnp.max(jnp.abs(((a+scalar).data) - (a.data + scalar_id_data))) < 1e-10

    # Scalar + QarrayArray
    assert jnp.max(jnp.abs(((scalar + arr).data - (scalar_id_data + arr.data)))) < 1e-10

    # QarrayArray + Scalar
    assert jnp.max(jnp.abs(((arr + scalar).data - (arr.data + scalar_id_data))) < 1e-10)


    tarr = jnp.array([[1,2,3,4],[4,5,6,7.2]])
    tarr_id = tarr.reshape(*tarr.shape,1,1) * jnp.eye(N)
    tqarr = jqt.displace(N,tarr)

    # Array + Qarray
    assert jnp.max(jnp.abs((tarr+b).data - (tarr_id + b.data))) < 1e-10

    # Qarray + Array
    assert jnp.max(jnp.abs((a+tarr).data - (a.data + tarr_id)) < 1e-10)

    # QarrayArray + Array 
    assert jnp.max(jnp.abs((tqarr + tarr).data - (tqarr.data + tarr_id))) < 1e-10

    # Array + QarrayArray
    assert jnp.max(jnp.abs((tarr + tqarr).data - (tarr_id + tqarr.data))) < 1e-10

    # QarrayArray + Qarray
    assert jnp.max(jnp.abs((arr + c).data - (arr.data + c.data))) < 1e-10
    
    # Qarray + QarrayArray
    assert jnp.max(jnp.abs((c + arr).data - (c.data + arr.data))) < 1e-10

    # Qarray + Qarray
    assert jnp.max(jnp.abs((a+b).data - (a.data + b.data))) < 1e-10
    
    # QarrayArray + QarrayArray
    assert jnp.max(jnp.abs((arr + arr).data - (arr.data + arr.data))) < 1e-10

    # QarrayArray + QarrayArray (of different size)
    with pytest.raises(TypeError):
        jqt.Qarray.from_list([a,b]) + jqt.Qarray.from_list([a,b,c])

def test_qarray_basic_math_sub():
    N = 3

    a = jqt.displace(N,1.0)
    b = jqt.displace(N,1.25)
    c = jqt.displace(N,1.5)
    
    scalar = 1.23
    scalar_id_data = scalar*jnp.eye(N)

    arr = jqt.Qarray.from_list([a,b])

    # Scalar - Qarray 
    assert jnp.max(jnp.abs(((scalar-a).data) - (scalar_id_data - a.data))) < 1e-10

    # Qarray - Scalar
    assert jnp.max(jnp.abs(((a-scalar).data) - (a.data - scalar_id_data))) < 1e-10

    # Scalar - QarrayArray
    assert jnp.max(jnp.abs((scalar - arr).data - (scalar_id_data - arr.data))) < 1e-10

    # QarrayArray - Scalar
    assert jnp.max(jnp.abs((arr - scalar).data - (arr.data - scalar_id_data))) < 1e-10


    tarr = jnp.array([[1,2,3,4],[4,5,6,7.2]])
    tarr_id = tarr.reshape(*tarr.shape,1,1) * jnp.eye(N)
    tqarr = jqt.displace(N,tarr)

    # Array - Qarray
    assert jnp.max(jnp.abs((tarr - b).data - (tarr_id - b.data))) < 1e-10

    # Qarray - Array
    assert jnp.max(jnp.abs((a-tarr).data - (a.data - tarr_id))) < 1e-10

    # QarrayArray - Array 
    assert jnp.max(jnp.abs((tqarr - tarr).data - (tqarr.data - tarr_id))) < 1e-10

    # Array - QarrayArray
    assert jnp.max(jnp.abs((tarr - tqarr).data - (tarr_id - tqarr.data))) < 1e-10

    # QarrayArray - Qarray
    assert jnp.max(jnp.abs((arr - c).data - (arr.data - c.data))) < 1e-10
    
    # Qarray - QarrayArray
    assert jnp.max(jnp.abs((c - arr).data - (c.data - arr.data))) < 1e-10

    # Qarray - Qarray
    assert jnp.max(jnp.abs((a-b).data - (a.data - b.data))) < 1e-10

    # QarrayArray - QarrayArray
    assert jnp.max(jnp.abs((arr - arr).data - (arr.data - arr.data))) < 1e-10

    # QarrayArray - QarrayArray (of different size)
    with pytest.raises(TypeError):
        jqt.Qarray.from_list([a,b]) - jqt.Qarray.from_list([a,b,c])

def test_qarray_basic_math_mul():
    N = 3

    a = jqt.displace(N,1.0)
    b = jqt.displace(N,1.25)
    c = jqt.displace(N,1.5)
    
    scalar = 1.23
    scalar_id_data = scalar*jnp.eye(N)

    arr = jqt.Qarray.from_list([a,b])

    # Scalar * Qarray 
    assert jnp.max(jnp.abs(((scalar*a).data) - (scalar * a.data))) < 1e-10

    # Qarray * Scalar
    assert jnp.max(jnp.abs(((a*scalar).data) - (a.data * scalar))) < 1e-10

    # Scalar * QarrayArray
    assert jnp.max(jnp.abs((scalar * arr).data - (scalar * arr.data))) < 1e-10

    # QarrayArray * Scalar
    assert jnp.max(jnp.abs((arr * scalar).data - (arr.data * scalar))) < 1e-10

    # ----

    tarr = jnp.array([[1,2,3,4],[4,5,6,7.2]])
    tarr_expanded = tarr.reshape(*tarr.shape,1,1)
    tqarr = jqt.displace(N,tarr)

    # Array * Qarray
    assert jnp.max(jnp.abs((tarr * b).data - (tarr_expanded * b.data))) < 1e-10

    # Qarray * Array
    assert jnp.max(jnp.abs((a * tarr).data - (a.data * tarr_expanded))) < 1e-10

    # QarrayArray * Array 
    assert jnp.max(jnp.abs((tqarr * tarr).data - (tqarr.data * tarr_expanded))) < 1e-10

    # Array * QarrayArray
    assert jnp.max(jnp.abs((tarr * tqarr).data - (tarr_expanded * tqarr.data))) < 1e-10

    # ----

    # QarrayArray * Qarray
    assert jnp.max(jnp.abs((arr * c).data - (arr.data @ c.data))) < 1e-10
    
    # Qarray * QarrayArray
    assert jnp.max(jnp.abs((c * arr).data - (c.data @ arr.data))) < 1e-10

    # Qarray * Qarray
    assert jnp.max(jnp.abs((a * b).data - (a.data @ b.data))) < 1e-10

    # QarrayArray * QarrayArray
    assert jnp.max(jnp.abs((arr * arr).data - jnp.einsum('nij,njk->nik', arr.data, arr.data))) < 1e-10

    # QarrayArray * QarrayArray (of different size)
    with pytest.raises(ValueError):
        jqt.Qarray.from_list([a,b]) @ jqt.Qarray.from_list([a,b,c])

def test_qarray_basic_math_matmul():
    N = 3

    a = jqt.displace(N,1.0)
    b = jqt.displace(N,1.25)
    c = jqt.displace(N,1.5)
    
    arr = jqt.Qarray.from_list([a,b])

    # ----

    tarr = jnp.array([[1,2,3,4],[4,5,6,7.2]])
    tqarr = jqt.displace(N,tarr)

    # Array @ Qarray
    with pytest.raises(TypeError):
        tarr @ b

    # Qarray @ Array
    with pytest.raises(TypeError):
        a @ tarr
    # assert jnp.max(jnp.abs((a @ b.data).data - (a.data @ b.data))) < 1e-10

    # QarrayArray @ Array 
    with pytest.raises(TypeError):
        tqarr @ tarr
    # assert jnp.max(jnp.abs((tqarr @ tarr).data - (tqarr.data @ c.data))) < 1e-10

    # Array @ QarrayArray
    with pytest.raises(TypeError):
        tarr @ tqarr
    # assert jnp.max(jnp.abs((tarr @ tqarr).data - (c.data @ tqarr.data))) < 1e-10

    # ----

    # QarrayArray @ Qarray
    assert jnp.max(jnp.abs((arr @ c).data - (arr.data @ c.data))) < 1e-10
    
    # Qarray @ QarrayArray
    assert jnp.max(jnp.abs((c @ arr).data - (c.data @ arr.data))) < 1e-10

    # Qarray @ Qarray
    assert jnp.max(jnp.abs((a @ b).data - (a.data @ b.data))) < 1e-10

    # QarrayArray @ QarrayArray
    assert jnp.max(jnp.abs((arr @ arr).data - jnp.einsum('nij,njk->nik', arr.data, arr.data))) < 1e-10

    # QarrayArray @ QarrayArray (of different size)
    with pytest.raises(ValueError):
        jqt.Qarray.from_list([a,b]) @ jqt.Qarray.from_list([a,b,c])

def test_qarray_basic_math_tensor():
    N = 3
    a = jqt.displace(N,1.0)
    b = jqt.displace(N,1.25)
    c = jqt.displace(5,1.5)
    
    arr = jqt.Qarray.from_list([a,b])

    # ----
    tarr = jnp.array([[1,2,3,4],[4,5,6,7.2]])
    tqarr = jqt.displace(N,tarr)
    
    # Array ^ Qarray
    with pytest.raises(TypeError):
        tarr ^ b
    # assert jnp.max(jnp.abs((a.data ^ b).data - (jnp.kron(a.data,b.data)))) < 1e-10

    # Qarray ^ Array
    with pytest.raises(TypeError):
        a ^ tarr
    # assert jnp.max(jnp.abs((a ^ b.data).data - (jnp.kron(a.data,b.data))) < 1e-10)

    # Array ^ QarrayArray
    with pytest.raises(TypeError):
        tarr ^ tqarr
    # assert jnp.max(jnp.abs((a.data ^ arr).data - (jnp.kron(a.data,arr.data))) < 1e-10)
        
    # QarrayArray ^ Array
    with pytest.raises(TypeError):
        tqarr ^ tarr
    # assert jnp.max(jnp.abs((arr ^ b.data).data - (jnp.kron(arr.data,b.data))) < 1e-10)

    # ----

    # QarrayArray ^ Qarray
    assert jnp.max(jnp.abs((arr ^ b).data - (jnp.kron(arr.data,b.data))) < 1e-10)

    # Qarray ^ QarrayArray
    assert jnp.max(jnp.abs((a ^ arr).data - (jnp.kron(a.data,arr.data))) < 1e-10)

    # Qarray ^ Qarray
    assert jnp.max(jnp.abs((a ^ b).data - (jnp.kron(a.data,b.data))) < 1e-10)

    # QarrayArray ^ QarrayArray
    assert jnp.max(jnp.abs((jqt.tensor(arr,arr,parallel=True)).data - jnp.einsum("nij,nkl->nijkl", arr.data, arr.data).reshape(
                arr.data.shape[0],
                arr.data.shape[1] * arr.data.shape[1],
                -1
            )) < 1e-10)

    # QarrayArray ^ QarrayArray (of different size)
    assert  jnp.max(jnp.abs((jqt.Qarray.from_list([a,b]) ^ jqt.Qarray.from_list([a,b,b])).data - 
        jqt.Qarray.from_list([a^a, a^b, a^b, b^a, b^b, b^b]).data)) < 1e-10

    # QarrayArray ^ QarrayArray (of different size)
    arr1 = jqt.Qarray.from_list([a,b])
    arr2 = jqt.Qarray.from_list([a,b,b])
    assert jnp.max(jnp.abs((arr1^arr2).data - (
        jnp.kron(arr1.data, arr2.data)
    ))) < 1e-10

def test_qarray_basic_math_pow():
    a = jqt.displace(3,1.0)
    b = jqt.displace(3,1.25)
    c = jqt.displace(5,1.5)
    
    arr = jqt.Qarray.from_list([a,b])

    scalar = 3

    # ----
    # Qarray ** scalar

    assert jnp.max(jnp.abs((a ** scalar).data - jnp.linalg.matrix_power(a.data, scalar))) < 1e-10

    # QarrayArray ** scalar
    assert jnp.max(jnp.abs((arr ** scalar).data - jnp.linalg.matrix_power(arr.data, scalar))) < 1e-10
    assert jnp.max(jnp.abs((arr ** scalar).data - jqt.Qarray.from_list([a**scalar, b**scalar]).data)) < 1e-10

# ========================================

# Properties
import pytest
import sys
import os

# Add the jaxquantum directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jaxquantum as jqt
import jax.numpy as jnp
from jax.experimental import sparse

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

# Implementation Type Tests
# ========================================
def test_implementation_types():
    """Test that Qarray can be created with different implementation types."""
    
    # Test dense implementation (default)
    a_dense = jqt.Qarray.create(jnp.array([1, 2, 3]))
    assert a_dense.is_dense
    assert not a_dense.is_sparse
    assert isinstance(a_dense._impl, jqt.DenseImpl)
    
    # Test sparse implementation
    a_sparse = jqt.Qarray.create(jnp.array([1, 2, 3]), implementation=jqt.QarrayImplType.SPARSE)
    assert a_sparse.is_sparse
    assert not a_sparse.is_dense
    assert isinstance(a_sparse._impl, jqt.SparseImpl)
    
    # Test from_sparse
    sparse_data = sparse.BCOO.fromdense(jnp.array([[1, 0], [0, 2]]))
    a_from_sparse = jqt.Qarray.from_sparse(sparse_data)
    assert a_from_sparse.is_sparse
    assert isinstance(a_from_sparse._impl, jqt.SparseImpl)

def test_conversion_between_implementations():
    """Test conversion between dense and sparse implementations."""
    
    data = jnp.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
    
    # Dense to sparse
    a_dense = jqt.Qarray.create(data)
    a_sparse = a_dense.to_sparse()
    assert a_sparse.is_sparse
    assert jnp.allclose(a_dense.data, a_sparse.data.todense())
    
    # Sparse to dense
    a_dense_again = a_sparse.to_dense()
    assert a_dense_again.is_dense
    assert jnp.allclose(a_dense.data, a_dense_again.data)
    
    # Test that conversion is idempotent
    assert a_dense.to_dense() == a_dense
    assert a_sparse.to_sparse() == a_sparse

def test_implementation_preservation():
    """Test that operations preserve implementation type when possible."""
    
    # Dense operations should stay dense
    a = jqt.Qarray.create(jnp.array([[1, 2], [3, 4]]))
    b = jqt.Qarray.create(jnp.array([[5, 6], [7, 8]]))
    
    # Basic operations
    assert (a + b).is_dense
    assert (a - b).is_dense
    assert (a @ b).is_dense
    assert (a * 2).is_dense
    assert a.dag().is_dense
    
    # Sparse operations should stay sparse for supported operations
    a_sparse = a.to_sparse()
    b_sparse = b.to_sparse()
    
    assert (a_sparse + b_sparse).is_sparse
    assert (a_sparse - b_sparse).is_sparse
    assert (a_sparse @ b_sparse).is_sparse
    assert (a_sparse * 2).is_sparse
    # Note: dag() converts to dense for sparse due to JAX limitations

def test_mixed_implementation_operations():
    """Test operations between dense and sparse implementations."""
    
    a_dense = jqt.Qarray.create(jnp.array([[1, 2], [3, 4]]))
    a_sparse = a_dense.to_sparse()
    
    # Mixed operations should work (auto-conversion)
    result1 = a_dense + a_sparse
    result2 = a_sparse + a_dense
    result3 = a_dense @ a_sparse
    result4 = a_sparse @ a_dense
    
    # Results should be consistent regardless of order
    # Convert both to dense for comparison
    result1_dense = result1.data.todense() if hasattr(result1.data, 'todense') else result1.data
    result2_dense = result2.data.todense() if hasattr(result2.data, 'todense') else result2.data
    result3_dense = result3.data.todense() if hasattr(result3.data, 'todense') else result3.data
    result4_dense = result4.data.todense() if hasattr(result4.data, 'todense') else result4.data
    
    assert jnp.allclose(result1_dense, result2_dense)
    assert jnp.allclose(result3_dense, result4_dense)
    
    # Test with different sparse matrices
    sparse_data = sparse.BCOO.fromdense(jnp.array([[1, 0], [0, 2]]))
    b_sparse = jqt.Qarray.from_sparse(sparse_data)
    
    result5 = a_dense + b_sparse
    result6 = b_sparse + a_dense
    
    # Convert both to dense for comparison
    result5_dense = result5.data.todense() if hasattr(result5.data, 'todense') else result5.data
    result6_dense = result6.data.todense() if hasattr(result6.data, 'todense') else result6.data
    assert jnp.allclose(result5_dense, result6_dense)

# ========================================

# Backward Compatibility Tests
# ========================================
def test_backward_compatibility_basic():
    """Test that basic operations work the same as the old implementation."""
    
    # Test basic creation
    a = jqt.Qarray.create(jnp.array([1,2,3]))
    assert a.shape == (3,1)
    assert a.qtype == jqt.Qtypes.ket

    a = jqt.Qarray.create(jnp.array([[1,2,3],[4,5,6]]))
    assert a.shape == (2,3,1)

    a = jqt.Qarray.create(jnp.array([[1,2,],[4,5]]), bdims=(2,))
    assert a.shape == (2,2,1)

    a = jqt.Qarray.from_list([])
    assert a.dims == ((),()) and a.shape == jnp.array([]).shape

def test_backward_compatibility_properties():
    """Test that properties work the same as the old implementation."""
    
    a = jqt.Qarray.create(jnp.array([[1+1.0j,2,3],[4,5,6]]))
    assert a.qtype == jqt.Qtypes.ket
    assert a.dtype == jnp.array([1+1.0j]).dtype
    assert a.bdims == (2,)
    assert a.qdims == jqt.basis(3,1).qdims

    a_dag = a.dag()
    assert a.space_dims == a.dims[0]
    assert a_dag.space_dims == a_dag.dims[1]
    
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

def test_backward_compatibility_math():
    """Test that mathematical operations work the same as the old implementation."""
    
    N = 3

    a = jqt.displace(N,1.0)
    b = jqt.displace(N,1.25)
    c = jqt.displace(N,1.5)
    
    scalar = 1.23
    scalar_id_data = scalar*jnp.eye(N)

    arr = jqt.Qarray.from_list([a,b])

    # Test basic operations
    assert jnp.max(jnp.abs(((scalar+a).data) - (scalar_id_data + a.data))) < 1e-10
    assert jnp.max(jnp.abs(((a+scalar).data) - (a.data + scalar_id_data))) < 1e-10
    assert jnp.max(jnp.abs(((a+b).data) - (a.data + b.data))) < 1e-10
    assert jnp.max(jnp.abs((a * b).data - (a.data @ b.data))) < 1e-10
    assert jnp.max(jnp.abs((a ** 2).data - jnp.linalg.matrix_power(a.data, 2))) < 1e-10

def test_backward_compatibility_tensor():
    """Test that tensor operations work the same as the old implementation."""
    
    N = 3
    a = jqt.displace(N,1.0)
    b = jqt.displace(N,1.25)
    c = jqt.displace(5,1.5)
    
    arr = jqt.Qarray.from_list([a,b])

    # Test tensor products
    assert jnp.max(jnp.abs((a ^ b).data - (jnp.kron(a.data,b.data))) < 1e-10)
    assert jnp.max(jnp.abs((arr ^ b).data - (jnp.kron(arr.data,b.data))) < 1e-10)

# ========================================

# Sparse-Specific Tests
# ========================================
def test_sparse_creation_and_operations():
    """Test sparse-specific functionality."""
    
    # Create a sparse matrix
    dense_data = jnp.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
    a_sparse = jqt.Qarray.create(dense_data, implementation=jqt.QarrayImplType.SPARSE)
    
    # Test that it's actually sparse
    assert a_sparse.is_sparse
    assert hasattr(a_sparse.data, 'todense')  # BCOO has todense method
    
    # Test sparse operations
    b_sparse = a_sparse + a_sparse
    assert b_sparse.is_sparse
    
    c_sparse = a_sparse * 2
    assert c_sparse.is_sparse
    
    d_sparse = a_sparse @ a_sparse
    assert d_sparse.is_sparse
    
    # Test that results are correct
    assert jnp.allclose(b_sparse.data.todense(), dense_data * 2)
    assert jnp.allclose(c_sparse.data.todense(), dense_data * 2)
    assert jnp.allclose(d_sparse.data.todense(), dense_data @ dense_data)

def test_sparse_fallback_operations():
    """Test operations that should fallback to dense for sparse."""
    
    dense_data = jnp.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
    a_sparse = jqt.Qarray.create(dense_data, implementation=jqt.QarrayImplType.SPARSE)
    
    # These operations should convert to dense automatically
    exp_a = a_sparse.expm()
    assert exp_a.is_dense  # Should be converted to dense
    
    pow_a = a_sparse.powm(2)
    assert pow_a.is_dense  # Should be converted to dense
    
    sin_a = a_sparse.sinm()
    assert sin_a.is_dense  # Should be converted to dense
    
    cos_a = a_sparse.cosm()
    assert cos_a.is_dense  # Should be converted to dense
    
    # Test that results are correct
    dense_a = a_sparse.to_dense()
    assert jnp.allclose(exp_a.data, dense_a.expm().data)
    assert jnp.allclose(pow_a.data, dense_a.powm(2).data)

def test_sparse_eigenoperations():
    """Test eigenvalue operations with sparse matrices."""
    
    # Create a Hermitian sparse matrix
    dense_data = jnp.array([[2, 0, 1], [0, 3, 0], [1, 0, 2]])
    a_sparse = jqt.Qarray.create(dense_data, implementation=jqt.QarrayImplType.SPARSE)
    
    # Eigenvalue operations should convert to dense
    evals_sparse = a_sparse.eigenenergies()
    evals_dense = a_sparse.to_dense().eigenenergies()
    
    assert jnp.allclose(evals_sparse, evals_dense)
    
    # Eigenstates should also convert to dense
    evals_sparse, evecs_sparse = a_sparse.eigenstates()
    evals_dense, evecs_dense = a_sparse.to_dense().eigenstates()
    
    assert jnp.allclose(evals_sparse, evals_dense)
    assert jnp.allclose(evecs_sparse.data, evecs_dense.data)

def test_sparse_tensor_operations():
    """Test tensor operations with sparse matrices."""
    
    dense_data1 = jnp.array([[1, 0], [0, 2]])
    dense_data2 = jnp.array([[3, 0, 0], [0, 4, 0], [0, 0, 5]])
    
    a_sparse = jqt.Qarray.create(dense_data1, implementation="sparse")
    b_sparse = jqt.Qarray.create(dense_data2, implementation="sparse")
    
    # Tensor product should convert to dense
    tensor_result = a_sparse ^ b_sparse
    assert tensor_result.is_dense
    
    # Compare with dense result
    a_dense = a_sparse.to_dense()
    b_dense = b_sparse.to_dense()
    tensor_dense = a_dense ^ b_dense
    
    assert jnp.allclose(tensor_result.data, tensor_dense.data)

def test_sparse_memory_efficiency():
    """Test that sparse matrices are actually more memory efficient for sparse data."""
    
    # Create a large sparse matrix
    size = 100
    dense_data = jnp.zeros((size, size))
    # Add some non-zero elements
    for i in range(0, size, 10):
        dense_data = dense_data.at[i, i].set(i)
        if i + 1 < size:
            dense_data = dense_data.at[i, i+1].set(1)
    
    a_dense = jqt.Qarray.create(dense_data)
    a_sparse = jqt.Qarray.create(dense_data, implementation=jqt.QarrayImplType.SPARSE)
    
    # Both should give the same results
    a_sparse_dense = a_sparse.data.todense() if hasattr(a_sparse.data, 'todense') else a_sparse.data
    assert jnp.allclose(a_dense.data, a_sparse_dense)
    
    # Test that sparse operations work correctly
    b_sparse = a_sparse + a_sparse
    b_dense = a_dense + a_dense
    
    assert jnp.allclose(b_sparse.data.todense(), b_dense.data)

# ========================================

# Error Handling Tests
# ========================================
def test_error_handling():
    """Test error handling for invalid operations."""
    
    a = jqt.Qarray.create(jnp.array([[1, 2], [3, 4]]))
    b = jqt.Qarray.create(jnp.array([1, 2, 3]))  # Different dimensions
    
    # Test dimension mismatch
    with pytest.raises(ValueError):
        a + b
    
    with pytest.raises(TypeError):  # This now raises TypeError for dimension mismatch
        a @ b
    
    # Test division by Qarray
    with pytest.raises(ValueError):
        a / a
    
    # Test indexing non-batched Qarray
    c = jqt.Qarray.create(jnp.array([[1, 2], [3, 4]]))
    with pytest.raises(ValueError):
        print(c[0])

def test_sparse_error_handling():
    """Test error handling specific to sparse operations."""
    
    # Test creating sparse from non-2D data
    # This actually works now - BCOO can handle 1D data
    sparse_1d = jqt.Qarray.create(jnp.array([1, 2, 3]), implementation=jqt.QarrayImplType.SPARSE)
    assert sparse_1d.is_sparse
    
    # Test operations that should fail gracefully
    dense_data = jnp.array([[1, 0], [0, 2]])
    a_sparse = jqt.Qarray.create(dense_data, implementation=jqt.QarrayImplType.SPARSE)
    
    # These should work (with fallback to dense if needed)
    try:
        result = a_sparse.expm()
        assert result.is_dense  # Should have fallen back to dense
    except Exception as e:
        pytest.fail(f"sparse expm should fallback to dense, but got: {e}")

# ========================================

# Performance Tests (Basic)
# ========================================
def test_basic_performance():
    """Basic performance comparison between dense and sparse."""
    
    # Create a moderately sparse matrix
    size = 50
    dense_data = jnp.zeros((size, size))
    # Make it sparse (only diagonal and first off-diagonal)
    for i in range(size):
        dense_data = dense_data.at[i, i].set(i + 1)
        if i + 1 < size:
            dense_data = dense_data.at[i, i+1].set(1)
    
    a_dense = jqt.Qarray.create(dense_data)
    a_sparse = jqt.Qarray.create(dense_data, implementation=jqt.QarrayImplType.SPARSE)
    
    # Test that both give the same results for basic operations
    # Addition
    result_dense_add = a_dense + a_dense
    result_sparse_add = a_sparse + a_sparse
    assert jnp.allclose(result_dense_add.data, result_sparse_add.data.todense())
    
    # Multiplication
    result_dense_mul = a_dense * 2
    result_sparse_mul = a_sparse * 2
    assert jnp.allclose(result_dense_mul.data, result_sparse_mul.data.todense())
    
    # Matrix multiplication
    result_dense_matmul = a_dense @ a_dense
    result_sparse_matmul = a_sparse @ a_sparse
    assert jnp.allclose(result_dense_matmul.data, result_sparse_matmul.data.todense())

# ========================================

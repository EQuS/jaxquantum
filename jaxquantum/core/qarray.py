"""New Qarray implementation with sparse support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from flax import struct
from jax import Array, config, vmap
from typing import List, Union, TypeVar, Generic, overload, Literal, Any
import jax.numpy as jnp
import jax.scipy as jsp
from jax.experimental import sparse
from numpy import ndarray
from copy import deepcopy
from math import prod
from enum import Enum

from jaxquantum.core.settings import SETTINGS
from jaxquantum.utils.utils import robust_isscalar
from jaxquantum.core.dims import Qtypes, Qdims, check_dims, ket_from_op_dims

config.update("jax_enable_x64", True)

# Type variable for implementation types
ImplT = TypeVar("ImplT", bound="QarrayImpl")


class QarrayImplType(Enum):
    """Enumeration of available Qarray implementation types."""
    DENSE = "dense"
    SPARSE = "sparse"

    @classmethod
    def has(cls, x) -> bool:
        """Return True if x corresponds to a member of QarrayImplType.

        Accepts:
        - an existing QarrayImplType member
        - a string equal to the member name or value (case-insensitive)
        - an implementation class (e.g. DenseImpl, SparseImpl) if available
        """
        if isinstance(x, cls):
            return True

        if isinstance(x, str):
            xl = x.lower()
            return any(xl == member.value or xl == member.name.lower() for member in cls)

        # Try mapping from an implementation class to an enum member
        try:
            cls.from_impl_class(x)
            return True
        except Exception:
            return False
    
    @classmethod
    def from_impl_class(cls, impl_class) -> "QarrayImplType":
        """Get implementation type from implementation class."""
        if impl_class == DenseImpl:
            return cls.DENSE
        elif impl_class == SparseImpl:
            return cls.SPARSE
        else:
            raise ValueError(f"Unknown implementation class: {impl_class}")
    
    def get_impl_class(self):
        """Get the implementation class for this type."""
        if self == QarrayImplType.DENSE:
            return DenseImpl
        elif self == QarrayImplType.SPARSE:
            return SparseImpl
        else:
            raise ValueError(f"No implementation class for type: {self}")




def robust_asarray(data) -> Union[Array, sparse.BCOO]:
    """Convert data to JAX array or sparse BCOO array."""
    if isinstance(data, sparse.BCOO):
        return data
    return jnp.asarray(data)

class QarrayImpl(ABC):
    """Abstract base class for Qarray implementations."""
    
    @abstractmethod
    def get_data(self) -> Array:
        """Get the underlying data array."""
        pass

    @property 
    def data(self) -> Array:
        return self.get_data()

    @property
    def impl_type(self) -> QarrayImplType:
        return QarrayImplType.from_impl_class(type(self))
    
    @abstractmethod
    def matmul(self, other: "QarrayImpl") -> "QarrayImpl":
        """Matrix multiplication."""
        pass
    
    @abstractmethod
    def add(self, other: "QarrayImpl") -> "QarrayImpl":
        """Addition."""
        pass
    
    @abstractmethod
    def sub(self, other: "QarrayImpl") -> "QarrayImpl":
        """Subtraction."""
        pass
    
    @abstractmethod
    def mul(self, scalar) -> "QarrayImpl":
        """Scalar multiplication."""
        pass
    
    @abstractmethod
    def dag(self) -> "QarrayImpl":
        """Conjugate transpose."""
        pass
    
    @abstractmethod
    def to_dense(self) -> "DenseImpl":
        """Convert to dense implementation."""
        pass
    
    @abstractmethod
    def to_sparse(self) -> "SparseImpl":
        """Convert to sparse implementation."""
        pass
    
    @abstractmethod
    def shape(self) -> tuple:
        """Get shape of data."""
        pass
    
    @abstractmethod
    def dtype(self):
        """Get dtype of data."""
        pass

    @abstractmethod
    def __deepcopy__(self, memo=None):
        pass

    @abstractmethod
    def tidy_up(self, atol):
        """Tidy up small values in data."""
        pass

@struct.dataclass
class DenseImpl(QarrayImpl):
    """Dense implementation using JAX dense arrays."""
    _data: Array
    
    def get_data(self) -> Array:
        return self._data
    
    def matmul(self, other: QarrayImpl) -> QarrayImpl:
        if isinstance(other, DenseImpl):
            return DenseImpl(self._data @ other._data)
        elif isinstance(other, SparseImpl):
            # Convert sparse to dense for matmul
            dense_other = other.to_dense()
            return DenseImpl(self._data @ dense_other._data)
        else:
            raise TypeError(f"Unsupported type for matmul: {type(other)}")
    
    def add(self, other: QarrayImpl) -> QarrayImpl:
        if isinstance(other, DenseImpl):
            return DenseImpl(self._data + other._data)
        elif isinstance(other, SparseImpl):
            # Convert sparse to dense for addition
            dense_other = other.to_dense()
            return DenseImpl(self._data + dense_other._data)
        else:
            raise TypeError(f"Unsupported type for add: {type(other)}")
    
    def sub(self, other: QarrayImpl) -> QarrayImpl:
        if isinstance(other, DenseImpl):
            return DenseImpl(self._data - other._data)
        elif isinstance(other, SparseImpl):
            # Convert sparse to dense for subtraction
            dense_other = other.to_dense()
            return DenseImpl(self._data - dense_other._data)
        else:
            raise TypeError(f"Unsupported type for sub: {type(other)}")
    
    def mul(self, scalar) -> QarrayImpl:
        return DenseImpl(scalar * self._data)
    
    def dag(self) -> QarrayImpl:
        return DenseImpl(jnp.moveaxis(jnp.conj(self._data), -1, -2))
    
    def to_dense(self) -> "DenseImpl":
        return self
    
    def to_sparse(self) -> "SparseImpl":
        sparse_data = sparse.BCOO.fromdense(self._data)
        return SparseImpl(sparse_data)
    
    def shape(self) -> tuple:
        return self._data.shape
    
    def dtype(self):
        return self._data.dtype
    
    def frobenius_norm(self) -> float:
        """Compute Frobenius norm."""
        return jnp.sqrt(jnp.sum(jnp.abs(self._data) ** 2))
    
    def real(self) -> QarrayImpl:
        """Element-wise real part."""
        return DenseImpl(jnp.real(self._data))
    
    def imag(self) -> QarrayImpl:
        """Element-wise imaginary part."""
        return DenseImpl(jnp.imag(self._data))
    
    def conj(self) -> QarrayImpl:
        """Element-wise complex conjugate."""
        return DenseImpl(jnp.conj(self._data))
    
    def __deepcopy__(self, memo=None):
        return DenseImpl(
            _data=deepcopy(self._data, memo)
        )

    def tidy_up(self, atol):
        """Tidy up small values in data."""

        data = self._data
        data_re = jnp.real(data)
        data_im = jnp.imag(data)
        data_re_mask = jnp.abs(data_re) > atol
        data_im_mask = jnp.abs(data_im) > atol
        data_new = data_re * data_re_mask + 1j * data_im * data_im_mask

        return DenseImpl(
            _data=data_new
        )

@struct.dataclass
class SparseImpl(QarrayImpl):
    """Sparse implementation using JAX sparse BCOO arrays."""
    _data: sparse.BCOO
    
    def get_data(self) -> Array:
        return self._data
    
    def matmul(self, other: QarrayImpl) -> QarrayImpl:
        if isinstance(other, DenseImpl):
            # Convert sparse to dense for matmul
            dense_self = self.to_dense()
            return DenseImpl(dense_self._data @ other._data)

        elif isinstance(other, SparseImpl):
            return SparseImpl(self._data @ other._data)
        else:
            raise TypeError(f"Unsupported type for matmul: {type(other)}")
    
    def add(self, other: QarrayImpl) -> QarrayImpl:
        if isinstance(other, DenseImpl):
            # Convert sparse to dense for addition
            dense_self = self.to_dense()
            return DenseImpl(dense_self._data + other._data)
        elif isinstance(other, SparseImpl):
            return SparseImpl(self._data + other._data)
        else:
            raise TypeError(f"Unsupported type for add: {type(other)}")
    
    def sub(self, other: QarrayImpl) -> QarrayImpl:
        if isinstance(other, DenseImpl):
            # Convert sparse to dense for subtraction
            dense_self = self.to_dense()
            return DenseImpl(dense_self._data - other._data)
        elif isinstance(other, SparseImpl):
            return SparseImpl(self._data - other._data)
        else:
            raise TypeError(f"Unsupported type for sub: {type(other)}")
    
    def mul(self, scalar) -> QarrayImpl:
        return SparseImpl(scalar * self._data)
    
    def dag(self) -> QarrayImpl:
        # Implement sparse conjugate transpose directly
        # Transpose the sparse matrix (last two dimensions only) and conjugate the data
        ndim = self._data.ndim
        if ndim >= 2:
            # Create permutation that swaps only the last two dimensions
            permutation = tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
            transposed_data = sparse.bcoo_transpose(self._data, permutation=permutation)
        else:
            transposed_data = self._data
        
        conjugated_data = sparse.BCOO((jnp.conj(transposed_data.data), transposed_data.indices), 
                                      shape=transposed_data.shape)
        return SparseImpl(conjugated_data)
    
    def to_dense(self) -> "DenseImpl":
        return DenseImpl(self._data.todense())
    
    @classmethod
    def _to_sparse(cls, data) -> sparse.BCOO:
        if isinstance(data, sparse.BCOO):
            return data
        return sparse.BCOO.fromdense(data)

    def to_sparse(self) -> "SparseImpl":
        return self
    
    def shape(self) -> tuple:
        return self._data.shape
    
    def dtype(self):
        return self._data.dtype
    
    def frobenius_norm(self) -> float:
        """Compute Frobenius norm directly from sparse data."""
        # For sparse matrices, we can compute the Frobenius norm as sqrt(sum(|data|^2))
        # This avoids converting to dense
        return jnp.sqrt(jnp.sum(jnp.abs(self._data.data) ** 2))
    
    @classmethod
    def _real(cls, data):
        return sparse.BCOO(
            (jnp.real(data.data), data.indices), 
            shape=data.shape
        )

    def real(self) -> QarrayImpl:
        """Element-wise real part."""
        return SparseImpl(SparseImpl._real(self._data))
    
    @classmethod
    def _imag(cls, data):
        return sparse.BCOO(
            (jnp.imag(data.data), data.indices), 
            shape=data.shape
        )

    def imag(self) -> QarrayImpl:
        """Element-wise imaginary part."""
        return SparseImpl(SparseImpl._imag(self._data))
    
    @classmethod
    def _conj(cls, data):
        return sparse.BCOO(
            (jnp.conj(data.data), data.indices), 
            shape=data.shape
        )

    def conj(self) -> QarrayImpl:
        """Element-wise complex conjugate."""
        return SparseImpl(SparseImpl._conj(self._data))

    @classmethod
    def _abs(cls, data):
        return sparse.sparsify(jnp.abs)(data)

    def abs(self) -> QarrayImpl:
        """Element-wise absolute value."""
        return SparseImpl(SparseImpl._abs(self._data))
    
    def __deepcopy__(self, memo=None):
        return SparseImpl(
            _data=deepcopy(self._data, memo)
        )

    def tidy_up(self, atol):
    #     """Tidy up small values in data."""

    #     data = self._data
    #     data_re = SparseImpl._real(data)
    #     data_im = SparseImpl._imag(data)
    #     data_re_mask = SparseImpl._abs(data_re) > atol
    #     data_im_mask = SparseImpl._abs(data_im) > atol # NOTE: This does not work for sparse arrays
    #     data_new = data_re * data_re_mask + 1j * data_im * data_im_mask

    #     return SparseImpl(
    #         _data=data_new
    #     )
        pass

@struct.dataclass
class Qarray(Generic[ImplT]):
    """Quantum array with implementation-based architecture."""
    _impl: ImplT
    _qdims: Qdims = struct.field(pytree_node=False)
    _bdims: tuple[int] = struct.field(pytree_node=False)

    # Initialization ----
    @classmethod
    @classmethod
    @overload
    def create(cls, data, dims=None, bdims=None, implementation: Literal[QarrayImplType.DENSE] = QarrayImplType.DENSE) -> "Qarray[DenseImpl]":
        ...

    @classmethod
    @overload
    def create(cls, data, dims=None, bdims=None, implementation: Literal[QarrayImplType.SPARSE] = ...) -> "Qarray[SparseImpl]":
        ...

    @classmethod
    @overload
    def create(cls, data, dims=None, bdims=None, implementation=...) -> "Qarray[DenseImpl]":
        ...

    @classmethod
    def create(cls, data, dims=None, bdims=None, implementation=QarrayImplType.DENSE):
        """Create a Qarray from data.

        Args:
            data: Input data array
            dims: Quantum dimensions
            bdims: Batch dimensions
            implementation: QarrayImplType.DENSE or QarrayImplType.SPARSE
        """
        # Step 1: Prepare data ----
        data = robust_asarray(data)

        if len(data.shape) == 1 and data.shape[0] > 0:
            data = data.reshape(data.shape[0], 1)

        if len(data.shape) >= 2:
            if data.shape[-2] != data.shape[-1] and not (
                data.shape[-2] == 1 or data.shape[-1] == 1
            ):
                data = data.reshape(*data.shape[:-1], data.shape[-1], 1)

        if bdims is not None:
            if len(data.shape) - len(bdims) == 1:
                data = data.reshape(*data.shape[:-1], data.shape[-1], 1)
        # ----

        # Step 2: Prepare dimensions ----
        if bdims is None:
            bdims = tuple(data.shape[:-2])

        if dims is None:
            dims = ((data.shape[-2],), (data.shape[-1],))

        if not isinstance(dims[0], (list, tuple)):
            # This handles the case where only the hilbert space dimensions are sent in.
            if data.shape[-1] == 1:
                dims = (tuple(dims), tuple([1 for _ in dims]))
            elif data.shape[-2] == 1:
                dims = (tuple([1 for _ in dims]), tuple(dims))
            else:
                dims = (tuple(dims), tuple(dims))
        else:
            dims = (tuple(dims[0]), tuple(dims[1]))

        check_dims(dims, bdims, data.shape)

        qdims = Qdims(dims)

        # NOTE: Constantly tidying up on Qarray creation might be a bit overkill.
        # It increases the compilation time, but only very slightly
        # increased the runtime of the jit compiled function.
        # We could instead use this tidy up where we think we need it.
        
        implementation = QarrayImplType(implementation)
        if implementation == QarrayImplType.SPARSE:
            impl = SparseImpl(SparseImpl._to_sparse(data))
            # impl = impl.tidy_up(SETTINGS["auto_tidyup_atol"])
            # Sparse tidy up is currently not implemented.

        elif implementation == QarrayImplType.DENSE:
            impl = DenseImpl(data)
            impl = impl.tidy_up(SETTINGS["auto_tidyup_atol"])

        return cls(impl, qdims, bdims)

    @classmethod
    @classmethod
    @overload
    def from_sparse(cls, data, dims=None, bdims=None) -> "Qarray[SparseImpl]":
        ...

    @classmethod
    def from_sparse(cls, data, dims=None, bdims=None):
        """Create a Qarray from sparse data."""
        return cls.create(data.todense(), dims=dims, bdims=bdims, implementation=QarrayImplType.SPARSE)

    @classmethod
    @classmethod
    @overload
    def from_list(cls, qarr_list: List["Qarray[DenseImpl]"]) -> "Qarray[DenseImpl]":
        ...

    @classmethod
    @overload
    def from_list(cls, qarr_list: List["Qarray[SparseImpl]"]) -> "Qarray[SparseImpl]":
        ...

    @classmethod
    def from_list(cls, qarr_list: List[Qarray]) -> Qarray:
        """Create a Qarray from a list of Qarrays."""

        data = jnp.array([qarr.data for qarr in qarr_list])

        if len(qarr_list) == 0:
            dims = ((), ())
            bdims = ()
        else:
            dims = qarr_list[0].dims
            bdims = qarr_list[0].bdims
            # Check if all have the same implementation type
            impl_type = type(qarr_list[0]._impl)

        if not all(qarr.dims == dims and qarr.bdims == bdims for qarr in qarr_list):
            raise ValueError("All Qarrays in the list must have the same dimensions.")

        bdims = (len(qarr_list),) + bdims

        # Check if all implementations are the same type
        if len(qarr_list) > 0 and all(isinstance(qarr._impl, impl_type) for qarr in qarr_list):
            implementation = QarrayImplType.from_impl_class(impl_type)
        else:
            # If mixed or empty, default to dense
            implementation = QarrayImplType.DENSE

        return cls.create(data, dims=dims, bdims=bdims, implementation=implementation)

    @classmethod
    @classmethod
    @overload
    def from_array(cls, qarr_arr: "Qarray[DenseImpl]") -> "Qarray[DenseImpl]":
        ...

    @classmethod
    @overload
    def from_array(cls, qarr_arr: "Qarray[SparseImpl]") -> "Qarray[SparseImpl]":
        ...

    @classmethod
    def from_array(cls, qarr_arr) -> Qarray:
        """Create a Qarray from a nested list of Qarrays.

        Args:
            qarr_arr (list): nested list of Qarrays

        Returns:
            Qarray: Qarray object
        """
        if isinstance(qarr_arr, Qarray):
            return qarr_arr

        bdims = ()
        lvl = qarr_arr
        while not isinstance(lvl, Qarray):
            bdims = bdims + (len(lvl),)
            if len(lvl) > 0:
                lvl = lvl[0]
            else:
                break

        def flat(lis):
            flatList = []
            for element in lis:
                if type(element) is list:
                    flatList += flat(element)
                else:
                    flatList.append(element)
            return flatList

        qarr_list = flat(qarr_arr)
        qarr = cls.from_list(qarr_list)
        qarr = qarr.reshape_bdims(*bdims)
        return qarr

    # Properties ----
    @property
    def qtype(self):
        return self._qdims.qtype

    @property
    def dtype(self):
        return self._impl.dtype()

    @property
    def dims(self):
        return self._qdims.dims

    @property
    def bdims(self):
        return self._bdims

    @property
    def qdims(self):
        return self._qdims

    @property
    def space_dims(self):
        if self.qtype in [Qtypes.oper, Qtypes.ket]:
            return self.dims[0]
        elif self.qtype == Qtypes.bra:
            return self.dims[1]
        else:
            # TODO: not reached for some reason
            raise ValueError("Unsupported qtype.")

    @property
    def data(self):
        return self._impl.data

    @property
    def shaped_data(self):
        return self.data.reshape(self.bdims + self.dims[0] + self.dims[1])

    @property
    def shape(self):
        return self.data.shape

    @property
    def is_batched(self):
        return len(self.bdims) > 0

    @property
    def is_sparse(self):
        return self._impl.impl_type == QarrayImplType.SPARSE

    @property
    def is_dense(self):
        return self._impl.impl_type == QarrayImplType.DENSE
    
    @property
    def impl_type(self):
        """Get the implementation type."""
        return self._impl.impl_type

    def to_sparse(self) -> "Qarray[SparseImpl]":
        """Convert to sparse implementation."""
        if self.is_sparse:
            return self
        new_impl = self._impl.to_sparse()
        return Qarray(new_impl, self._qdims, self._bdims)

    def to_dense(self) -> "Qarray[DenseImpl]":
        """Convert to dense implementation."""
        if self.is_dense:
            return self
        new_impl = self._impl.to_dense()
        return Qarray(new_impl, self._qdims, self._bdims)

    def __getitem__(self, index):
        if len(self.bdims) > 0:
            return Qarray.create(
                self.data[index],
                dims=self.dims,
            )
        else:
            raise ValueError("Cannot index a non-batched Qarray.")

    def reshape_bdims(self, *args):
        """Reshape the batch dimensions of the Qarray."""
        new_bdims = tuple(args)

        if prod(new_bdims) == 0:
            new_shape = new_bdims
        else:
            new_shape = new_bdims + (prod(self.dims[0]),) + (-1,)
        
        # Preserve implementation type
        implementation = self.impl_type
        return Qarray.create(
            self.data.reshape(new_shape),
            dims=self.dims,
            bdims=new_bdims,
            implementation=implementation,
        )

    def space_to_qdims(self, space_dims: List[int]):
        if isinstance(space_dims[0], (list, tuple)):
            return space_dims

        if self.qtype in [Qtypes.oper, Qtypes.ket]:
            return (tuple(space_dims), tuple([1 for _ in range(len(space_dims))]))
        elif self.qtype == Qtypes.bra:
            return (tuple([1 for _ in range(len(space_dims))]), tuple(space_dims))
        else:
            raise ValueError("Unsupported qtype for space_to_qdims conversion.")

    def reshape_qdims(self, *args):
        """Reshape the quantum dimensions of the Qarray.

        Note that this does not take in qdims but rather the new Hilbert space dimensions.

        Args:
            *args: new Hilbert dimensions for the Qarray.

        Returns:
            Qarray: reshaped Qarray.
        """

        new_space_dims = tuple(args)
        current_space_dims = self.space_dims
        assert prod(new_space_dims) == prod(current_space_dims)

        new_qdims = self.space_to_qdims(new_space_dims)
        new_bdims = self.bdims

        # Preserve implementation type
        implementation = self.impl_type
        return Qarray.create(self.data, dims=new_qdims, bdims=new_bdims, implementation=implementation)

    def resize(self, new_shape):
        """Resize the Qarray to a new shape.

        TODO: review and maybe deprecate this method.
        """
        dims = self.dims
        data = jnp.resize(self.data, new_shape)
        # Preserve implementation type
        implementation = self.impl_type
        return Qarray.create(
            data,
            dims=dims,
            implementation=implementation,
        )

    def __len__(self):
        """Length of the Qarray."""
        if len(self.bdims) > 0:
            return self.data.shape[0]
        else:
            raise ValueError("Cannot get length of a non-batched Qarray.")

    def __eq__(self, other):
        if not isinstance(other, Qarray):
            raise ValueError("Cannot calculate equality of a Qarray with a non-Qarray.")

        if self.dims != other.dims:
            return False

        if self.bdims != other.bdims:
            return False

        # Convert both to dense for comparison to handle sparse arrays
        self_dense = self.data.todense() if hasattr(self.data, 'todense') else self.data
        other_dense = other.data.todense() if hasattr(other.data, 'todense') else other.data
        
        return jnp.all(self_dense == other_dense)

    def __ne__(self, other):
        return not self.__eq__(other)

    # Elementary Math ----
    def __matmul__(self, other):
        if not isinstance(other, Qarray):
            return NotImplemented
        
        _qdims_new = self._qdims @ other._qdims
        new_impl = self._impl.matmul(other._impl)
        
        return Qarray.create(
            new_impl.data, 
            dims=_qdims_new.dims,
            implementation=new_impl.impl_type,
        )

    def __mul__(self, other):
        if isinstance(other, Qarray):
            return self.__matmul__(other)

        other = other + 0.0j
        if not robust_isscalar(other) and len(other.shape) > 0:  # not a scalar
            other = other.reshape(other.shape + (1, 1))

        new_impl = self._impl.mul(other)
        return Qarray.create(
            new_impl.data, 
            dims=self._qdims.dims,
            implementation=new_impl.impl_type,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def __truediv__(self, other):
        """For Qarray's, this only really makes sense in the context of division by a scalar."""

        if isinstance(other, Qarray):
            raise ValueError("Cannot divide a Qarray by another Qarray.")

        return self.__mul__(1 / other)

    def __add__(self, other):
        if isinstance(other, Qarray):
            if self.dims != other.dims:
                msg = (
                    "Dimensions are incompatible: "
                    + repr(self.dims)
                    + " and "
                    + repr(other.dims)
                )
                raise ValueError(msg)
            new_impl = self._impl.add(other._impl)
            return Qarray.create(
                new_impl.data, 
                dims=self.dims, 
                implementation=new_impl.impl_type,
            )

        if robust_isscalar(other) and other == 0:
            return self.copy()

        if self.data.shape[-2] == self.data.shape[-1]:
            other = other + 0.0j
            if not robust_isscalar(other) and len(other.shape) > 0:  # not a scalar
                other = other.reshape(other.shape + (1, 1))
            other = Qarray.create(
                other * jnp.eye(self.data.shape[-2], dtype=self.data.dtype),
                dims=self.dims,
            )
            # TODO: move this math into implementaiton to support sparse!
            return self.__add__(other)

        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Qarray):
            if self.dims != other.dims:
                msg = (
                    "Dimensions are incompatible: "
                    + repr(self.dims)
                    + " and "
                    + repr(other.dims)
                )
                raise ValueError(msg)
            new_impl = self._impl.sub(other._impl)
            return Qarray.create(
                new_impl.data, 
                dims=self.dims, 
                implementation=new_impl.impl_type,
            )

        if robust_isscalar(other) and other == 0:
            return self.copy()

        if self.data.shape[-2] == self.data.shape[-1]:
            other = other + 0.0j
            if not robust_isscalar(other) and len(other.shape) > 0:  # not a scalar
                other = other.reshape(other.shape + (1, 1))
            other = Qarray.create(
                other * jnp.eye(self.data.shape[-2], dtype=self.data.dtype),
                dims=self.dims,
            )
            # TODO: move this math into implementaiton to support sparse!
            return self.__sub__(other)

        return NotImplemented

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __xor__(self, other):
        if not isinstance(other, Qarray):
            return NotImplemented
        return tensor(self, other)

    def __rxor__(self, other):
        if not isinstance(other, Qarray):
            return NotImplemented
        return tensor(other, self)

    def __pow__(self, other):
        if not isinstance(other, int):
            return NotImplemented

        return powm(self, other)

    # String Representation ----
    def _str_header(self):
        impl_type = self.impl_type.value
        out = ", ".join(
            [
                "Quantum array: dims = " + str(self.dims),
                "bdims = " + str(self.bdims),
                "shape = " + str(self.data.shape),
                "type = " + str(self.qtype),
                "impl = " + impl_type,
            ]
        )
        return out

    def __str__(self):
        return self._str_header() + "\nQarray data =\n" + str(self.data)

    @property
    def header(self):
        """Print the header of the Qarray."""
        return self._str_header()

    def __repr__(self):
        return self.__str__()

    # Utilities ----
    def copy(self, memo=None):
        return self.__deepcopy__(memo)

    def __deepcopy__(self, memo):
        """Need to override this when defininig __getattr__."""

        return Qarray(
            _impl=deepcopy(self._impl, memo=memo),
            _qdims=deepcopy(self._qdims, memo=memo),
            _bdims=deepcopy(self._bdims, memo=memo),
        )

    def __getattr__(self, method_name):
        if "__" == method_name[:2]:
            # NOTE: we return NotImplemented for binary special methods logic in python, plus things like __jax_array__
            return lambda *args, **kwargs: NotImplemented

        modules = [jnp, jnp.linalg, jsp, jsp.linalg]

        method_f = None
        for mod in modules:
            method_f = getattr(mod, method_name, None)
            if method_f is not None:
                break

        if method_f is None:
            raise NotImplementedError(
                f"Method {method_name} does not exist. No backup method found in {modules}."
            )

        def func(*args, **kwargs):
            # For operations that might not be supported in sparse, convert to dense
            if self.is_sparse:
                dense_self = self.to_dense()
                res = method_f(dense_self.data, *args, **kwargs)
            else:
                res = method_f(self.data, *args, **kwargs)

            if getattr(res, "shape", None) is None or res.shape != self.data.shape:
                return res
            else:
                # Preserve implementation type
                return Qarray.create(res, dims=self._qdims.dims, implementation=self.impl_type)

        return func

    # Conversions / Reshaping ----
    def dag(self):
        return dag(self)

    def to_dm(self):
        return ket2dm(self)

    def is_dm(self):
        return self.qtype == Qtypes.oper

    def is_vec(self):
        return self.qtype == Qtypes.ket or self.qtype == Qtypes.bra

    def to_ket(self):
        return to_ket(self)

    def transpose(self, *args):
        return transpose(self, *args)

    def keep_only_diag_elements(self):
        return keep_only_diag_elements(self)

    # Math Functions ----
    def unit(self):
        return unit(self)

    def norm(self):
        return norm(self)
    
    def frobenius_norm(self):
        """Compute Frobenius norm directly from implementation."""
        return self._impl.frobenius_norm()
    
    def real(self):
        """Element-wise real part."""
        new_impl = self._impl.real()
        return Qarray.create(
            new_impl.data, 
            dims=self.dims,
            implementation=new_impl.impl_type,
        )
    
    def imag(self):
        """Element-wise imaginary part."""
        new_impl = self._impl.imag()
        
        return Qarray.create(
            new_impl.data, 
            dims=self.dims,
            implementation=new_impl.impl_type,
        )
    
    def conj(self):
        """Element-wise complex conjugate."""
        new_impl = self._impl.conj()
        return Qarray.create(
            new_impl.data, 
            dims=self.dims, 
            implementation=new_impl.impl_type,
        )

    def expm(self):
        return expm(self)

    def powm(self, n):
        return powm(self, n)

    def cosm(self):
        return cosm(self)

    def sinm(self):
        return sinm(self)

    def tr(self, **kwargs):
        return tr(self, **kwargs)

    def trace(self, **kwargs):
        return tr(self, **kwargs)

    def ptrace(self, indx):
        return ptrace(self, indx)

    def eigenstates(self):
        return eigenstates(self)

    def eigenenergies(self):
        return eigenenergies(self)

    def eigenvalues(self):
        return eigenenergies(self)

    def collapse(self, mode="sum"):
        return collapse(self, mode=mode)


# Qarray operations ---------------------------------------------------------------------

def concatenate(qarr_list: List[Qarray], axis: int = 0) -> Qarray:
    """Concatenate a list of Qarrays along a specified axis.

    Args:
        qarr_list (List[Qarray]): List of Qarrays to concatenate.
        axis (int): Axis along which to concatenate. Default is 0.

    Returns:
        Qarray: Concatenated Qarray.
    """

    non_empty_qarr_list = [qarr for qarr in qarr_list if len(qarr.data) != 0]

    if len(non_empty_qarr_list) == 0:
        return Qarray.from_list([])

    concatenated_data = jnp.concatenate(
        [qarr.data for qarr in non_empty_qarr_list], axis=axis
    )

    dims = non_empty_qarr_list[0].dims
    return Qarray.create(concatenated_data, dims=dims)

def collapse(qarr: Qarray, mode="sum") -> Qarray:
    """Collapse the Qarray.

    Args:
        qarr (Qarray): quantum array array

    Returns:
        Collapsed quantum array
    """

    if mode == "sum":
        if len(qarr.bdims) == 0:
            return qarr

        batch_axes = list(range(len(qarr.bdims)))
        
        # Preserve implementation type
        implementation = qarr.impl_type
        return Qarray.create(jnp.sum(qarr.data, axis=batch_axes), dims=qarr.dims, implementation=implementation)

def transpose(qarr: Qarray, indices: List[int]) -> Qarray:
    """Transpose the quantum array."""

    qarr = qarr.to_dense()
    
    indices = list(indices)

    shaped_data = qarr.shaped_data
    dims = qarr.dims
    bdims_indxs = list(range(len(qarr.bdims)))

    reshape_indices = indices + [j + len(dims[0]) for j in indices]
    reshape_indices = bdims_indxs + [j + len(bdims_indxs) for j in reshape_indices]

    shaped_data = shaped_data.transpose(reshape_indices)
    new_dims = (
        tuple([dims[0][j] for j in indices]),
        tuple([dims[1][j] for j in indices]),
    )

    full_dims = prod(dims[0])
    full_data = shaped_data.reshape(*qarr.bdims, full_dims, -1)
    
    # Preserve implementation type
    implementation = qarr.impl_type
    return Qarray.create(full_data, dims=new_dims, implementation=implementation)

def unit(qarr: Qarray) -> Qarray:
    """Normalize the quantum array.

    Args:
        qarr (Qarray): quantum array

    Returns:
        Normalized quantum array
    """
    return qarr / qarr.norm()


def norm(qarr: Qarray) -> float:
    qarr = qarr.to_dense() # TODO: support sparse norm!

    qdata = qarr.data
    bdims = qarr.bdims

    if qarr.qtype == Qtypes.oper:
        qdata_dag = qarr.dag().data

        if len(bdims) > 0:
            qdata = qdata.reshape(-1, qdata.shape[-2], qdata.shape[-1])
            qdata_dag = qdata_dag.reshape(-1, qdata_dag.shape[-2], qdata_dag.shape[-1])

            evals, _ = vmap(jnp.linalg.eigh)(qdata @ qdata_dag)
            rho_norm = jnp.sum(jnp.sqrt(jnp.abs(evals)), axis=-1)
            rho_norm = rho_norm.reshape(*bdims)
            return rho_norm
        else:
            evals, _ = jnp.linalg.eigh(qdata @ qdata_dag)
            rho_norm = jnp.sum(jnp.sqrt(jnp.abs(evals)))
            return rho_norm
        
    elif qarr.qtype in [Qtypes.ket, Qtypes.bra]:
        if len(bdims) > 0:
            qdata = qdata.reshape(-1, qdata.shape[-2], qdata.shape[-1])
            return vmap(jnp.linalg.norm)(qdata).reshape(*bdims)
        else:
            return jnp.linalg.norm(qdata)


def tensor(*args, **kwargs) -> Qarray:
    """Tensor product."""
    parallel = kwargs.pop("parallel", False)

    # For tensor operations, we'll need to handle mixed implementations
    # For now, convert all to dense for tensor operations
    dense_args = [arg.to_dense() if arg.impl_type != QarrayImplType.DENSE else arg for arg in args]
    
    data = dense_args[0].data
    dims = deepcopy(dense_args[0].dims)
    dims_0 = dims[0]
    dims_1 = dims[1]
    
    for arg in dense_args[1:]:
        if parallel:
            a = data
            b = arg.data

            if len(a.shape) > len(b.shape):
                batch_dim = a.shape[:-2]
            elif len(a.shape) == len(b.shape):
                if prod(a.shape[:-2]) > prod(b.shape[:-2]):
                    batch_dim = a.shape[:-2]
                else:
                    batch_dim = b.shape[:-2]
            else:
                batch_dim = b.shape[:-2]

            data = jnp.einsum("...ij,...kl->...ikjl", a, b).reshape(
                *batch_dim, a.shape[-2] * b.shape[-2], -1
            )
        else:
            data = jnp.kron(data, arg.data, **kwargs)

        dims_0 = dims_0 + arg.dims[0]
        dims_1 = dims_1 + arg.dims[1]

    return Qarray.create(data, dims=(dims_0, dims_1))

def tr(qarr: Qarray, **kwargs) -> Array:
    """Full trace."""
    axis1 = kwargs.get("axis1", -2)
    axis2 = kwargs.get("axis2", -1)
    return jnp.trace(qarr.data, axis1=axis1, axis2=axis2, **kwargs)

def trace(qarr: Qarray, **kwargs) -> Array:
    """Full trace."""
    return tr(qarr, **kwargs)

def expm_data(data: Array, **kwargs) -> Array:
    """Matrix exponential wrapper."""
    return jsp.linalg.expm(data, **kwargs)

def expm(qarr: Qarray, **kwargs) -> Qarray:
    """Matrix exponential wrapper."""
    dims = qarr.dims
    # Convert to dense for expm
    dense_data = qarr.to_dense().data
    data = expm_data(dense_data, **kwargs)
    return Qarray.create(data, dims=dims)

def powm(qarr: Qarray, n: Union[int, float], clip_eigvals=False) -> Qarray:
    """Matrix power."""
    # Convert to dense for powm
    dense_qarr = qarr.to_dense()
    
    if isinstance(n, int):
        data_res = jnp.linalg.matrix_power(dense_qarr.data, n)
    else:
        evalues, evectors = jnp.linalg.eig(dense_qarr.data)
        if clip_eigvals:
            evalues = jnp.maximum(evalues, 0)
        else:
            if not (evalues >= 0).all():
                raise ValueError(
                    "Non-integer power of a matrix can only be "
                    "computed if the matrix is positive semi-definite."
                    "Got a matrix with a negative eigenvalue."
                )
        data_res = evectors * jnp.pow(evalues, n) @ jnp.linalg.inv(evectors)
    
    return Qarray.create(data_res, dims=qarr.dims)

def cosm_data(data: Array, **kwargs) -> Array:
    """Matrix cosine wrapper."""
    return (expm_data(1j * data) + expm_data(-1j * data)) / 2

def cosm(qarr: Qarray) -> Qarray:
    """Matrix cosine wrapper."""
    dims = qarr.dims
    # Convert to dense for cosm
    dense_data = qarr.to_dense().data
    data = cosm_data(dense_data)
    return Qarray.create(data, dims=dims)

def sinm_data(data: Array, **kwargs) -> Array:
    """Matrix sine wrapper."""
    return (expm_data(1j * data) - expm_data(-1j * data)) / (2j)

def sinm(qarr: Qarray) -> Qarray:
    """Matrix sine wrapper."""
    dims = qarr.dims
    # Convert to dense for sinm
    dense_data = qarr.to_dense().data
    data = sinm_data(dense_data)
    return Qarray.create(data, dims=dims)

def keep_only_diag_elements(qarr: Qarray) -> Qarray:
    """Keep only diagonal elements."""
    if len(qarr.bdims) > 0:
        raise ValueError("Cannot keep only diagonal elements of a batched Qarray.")

    dims = qarr.dims
    data = jnp.diag(jnp.diag(qarr.data))
    # Preserve implementation type
    implementation = qarr.impl_type
    return Qarray.create(data, dims=dims, implementation=implementation)

def to_ket(qarr: Qarray) -> Qarray:
    """Convert to ket."""
    if qarr.qtype == Qtypes.ket:
        return qarr
    elif qarr.qtype == Qtypes.bra:
        return qarr.dag()
    else:
        raise ValueError("Can only get ket from a ket or bra.")

def eigenstates(qarr: Qarray) -> Qarray:
    """Eigenstates of a quantum array."""
    # Convert to dense for eigenstates
    dense_qarr = qarr.to_dense()
    
    evals, evecs = jnp.linalg.eigh(dense_qarr.data)
    idxs_sorted = jnp.argsort(evals, axis=-1)

    dims = ket_from_op_dims(qarr.dims)

    evals = jnp.take_along_axis(evals, idxs_sorted, axis=-1)
    evecs = jnp.take_along_axis(evecs, idxs_sorted[..., None, :], axis=-1)

    # numpy returns [batch, :, i] as the i-th eigenvector
    # we want [batch, i, :] as the i-th eigenvector
    evecs = jnp.swapaxes(evecs, -2, -1)

    evecs = Qarray.create(
        evecs,
        dims=dims,
        bdims=evecs.shape[:-1],
    )

    return evals, evecs

def eigenenergies(qarr: Qarray) -> Array:
    """Eigenvalues of a quantum array."""
    # Convert to dense for eigenenergies
    dense_qarr = qarr.to_dense()
    evals = jnp.linalg.eigvalsh(dense_qarr.data)
    return evals

def ptrace(qarr: Qarray, indx) -> Qarray:
    """Partial Trace."""
    # Convert to dense for ptrace
    dense_qarr = qarr.to_dense()
    dense_qarr = ket2dm(dense_qarr)
    rho = dense_qarr.shaped_data
    dims = dense_qarr.dims

    Nq = len(dims[0])

    indxs = [indx, indx + Nq]
    for j in range(Nq):
        if j == indx:
            continue
        indxs.append(j)
        indxs.append(j + Nq)

    bdims = dense_qarr.bdims
    len_bdims = len(bdims)
    bdims_indxs = list(range(len_bdims))
    indxs = bdims_indxs + [j + len_bdims for j in indxs]
    rho = rho.transpose(indxs)

    for j in range(Nq - 1):
        rho = jnp.trace(rho, axis1=2 + len_bdims, axis2=3 + len_bdims)

    return Qarray.create(rho)

def dag(qarr: Qarray) -> Qarray:
    """Conjugate transpose."""
    dims = qarr.dims[::-1]
    new_impl = qarr._impl.dag()
    return Qarray.create(
        new_impl.data, 
        dims=dims, 
        implementation=new_impl.impl_type,
    )

def dag_data(arr: Array) -> Array:
    """Conjugate transpose.

    Args:
        arr: operator

    Returns:
        conjugate of op, and transposes last two axes
    """
    # TODO: revisit this case...
    if len(arr.shape) == 1:
        return jnp.conj(arr)

    return jnp.moveaxis(
        jnp.conj(arr), -1, -2
    )  # transposes last two axes, good for batching

def ket2dm(qarr: Qarray) -> Qarray:
    """Turns ket into density matrix."""
    if qarr.qtype == Qtypes.oper:
        return qarr

    if qarr.qtype == Qtypes.bra:
        qarr = qarr.dag()

    return qarr @ qarr.dag()

# Data level operations
def is_dm_data(data: Array) -> bool:
    """Check if data is a density matrix."""
    return data.shape[-2] == data.shape[-1]

def powm_data(data: Array, n: int) -> Array:
    """Matrix power."""
    return jnp.linalg.matrix_power(data, n)


# Type aliases for readability
DenseQarray = Qarray[DenseImpl]
SparseQarray = Qarray[SparseImpl]

ARRAY_TYPES = (Array, ndarray, Qarray)
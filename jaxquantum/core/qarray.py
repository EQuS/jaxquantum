""" QArray. """

import functools
from flax import struct
from enum import Enum
from jax import Array, config
from typing import List
from math import prod
from copy import deepcopy
from numbers import Number
import jax.numpy as jnp
import jax.scipy as jsp

config.update("jax_enable_x64", True)

DIMS_TYPE = List[List[int]]

def isket_dims(dims: DIMS_TYPE) -> bool:
    return prod(dims[1]) == 1

def isbra_dims(dims: DIMS_TYPE) -> bool:
    return prod(dims[0]) == 1

def isop_dims(dims: DIMS_TYPE) -> bool:
    return prod(dims[1]) == prod(dims[0])

class Qtypes(str, Enum):
    ket = "ket"
    bra = "bra"
    oper = "oper"

    @classmethod
    def from_dims(cls, dims: Array):
        if isket_dims(dims):
            return cls.ket
        if isbra_dims(dims):
            return cls.bra
        if isop_dims(dims):
            return cls.oper
        raise ValueError("Invalid data shape")

    @classmethod
    def from_str(cls, string: str):
        return cls(string)

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __hash__(self):
        return hash(self.value)


def check_dims(dims: Array, data_shape: Array) -> bool:
    assert data_shape[0] == prod(dims[0]), "Data shape should be consistent with dimensions."
    assert data_shape[1] == prod(dims[1]), "Data shape should be consistent with dimensions."

class Qdims:
    def __init__(self, dims):
        self._dims = dims
        self._qtype = Qtypes.from_dims(self._dims)

    @property
    def dims(self):
        return self._dims

    @property
    def from_(self):
        return self._dims[1]
    
    @property
    def to_(self):
        return self._dims[0]
    
    @property
    def qtype(self):
        return self._qtype

    def __str__(self):
        return str(self.dims)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.dims == other.dims

    def __ne__(self, other):
        return self.dims != other.dims

    def __hash__(self):
        return hash(self.dims)
    
    def __matmul__(self, other):
        if self.from_ != other.to_:
            raise TypeError(f"incompatible dimensions {self} and {other}")

        new_dims = [self.to_, other.from_]
        return Qdims(new_dims)


def _ensure_equal_type(method):
    """
    Function decorator for Qarray method to ensure both operands are Qarray and
    of the same type and dimensions. Promotes numeric scalar a to a*I, where I 
    is the identity matrix of the same type and dims.
    """
    @functools.wraps(method)
    def out(self, other):
        if isinstance(other, Qarray):
            if self.dims != other.dims:
                msg = (
                    "Dimensions are incompatible: "
                    + repr(self.dims) + " and " + repr(other.dims)
                )
                raise ValueError(msg)
            return method(self, other)
        if other == 0:
            return method(self, other)
        if (self.data.shape[0] == self.data.shape[1]) and isinstance(other, Number):
            scalar = complex(other)
            other = Qarray.create(jnp.eye(self.data.shape[0], dtype=self.data.dtype) * scalar, dims=self.dims)
            return method(self, other)
        return NotImplemented
    return out

@struct.dataclass # this allows us to send in and return Qarray from jitted functions
class Qarray:
    _data: Array
    _qdims: Qdims = struct.field(pytree_node=False)

    @classmethod
    def create(cls, data, dims=None):
        # Prepare data ----
        data = jnp.asarray(data)
        if len(data.shape) == 1:
            data = data.reshape(data.shape[0], 1)

        # Prepare dimensions ----
        if dims is None:
            dims = [[data.shape[0]], [data.shape[1]]]
        
        check_dims(dims, data.shape)

        qdims = Qdims(dims)

        dims = deepcopy(dims)
        return cls(data, qdims)


    def _covert_to_qarray(self, other):
        if not isinstance(other, Qarray):
            try:
                other = Qarray.create(other)
            except TypeError:
                return NotImplemented
        return other
    
    def __matmul__(self, other):
        other = self._covert_to_qarray(other)
        _qdims_new = self._qdims @ other._qdims
        return Qarray.create(
            self.data @ other.data,
            dims=_qdims_new.dims,
        )
    
    def __mul__(self, other):
        if isinstance(other, Qarray):
            return self.__matmul__(other)
        
        multiplier = complex(other)
        return Qarray.create(
            self.data * multiplier,
            dims=self._qdims.dims,
        )
    
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __truediv__(self, other):
        return self.__mul__(1 / other)
    
    def __neg__(self):
        return self.__mul__(-1)
    
    @_ensure_equal_type
    def __add__(self, other):
        if other == 0:
            return self.copy()
        return Qarray.create(self.data + other.data, dims=self.dims)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    @_ensure_equal_type
    def __sub__(self, other):
        if other == 0:
            return self.copy()
        return Qarray.create(self.data - other.data, dims=self.dims)
        
    def __rsub__(self, other):
        return self.__neg__().__add__(other)
    
    @property
    def qtype(self):
        return self._qdims.qtype
    
    @property
    def dtype(self):
        return self._data.dtype

    @property
    def dims(self):
        return self._qdims.dims
    
    @property
    def data(self):
        return self._data
    

    def _str_header(self):
        out = ", ".join([
            "Quantum array: dims = " + str(self.dims),
            "shape = " + str(self._data.shape),
            "type = " + str(self.qtype),
        ])
        return out

    def __str__(self):
        return self._str_header() + "\nQarray data =\n" + str(self.data)
    
    def __repr__(self):
        return self.__str__()
    
    def __xor__(self, other):
        other = self._covert_to_qarray(other)
        return tensor(self, other)
    
    def dag(self):
        return dag(self)
    
    def to_dm(self):
        return ket2dm(self)
    
    def copy(self):
        return Qarray.create(deepcopy(self.data), dims=self.dims)
    
    def unit(self):
        data = self.data

        if self.qtype == Qtypes.oper:
            evals, _ = jnp.linalg.eigh(data @ jnp.conj(data).T)
            rho_norm = jnp.sum(jnp.sqrt(jnp.abs(evals)))
            data = data / rho_norm
        elif self.qtype in [Qtypes.ket, Qtypes.bra]:
            data = data / jnp.linalg.norm(data)
        
        return Qarray.create(data, dims=self.dims)

    def expm(self):
        return expm(self)
    
    def tr(self, **kwargs):
        return tr(self, **kwargs)

    def ptrace(self, indx, dims):
        return ptrace(self, indx, dims)



# Qarray operations ---------------------------------------------------------------------
    
def tensor(*args, **kwargs) -> jnp.ndarray:
    """Tensor product.

    Args:
        *args (Qarray): tensors to take the product of

    Returns:
        Tensor product of given tensors

    """
    data = args[0].data
    dims = deepcopy(args[0].dims)
    for arg in args[1:]:
        data = jnp.kron(data, arg.data)
        dims[0] += arg.dims[0]
        dims[1] += arg.dims[1]
    return Qarray.create(data, dims=dims)

def tr(qarr: Qarray, **kwargs) -> jnp.ndarray:
    """Full trace.

    Args:
        qarr (Qarray): quantum array    

    Returns:
        Full trace.
    """
    return jnp.trace(qarr.data, **kwargs)

def expm(qarr: Qarray, **kwargs) -> jnp.ndarray:
    """Matrix exponential wrapper.

    Returns:
        matrix exponential
    """
    data = jsp.linalg.expm(qarr.data, **kwargs)
    dims = deepcopy(qarr.dims)
    return Qarray.create(data, dims=dims)

def ptrace(qarr: Qarray, indx, dims):
    """Partial Trace.

    Args:
        rho: density matrix
        indx: index to trace out
        dims: list of dimensions of the tensored hilbert spaces

    Returns:
        partial traced out density matrix

    TODO: Fix weird tracing errors that arise with reshape
    TODO: return Qarray
    """

    qarr = ket2dm(qarr)
    rho = qarr.data

    Nq = len(dims)

    if isinstance(dims, jnp.ndarray):
        dims2 = jnp.concatenate(jnp.array([dims, dims]))
    else:
        dims2 = dims + dims

    rho = rho.reshape(dims2)

    indxs = [indx, indx + Nq]
    for j in range(Nq):
        if j == indx:
            continue
        indxs.append(j)
        indxs.append(j + Nq)
    rho = rho.transpose(indxs)

    for j in range(Nq - 1):
        rho = jnp.trace(rho, axis1=2, axis2=3)

    return rho

# Kets & Density Matrices -----------------------------------------------------

def dag(qarr: Qarray) -> Qarray:
    """Conjugate transpose.

    Args:
        qarr (Qarray): quantum array

    Returns:
        conjugate transpose of qarr
    """
    data = jnp.conj(qarr.data).T
    dims = deepcopy(qarr.dims)
    dims = dims[::-1]
    return Qarray.create(data, dims=dims)

def ket2dm(qarr: Qarray) -> Qarray:
    """Turns ket into density matrix.
    Does nothing if already operator.

    Args:
        qarr (Qarray): qarr

    Returns:
        Density matrix
    """

    if qarr.qtype == Qtypes.oper:
        return qarr
    
    if qarr.qtype == Qtypes.bra:
        qarr = qarr.dag()

    return qarr @ qarr.dag()
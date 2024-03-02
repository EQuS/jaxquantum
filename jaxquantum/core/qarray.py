""" QArray. """

from flax import struct
from enum import Enum
from jax import Array
from typing import List
from math import prod
import jax.numpy as jnp


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


@struct.dataclass # this allows us to send in and return Qarray from jitted functions
class Qarray:
    _data: Array
    _qdims: Qdims = struct.field(pytree_node=False)

    @classmethod
    def create(cls, N, params, label=0, use_linear=True, N_pre_diag=None):
        if N_pre_diag is None:
            N_pre_diag = N
        return cls(N, N_pre_diag, params, label, use_linear)

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

        return cls(data, qdims)


    def __matmul__(self, other):
        if not isinstance(other, Qarray):
            try:
                other = Qarray.create(other)
            except TypeError:
                return NotImplemented
            
        _qdims_new = self._qdims @ other._qdims
        return Qarray.create(
            self.data @ other.data,
            dims=_qdims_new.dims,
        )
    
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
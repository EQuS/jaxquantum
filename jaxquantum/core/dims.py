"""dims."""

from typing import List, Tuple
from copy import deepcopy
from math import prod
from jax import Array

from enum import Enum


DIMS_TYPE = List[List[int]]


def isket_dims(dims: DIMS_TYPE) -> bool:
    return prod(dims[1]) == 1


def isbra_dims(dims: DIMS_TYPE) -> bool:
    return prod(dims[0]) == 1


def isop_dims(dims: DIMS_TYPE) -> bool:
    return prod(dims[1]) == prod(dims[0])


def ket_from_op_dims(dims: DIMS_TYPE) -> DIMS_TYPE:
    return (dims[0], tuple([1 for _ in dims[1]]))


def check_dims(dims: Tuple[Tuple[int]], bdims: Tuple[int], data_shape: Array) -> bool:
    if len(data_shape) == 1 and data_shape[0] == 0:
        # E.g. empty list of operators
        assert bdims == (0,)
        assert dims == ((), ())
        return

    assert bdims == data_shape[:-2], "Data shape should be consistent with dimensions."
    assert data_shape[-2] == prod(dims[0]), (
        "Data shape should be consistent with dimensions."
    )
    assert data_shape[-1] == prod(dims[1]), (
        "Data shape should be consistent with dimensions."
    )


class Qdims:
    def __init__(self, dims):
        self._dims = deepcopy(dims)
        self._dims = (tuple(self._dims[0]), tuple(self._dims[1]))
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
        return (self.dims == other.dims) and (self.qtype == other.qtype)

    def __ne__(self, other):
        return (self.dims != other.dims) or (self.qtype != other.qtype)

    def __hash__(self):
        return hash(self.dims)

    def __matmul__(self, other):
        if self.from_ != other.to_:
            raise TypeError(f"incompatible dimensions {self} and {other}")

        new_dims = [self.to_, other.from_]
        return Qdims(new_dims)


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

""" QArray. """

import functools
from flax import struct
from enum import Enum
from jax import Array, config, vmap
from typing import List, Optional
from math import prod
from copy import deepcopy
from numbers import Number
import jax.numpy as jnp
import jax.scipy as jsp

from jaxquantum.core.settings import SETTINGS

config.update("jax_enable_x64", True)

DIMS_TYPE = List[List[int]]

def isket_dims(dims: DIMS_TYPE) -> bool:
    return prod(dims[1]) == 1

def isbra_dims(dims: DIMS_TYPE) -> bool:
    return prod(dims[0]) == 1

def isop_dims(dims: DIMS_TYPE) -> bool:
    return prod(dims[1]) == prod(dims[0])

def ket_from_op_dims(dims: DIMS_TYPE) -> DIMS_TYPE:
    return [dims[0], [1 for _ in dims[1]]]

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
        self._dims = deepcopy(dims)
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
        if (self.data.shape[0] == self.data.shape[1]):
            scalar = other + 0.0j
            other = Qarray.create(jnp.eye(self.data.shape[0], dtype=self.data.dtype) * scalar, dims=self.dims)
            return method(self, other)
        return NotImplemented
    return out

def tidy_up(data, atol):
    data_re = jnp.real(data)
    data_im = jnp.imag(data)
    data_re_mask = jnp.abs(data_re) > atol
    data_im_mask = jnp.abs(data_im) > atol
    data_new = data_re * data_re_mask + 1j * data_im * data_im_mask
    return data_new


@struct.dataclass # this allows us to send in and return Qarray from jitted functions
class Qarray:
    _data: Array
    _qdims: Qdims = struct.field(pytree_node=False)

    # Initialization ----
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

        # TODO: Constantly tidying up on Qarray creation might be a bit overkill.
        # It increases the compilation time, but only very slightly 
        # increased the runtime of the jit compiled function.
        # We could instead use this tidy_up where we think we need it.
        data = tidy_up(data, SETTINGS["auto_tidyup_atol"])

        return cls(data, qdims)

    def _covert_to_qarray(self, other):
        if not isinstance(other, Qarray):
            try:
                other = Qarray.create(other)
            except TypeError:
                return NotImplemented
        return other
    
    # ----

    # Properties ----
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
    def qdims(self):
        return self._qdims
        
    @property
    def space_dims(self):
        if self.qtype in [Qtypes.oper, Qtypes.ket]:
            return self.dims[0]
        elif self.qtype == Qtypes.bra:
            return self.dims[1]
        else:
            raise ValueError("Unsupported qtype.")
        
    @property
    def data(self):
        return self._data
    
    @property
    def shaped_data(self):
        return self._data.reshape(self.dims[0] + self.dims[1])

    # ----


    # Math Magic ----
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
        
        multiplier = other + 0.0j
        return Qarray.create(
            self.data * multiplier,
            dims=self._qdims.dims,
        )

    def __rmul__(self, other):
        return self.__mul__(other) 

    def __neg__(self):
        return self.__mul__(-1)
    
    def __truediv__(self, other):
        return self.__mul__(1 / other)
    
    @_ensure_equal_type
    def __add__(self, other):
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
    
    def __xor__(self, other):
        other = self._covert_to_qarray(other)
        return tensor(self, other)
    
    def __pow__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        
        return powm(self, other)
    
    # ----

    # String Representation ----
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
    
    # ----

    # Utilities ----
    def copy(self, memo=None):
        # return Qarray.create(deepcopy(self.data), dims=self.dims)
        return self.__deepcopy__(memo)
    
    def __deepcopy__(self, memo):
        """ Need to override this when defininig __getattr__. """

        return Qarray(
            _data = deepcopy(self._data, memo=memo),
            _qdims = deepcopy(self._qdims, memo=memo)
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
            raise NotImplementedError(f"Method {method_name} does not exist. No backup method found in {modules}.")

        def func(*args, **kwargs): 
            res = method_f(self.data, *args, **kwargs)
            
            if getattr(res, "shape", None) is None or res.shape != self.data.shape:
                return res
            else:
                return Qarray(
                    _data=res,
                    _qdims=self._qdims
                )
        return func

    # ----

    # Conversions / Reshaping ----
    def dag(self):
        return dag(self)
    
    def to_dm(self):
        return ket2dm(self)
    
    def is_dm(self):
        return self.qtype == Qtypes.oper
    
    def to_ket(self):
        return to_ket(self)
    
    def transpose(self, *args):
        return transpose(self, *args)

    def keep_only_diag_elements(self):
        return keep_only_diag_elements(self)
    
    # ----

    # Math Functions ----
    def unit(self):
        return unit(self)

    def expm(self):
        return expm(self)
    
    def cosm(self):
        return cosm(self)
    
    def sinm(self):
        return sinm(self)
    
    def tr(self, **kwargs):
        return tr(self, **kwargs)

    def ptrace(self, indx):
        return ptrace(self, indx)
        
    def eigenstates(self):
        return eigenstates(self)

    def eigenenergies(self):
        return eigenenergies(self)
    
    # ----

@struct.dataclass # this allows us to send in and return Qarray from jitted functions
class QarrayArray:
    """ This class provides a way to construct arrays of Qarrays for vectorized operations on Qarrays.
    """ 
    _data: Array
    _qdims: Optional[Qdims] = struct.field(pytree_node=False)

    # Initialization ----
    @classmethod
    def create(cls, qarr_list: List[Qarray]):
        if len(qarr_list) == 0:
            return cls(_data=jnp.array([]), _qdims=None)

        data = jnp.array([qarr._data for qarr in qarr_list])

        qdims_list = [qarr._qdims for qarr in qarr_list]
        if len(qdims_list) > 0:
            assert qdims_list[:-1] == qdims_list[1:], "All qdims of Qarrays in the list must be the same." # equal dims
        
        return cls(_data=data, _qdims=deepcopy(qdims_list[0]))
    
    # ----

    # List Methods ----
    def append(self, qarr: Qarray):
        if len(self._data) == 0:
            data = jnp.array([qarr._data])
            qdims = qarr._qdims
        else:
            assert qarr._qdims == self._qdims, "qdim of Qarray must match that of the current members of this QarrayArray" # equal dims
            data = jnp.concatenate([self._data, jnp.array([qarr._data])])
            qdims = qarr._qdims 
        return QarrayArray(_data=data, _qdims=qdims)

    def __getitem__(self, index):
        return Qarray.create(self.data[index], dims=self.dims)

    def __len__(self):
        return self._data.shape[0]
    # ----

    # Properties ----
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
    def qdims(self):
        return self._qdims

    @property
    def space_dims(self):
        if self.qtype in [Qtypes.oper, Qtypes.ket]:
            return self.dims[0]
        elif self.qtype == Qtypes.bra:
            return self.dims[1]
        else:
            raise ValueError("Unsupported qtype.")

    @property
    def data(self):
        return self._data

    @property
    def shaped_data(self):
        return self._data.reshape([-1] + self.dims[0] + self.dims[1])
    
    # ----

    # Math Magic ----
    def __add__(self, other):
        if not isinstance(other, QarrayArray):
            return ValueError("Both objects must be of type QarrayArray.")
        
        if len(self._data) == 0:
            return other 
        elif len(other._data) == 0:
            return self 
        else: 
            assert self._qdims == other._qdims, "qdims of each QarrayArray must match."
            data = jnp.concatenate([self._data, other._data])
            return QarrayArray(
                _data=data,
                _qdims=self._qdims
            )
            
    def __radd__(self, other):
        return self.__add__(other)

    # ----
    
    # String Representation ----
    def _str_header(self):
        out = ", ".join([
            "Array of quantum arrays with: dims = " + str(self.dims),
            "shape = " + str(self._data.shape),
            "type = " + str(self.qtype),
        ])
        return out

    def __str__(self):
        if len(self._data) == 0:
            return "QarrayArray is empty."

        return self._str_header() + "\nQarrayArray data =\n" + str(self.data)
    
    def __repr__(self):
        return self.__str__()
    
    # ----

    # Utilities ----
    def copy(self, memo=None):
        return self.__deepcopy__(memo)

    def __deepcopy__(self, memo):
        """ Need to override this when defininig __getattr__. """

        return QarrayArray(
            _data = deepcopy(self._data, memo=memo),
            _qdims = deepcopy(self._qdims, memo=memo)
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
            raise NotImplementedError(f"Method {method_name} does not exist. No backup method found in {modules}.")

        def func(*args, **kwargs): 
            res = vmap(method_f)(self.data, *args, **kwargs)

            if getattr(res, "shape", None) is None or res.shape != self.data.shape:
                return res
            else:
                return QarrayArray(
                    _data=res,
                    _qdims=self._qdims
                )
        return func
    
    # ----

# Qarray operations ---------------------------------------------------------------------

def transpose(qarr: Qarray, indices: List[int]) -> Qarray:
    """ Transpose the quantum array.

    Args:
        qarr (Qarray): quantum array
        *args: axes to transpose
    
    Returns:
        tranposed Qarray
    """
    shaped_data = qarr.shaped_data
    dims = qarr.dims

    reshape_indices = indices + [j + len(dims[0]) for j in indices]
    shaped_data = shaped_data.transpose(reshape_indices)
    new_dims = [[dims[0][j] for j in indices] , [dims[1][j] for j in indices]]

    full_dims = prod(dims[0])
    full_data = shaped_data.reshape(full_dims, -1)
    
    return Qarray.create(full_data, dims = new_dims)

def unit(qarr: Qarray) -> Qarray:
    """Normalize the quantum array.

    Args:
        qarr (Qarray): quantum array

    Returns:
        Normalized quantum array
    """
    data = qarr.data

    if qarr.qtype == Qtypes.oper:
        evals, _ = jnp.linalg.eigh(data @ jnp.conj(data).T)
        rho_norm = jnp.sum(jnp.sqrt(jnp.abs(evals)))
        data = data / rho_norm
    elif qarr.qtype in [Qtypes.ket, Qtypes.bra]:
        data = data / jnp.linalg.norm(data)
    
    return Qarray.create(data, dims=qarr.dims)
    
def tensor(*args, **kwargs) -> Qarray:
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

def tr(qarr: Qarray, **kwargs) -> jnp.complex128:
    """Full trace.

    Args:
        qarr (Qarray): quantum array    

    Returns:
        Full trace.
    """
    return trace(qarr, **kwargs)


def expm_data(data: Array, **kwargs) -> Array:
    """Matrix exponential wrapper.

    Returns:
        matrix exponential
    """
    return jsp.linalg.expm(data, **kwargs)

def expm(qarr: Qarray, **kwargs) -> Qarray:
    """Matrix exponential wrapper.

    Returns:
        matrix exponential
    """
    dims = qarr.dims
    data = expm_data(qarr.data, **kwargs)
    return Qarray.create(data, dims=dims)

def powm(qarr: Qarray, n: int) -> Qarray:
    """Matrix power.

    Args:
        qarr (Qarray): quantum array
        n (int): power

    Returns:
        matrix power
    """
    data = jnp.linalg.matrix_power(qarr.data, n)
    return Qarray.create(data, dims=qarr.dims)

def cosm_data(data: Array, **kwargs) -> Array:
    """Matrix cosine wrapper.

    Returns:
        matrix cosine
    """
    return (expm_data(1j*data) + expm_data(-1j*data))/2 

def cosm(qarr: Qarray) -> Qarray:
    """Matrix cosine wrapper.

    Args:
        qarr (Qarray): quantum array

    Returns:
        matrix cosine
    """
    dims = qarr.dims
    data = cosm_data(qarr.data)
    return Qarray.create(data, dims=dims)

def sinm_data(data: Array, **kwargs) -> Array:
    """Matrix sine wrapper.

    Args:
        data: matrix

    Returns:
        matrix sine
    """
    return (expm_data(1j*data) - expm_data(-1j*data))/(2j)

def sinm(qarr: Qarray) -> Qarray:
    dims = qarr.dims
    data = sinm_data(qarr.data)
    return Qarray.create(data, dims=dims)

def keep_only_diag_elements(qarr: Qarray) -> Qarray:
    dims = qarr.dims
    data = jnp.diag(jnp.diag(qarr.data))
    return Qarray.create(data, dims=dims)

def to_ket(qarr: Qarray) -> Qarray:
    if qarr.qtype == Qtypes.ket:
        return qarr
    elif qarr.qtype == Qtypes.bra:
        return qarr.dag()
    else:
        raise ValueError("Can only get ket from a ket or bra.")
    
def eigenstates(qarr: Qarray) -> Qarray:
    """Eigenstates of a quantum array.

    Args:
        qarr (Qarray): quantum array

    Returns:
        eigenvalues and eigenstates
    """

    evals, evecs = jnp.linalg.eigh(qarr.data)
    idxs_sorted = jnp.argsort(evals)
    
    dims = ket_from_op_dims(qarr.dims)

    evals =  evals[idxs_sorted]
    evecs = evecs[:, idxs_sorted]
    evecs = [Qarray.create(arr, dims=dims) for arr in evecs]

    return evals, evecs

def eigenenergies(qarr: Qarray) -> Array:
    """Eigenvalues of a quantum array.

    Args:
        qarr (Qarray): quantum array

    Returns:
        eigenvalues
    """

    evals = jnp.linalg.eigvalsh(qarr.data)
    return evals

# More quantum specific -----------------------------------------------------

def ptrace(qarr: Qarray, indx) -> Qarray:
    """Partial Trace.

    Args:
        rho: density matrix
        indx: index of quantum object to keep, rest will be partial traced out

    Returns:
        partial traced out density matrix

    TODO: Fix weird tracing errors that arise with reshape
    """

    qarr = ket2dm(qarr)
    rho = qarr.shaped_data
    dims = qarr.dims

    Nq = len(dims[0])

    indxs = [indx, indx + Nq]
    for j in range(Nq):
        if j == indx:
            continue
        indxs.append(j)
        indxs.append(j + Nq)
    rho = rho.transpose(indxs)

    for j in range(Nq - 1):
        rho = jnp.trace(rho, axis1=2, axis2=3)

    return Qarray.create(rho)

def trace(qarr: Qarray, **kwargs) -> Qarray:
    return jnp.trace(qarr.data, **kwargs)

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


# Data level operations ----


def batch_dag_data(op: Array) -> Array:
    """Conjugate transpose.

    Args:
        op: operator

    Returns:
        conjugate of op, and transposes last two axes
    """
    return jnp.moveaxis(
        jnp.conj(op), -1, -2
    )  # transposes last two axes, good for batching

def dag_data(op: Array) -> Array:
    """Conjugate transpose.

    Args:
        op: operator

    Returns:
        conjugate of op, and transposes last two axes
    """
    return jnp.conj(op.T) 
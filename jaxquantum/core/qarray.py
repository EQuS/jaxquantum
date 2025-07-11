""" QArray. """

from __future__ import annotations

import functools
from flax import struct
from enum import Enum
from jax import Array, config, vmap
from typing import List, Optional, Union


from math import prod
from copy import deepcopy
from numbers import Number
from numpy import ndarray
import jax.numpy as jnp
import jax.scipy as jsp

from jaxquantum.core.settings import SETTINGS
from jaxquantum.utils.utils import robust_isscalar
from jaxquantum.core.dims import Qtypes, Qdims, check_dims, isket_dims, isbra_dims, isop_dims, ket_from_op_dims

config.update("jax_enable_x64", True)

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
    _bdims: Tuple[int] = struct.field(pytree_node=False)


    # Initialization ----
    @classmethod
    def create(cls, data, dims=None, bdims=None):

        # Step 1: Prepare data ----
        data = jnp.asarray(data)

        if len(data.shape) == 1 and data.shape[0] > 0:
            data = data.reshape(data.shape[0], 1)

        if len(data.shape) >= 2:
            if data.shape[-2] != data.shape[-1] and not (data.shape[-2] == 1 or data.shape[-1] == 1):
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

        dims = (tuple(dims[0]), tuple(dims[1]))
        
        check_dims(dims, bdims, data.shape)

        qdims = Qdims(dims)

        # NOTE: Constantly tidying up on Qarray creation might be a bit overkill.
        # It increases the compilation time, but only very slightly 
        # increased the runtime of the jit compiled function.
        # We could instead use this tidy_up where we think we need it.
        data = tidy_up(data, SETTINGS["auto_tidyup_atol"])

        return cls(data, qdims, bdims)

    # ----

    @classmethod
    def from_list(cls, qarr_list: List[Qarray]) -> Qarray:
        """ Create a Qarray from a list of Qarrays. """
        
        data = jnp.array([qarr.data for qarr in qarr_list])
        

        if len(qarr_list) == 0:
            dims = ((),())
            bdims = ()
        else:
            dims = qarr_list[0].dims
            bdims = qarr_list[0].bdims
        
        if not all(qarr.dims == dims and qarr.bdims == bdims for qarr in qarr_list):
            raise ValueError("All Qarrays in the list must have the same dimensions.")
            
        bdims = (len(qarr_list),) + bdims

        return cls.create(data, dims=dims, bdims=bdims)

    @classmethod
    def from_array(cls, qarr_arr) -> Qarray:
        """ Create a Qarray from a nested list of Qarrays. 
        
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
        
        depth = len(bdims)

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
        return self._data.dtype

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
        return self._data
    
    @property
    def shaped_data(self):
        return self._data.reshape(self.bdims + self.dims[0] + self.dims[1])

    @property 
    def shape(self):
        return self.data.shape

    @property
    def is_batched(self):
        return len(self.bdims) > 0

    def __getitem__(self, index):
        if len(self.bdims) > 0:
            return Qarray.create(
                self.data[index],
                dims=self.dims,
            )
        else:
            raise ValueError("Cannot index a non-batched Qarray.")

    def reshape_bdims(self, *args):
        """ Reshape the batch dimensions of the Qarray. """
        new_bdims = tuple(args)

        if prod(new_bdims) == 0:
            new_shape = new_bdims 
        else:
            new_shape = new_bdims + self.dims[0] + self.dims[1]
        return Qarray.create(
            self.data.reshape(new_shape),
            dims=self.dims,
            bdims=new_bdims,
        )

    def resize(self, new_shape):
        """ Resize the Qarray to a new shape. 
        
        This is useful for 
        """
        dims = self.dims
        data = jnp.resize(self.data, new_shape)
        return Qarray.create(
            data,
            dims=dims,
        )

    def __len__(self):
        """ Length of the Qarray. """
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

        return jnp.all(self.data == other.data)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    # ----


    # Elementary Math ----
    def __matmul__(self, other):
        if not isinstance(other, Qarray):
            return NotImplemented
        _qdims_new = self._qdims @ other._qdims
        return Qarray.create(
            self.data @ other.data,
            dims=_qdims_new.dims,
        )
    
    # NOTE: not possible to reach this.
    # def __rmatmul__(self, other):        
    #     if not isinstance(other, Qarray):
    #         return NotImplemented

    #     _qdims_new = other._qdims @ self._qdims
    #     return Qarray.create(
    #         other.data @ self.data,
    #         dims=_qdims_new.dims,
    #     )
        
    
    def __mul__(self, other):
        if isinstance(other, Qarray):
            return self.__matmul__(other)
     

        other = other + 0.0j
        if not robust_isscalar(other) and len(other.shape) > 0: # not a scalar
            other = other.reshape(other.shape + (1,1))
            
        return Qarray.create(
            other * self.data,
            dims=self._qdims.dims,
        )

    def __rmul__(self, other):
        
        # NOTE: not possible to reach this.
        # if isinstance(other, Qarray):
        #     return self.__rmatmul__(other)
        
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)
    
    def __truediv__(self, other):
        """ For Qarray's, this only really makes sense in the context of division by a scalar. """

        if isinstance(other, Qarray):
            raise ValueError("Cannot divide a Qarray by another Qarray.")

        return self.__mul__(1/other)
    
    def __add__(self, other):
        if isinstance(other, Qarray):
            if self.dims != other.dims:
                msg = (
                    "Dimensions are incompatible: "
                    + repr(self.dims) + " and " + repr(other.dims)
                )
                raise ValueError(msg)
            return Qarray.create(self.data + other.data, dims=self.dims)

        if robust_isscalar(other) and other == 0:
            return self.copy()

        if self.data.shape[-2] == self.data.shape[-1]:
            other = other + 0.0j
            if not robust_isscalar(other) and len(other.shape) > 0: # not a scalar
                other = other.reshape(other.shape + (1,1))
            other = Qarray.create(other * jnp.eye(self.data.shape[-2], dtype=self.data.dtype), dims=self.dims)
            return self.__add__(other)
        
        return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):    
        if isinstance(other, Qarray):
            if self.dims != other.dims:
                msg = (
                    "Dimensions are incompatible: "
                    + repr(self.dims) + " and " + repr(other.dims)
                )
                raise ValueError(msg)
            return Qarray.create(self.data - other.data, dims=self.dims)

        if robust_isscalar(other) and other == 0:
            return self.copy()

        if self.data.shape[-2] == self.data.shape[-1]:
            other = other + 0.0j
            if not robust_isscalar(other) and len(other.shape) > 0: # not a scalar
                other = other.reshape(other.shape + (1,1))
            other = Qarray.create(other * jnp.eye(self.data.shape[-2], dtype=self.data.dtype), dims=self.dims)
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
    
    # ----

    # String Representation ----
    def _str_header(self):
        out = ", ".join([
            "Quantum array: dims = " + str(self.dims),
            "bdims = " + str(self.bdims),
            "shape = " + str(self._data.shape),
            "type = " + str(self.qtype),
        ])
        return out

    def __str__(self):
        return self._str_header() + "\nQarray data =\n" + str(self.data)
        
    @property
    def header(self):
        """ Print the header of the Qarray. """
        return self._str_header()

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
            _qdims = deepcopy(self._qdims, memo=memo),
            _bdims = deepcopy(self._bdims, memo=memo)
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
                return Qarray.create(
                    res,
                    dims=self._qdims.dims
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

    def is_vec(self):
        return self.qtype == Qtypes.ket or self.qtype == Qtypes.bra
    
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

    def norm(self):
        return norm(self)

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
    
    def collapse(self, mode="sum"):
        return collapse(self, mode=mode)    
    # ----

ARRAY_TYPES = (Array, ndarray, Qarray)

# Qarray operations ---------------------------------------------------------------------

def collapse(qarr: Qarray, mode="sum") -> Qarray:
    """Collapse the Qarray.

    Args:
        qarr (Qarray): quantum array array
    
    Returns:
        Collapsed quantum array
    """
    if mode == "sum":
        if len(qarr.bdims) == 0:
            return self

        batch_axes = list(range(len(qarr.bdims)))
        return Qarray.create(
            jnp.sum(qarr.data, axis=batch_axes),
            dims=qarr.dims
        )
    

def transpose(qarr: Qarray, indices: List[int]) -> Qarray:
    """ Transpose the quantum array.

    Args:
        qarr (Qarray): quantum array
        *args: axes to transpose
    
    Returns:
        tranposed Qarray
    """

    indices = list(indices)

    shaped_data = qarr.shaped_data
    dims = qarr.dims
    bdims_indxs = list(range(len(qarr.bdims)))

    reshape_indices = indices + [j + len(dims[0]) for j in indices]

    reshape_indices = bdims_indxs + [j + len(bdims_indxs) for j in reshape_indices]

    shaped_data = shaped_data.transpose(reshape_indices)
    new_dims = (tuple([dims[0][j] for j in indices]), tuple([dims[1][j] for j in indices]))

    full_dims = prod(dims[0])
    full_data = shaped_data.reshape(*qarr.bdims, full_dims, -1)
    return Qarray.create(full_data, dims = new_dims)

def unit(qarr: Qarray) -> Qarray:
    """Normalize the quantum array.

    Args:
        qarr (Qarray): quantum array

    Returns:
        Normalized quantum array
    """
    data = qarr.data
    data = data / qarr.norm()
    return Qarray.create(data, dims=qarr.dims)

def norm(qarr: Qarray) -> float:
    data = qarr.data
    data_dag = qarr.dag().data

    if qarr.qtype == Qtypes.oper:
        evals, _ = jnp.linalg.eigh(data @ data_dag)
        rho_norm = jnp.sum(jnp.sqrt(jnp.abs(evals)))
        return rho_norm
    elif qarr.qtype in [Qtypes.ket, Qtypes.bra]:
        return jnp.linalg.norm(data)

def tensor(*args, **kwargs) -> Qarray:
    """Tensor product.

    Args:
        *args (Qarray): tensors to take the product of
        parallel (bool): if True, use parallel einsum for tensor product 
            true: [A,B] ^ [C,D] = [A^C, B^D]
            false: [A,B] ^ [C,D] = [A^C, A^D, B^C, B^D]

    Returns:
        Tensor product of given tensors

    """

    parallel = kwargs.pop("parallel", False)
    
    data = args[0].data
    dims = deepcopy(args[0].dims)
    dims_0 = dims[0]
    dims_1 = dims[1]
    for arg in args[1:]:

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

            data = jnp.einsum("...ij,...kl->...ikjl", a, b).reshape(*batch_dim, a.shape[-2] * b.shape[-2], -1)
        else:
            data = jnp.kron(data, arg.data, **kwargs)

        dims_0 = dims_0 + arg.dims[0]
        dims_1 = dims_1 + arg.dims[1]

    return Qarray.create(data, dims=(dims_0, dims_1))

def tr(qarr: Qarray, **kwargs) -> Array:
    """Full trace.

    Args:
        qarr (Qarray): quantum array    

    Returns:
        Full trace.
    """
    axis1 = kwargs.get("axis1", -2)
    axis2 = kwargs.get("axis2", -1)
    return jnp.trace(qarr.data, axis1=axis1, axis2=axis2, **kwargs)

def trace(qarr: Qarray, **kwargs) -> Array:
    """Full trace.

    Args:
        qarr (Qarray): quantum array    

    Returns:
        Full trace.
    """
    return tr(qarr, **kwargs)
    


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

def powm(qarr: Qarray, n: Union[int, float]) -> Qarray:
    """Matrix power.

    Args:
        qarr (Qarray): quantum array
        n (int): power

    Returns:
        matrix power
    """
    if type(n) == int:
        data_res = jnp.linalg.matrix_power(qarr.data, n)
    else:
        evalues, evectors = jnp.linalg.eig(qarr.data)
        if not (evalues >= 0).all():
            raise ValueError("Non-integer power of a matrix can only be "
                             "computed if the matrix is positive semi-definite."
                             "Got a matrix with a negative eigenvalue.")
        data_res = evectors * jnp.pow(evalues, n) @ jnp.linalg.inv(evectors)
    return Qarray.create(data_res, dims=qarr.dims)
    
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
    if len(qarr.bdims) > 0:
        raise ValueError("Cannot keep only diagonal elements of a batched Qarray.")

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
    idxs_sorted = jnp.argsort(evals, axis=-1)

    dims = ket_from_op_dims(qarr.dims)

    evals = jnp.take_along_axis(evals, idxs_sorted, axis=-1)
    evecs = jnp.take_along_axis(evecs, idxs_sorted[...,None,:], axis=-1)

    evecs = Qarray.create(
        evecs,
        dims=dims,
        bdims=evecs.shape[:-1],
    )

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

    bdims = qarr.bdims
    len_bdims = len(bdims)
    bdims_indxs = list(range(len_bdims))
    indxs = bdims_indxs + [j + len_bdims for j in indxs]
    rho = rho.transpose(indxs)

    for j in range(Nq - 1):
        rho = jnp.trace(rho, axis1=2+len_bdims, axis2=3+len_bdims)

    return Qarray.create(rho)

def dag(qarr: Qarray) -> Qarray:
    """Conjugate transpose.

    Args:
        qarr (Qarray): quantum array

    Returns:
        conjugate transpose of qarr
    """
    dims = qarr.dims[::-1]

    data = dag_data(qarr.data)
    
    return Qarray.create(data, dims=dims)

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

def is_dm_data(data: Array) -> bool:
    """Check if data is a density matrix.

    Args:
        data: matrix
    Returns:
        True if data is a density matrix
    """
    return data.shape[-2] == data.shape[-1]

def powm_data(data: Array, n: int) -> Array:
    """Matrix power.

    Args:
        data: matrix
        n: power

    Returns:
        matrix power
    """
    return jnp.linalg.matrix_power(data, n)
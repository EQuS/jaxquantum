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

# Module-level registry mapping impl_class -> QarrayImplType member
_IMPL_REGISTRY: dict = {}


class QarrayImplType(Enum):
    """Enumeration of available Qarray storage backends.

    Each member maps one-to-one with a concrete ``QarrayImpl`` subclass.
    New backends should call ``QarrayImplType.register(MyImpl, QarrayImplType.MY_TYPE)``
    immediately after defining their impl class.

    Members:
        DENSE: Standard JAX dense array (``jnp.ndarray``).
        SPARSE: JAX experimental BCOO sparse array.
    """

    DENSE = "dense"
    SPARSE = "sparse"

    @classmethod
    def register(cls, impl_class, member):
        """Register an implementation class with a QarrayImplType member.

        Args:
            impl_class: The concrete ``QarrayImpl`` subclass to register.
            member: The ``QarrayImplType`` enum member to associate with it.
        """
        _IMPL_REGISTRY[impl_class] = member

    @classmethod
    def has(cls, x) -> bool:
        """Return True if x corresponds to a member of QarrayImplType.

        Accepts an existing ``QarrayImplType`` member, a string equal to the
        member name or value (case-insensitive), or an implementation class
        (e.g. ``DenseImpl``, ``SparseImpl``) that has been registered.

        Args:
            x: Value to test — a ``QarrayImplType``, ``str``, or impl class.

        Returns:
            True if ``x`` maps to a known ``QarrayImplType`` member.
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
        """Return the ``QarrayImplType`` member associated with *impl_class*.

        Args:
            impl_class: A concrete ``QarrayImpl`` subclass that has been
                registered via :meth:`register`.

        Returns:
            The corresponding ``QarrayImplType`` member.

        Raises:
            ValueError: If *impl_class* is not in the registry.
        """
        if impl_class in _IMPL_REGISTRY:
            return _IMPL_REGISTRY[impl_class]
        raise ValueError(f"Unknown implementation class: {impl_class}")

    def get_impl_class(self):
        """Return the implementation class registered for this member.

        Returns:
            The concrete ``QarrayImpl`` subclass associated with this member.

        Raises:
            ValueError: If no class has been registered for this member.
        """
        for cls_key, member in _IMPL_REGISTRY.items():
            if member is self:
                return cls_key
        raise ValueError(f"No impl class registered for {self}")


def robust_asarray(data) -> Union[Array, sparse.BCOO]:
    """Convert *data* to a JAX array, leaving sparse BCOO arrays untouched.

    Args:
        data: Input data — any array-like or ``sparse.BCOO``.

    Returns:
        A ``jax.Array`` or ``sparse.BCOO``.
    """
    if isinstance(data, sparse.BCOO):
        return data
    return jnp.asarray(data)


class QarrayImpl(ABC):
    """Abstract base class defining the interface every storage backend must implement.

    A ``QarrayImpl`` wraps a raw data array (dense ``jnp.ndarray`` or sparse
    ``BCOO``) and provides the mathematical primitives used by ``Qarray``.
    Concrete subclasses must implement every ``@abstractmethod``.

    Attributes:
        PROMOTION_ORDER: Integer priority used by ``_coerce`` to decide which
            side to promote when operands have different types.  Higher means
            "more general" (``DenseImpl = 1``, ``SparseImpl = 0``).
    """

    PROMOTION_ORDER: int = 0  # override in subclasses; higher = more general

    @abstractmethod
    def get_data(self) -> Array:
        """Return the underlying raw data array."""
        pass

    @property
    def data(self) -> Array:
        """The underlying raw data array."""
        return self.get_data()

    @property
    def impl_type(self) -> QarrayImplType:
        """The ``QarrayImplType`` member corresponding to this instance."""
        return QarrayImplType.from_impl_class(type(self))

    @classmethod
    @abstractmethod
    def from_data(cls, data) -> "QarrayImpl":
        """Wrap raw data in this impl type.

        Args:
            data: Raw array data (dense ``jnp.ndarray`` or ``sparse.BCOO``).

        Returns:
            A new instance of this implementation wrapping *data*.
        """
        pass

    @abstractmethod
    def matmul(self, other: "QarrayImpl") -> "QarrayImpl":
        """Matrix multiplication with *other*.

        Args:
            other: Right-hand operand.

        Returns:
            Result of ``self @ other`` as a ``QarrayImpl``.
        """
        pass

    @abstractmethod
    def add(self, other: "QarrayImpl") -> "QarrayImpl":
        """Element-wise addition with *other*.

        Args:
            other: Right-hand operand.

        Returns:
            Result of ``self + other`` as a ``QarrayImpl``.
        """
        pass

    @abstractmethod
    def sub(self, other: "QarrayImpl") -> "QarrayImpl":
        """Element-wise subtraction of *other*.

        Args:
            other: Right-hand operand.

        Returns:
            Result of ``self - other`` as a ``QarrayImpl``.
        """
        pass

    @abstractmethod
    def mul(self, scalar) -> "QarrayImpl":
        """Scalar multiplication.

        Args:
            scalar: Scalar value to multiply by.

        Returns:
            Result of ``scalar * self`` as a ``QarrayImpl``.
        """
        pass

    @abstractmethod
    def dag(self) -> "QarrayImpl":
        """Conjugate transpose.

        Returns:
            The conjugate transpose of this array as a ``QarrayImpl``.
        """
        pass

    @abstractmethod
    def to_dense(self) -> "DenseImpl":
        """Convert to a ``DenseImpl``.

        Returns:
            A ``DenseImpl`` wrapping the same data.
        """
        pass

    @abstractmethod
    def to_sparse(self) -> "SparseImpl":
        """Convert to a ``SparseImpl``.

        Returns:
            A ``SparseImpl`` wrapping the same data.
        """
        pass

    @abstractmethod
    def shape(self) -> tuple:
        """Shape of the underlying data array.

        Returns:
            Tuple of dimension sizes.
        """
        pass

    @abstractmethod
    def dtype(self):
        """Data type of the underlying array.

        Returns:
            A numpy/JAX dtype object.
        """
        pass

    @abstractmethod
    def __deepcopy__(self, memo=None):
        pass

    @abstractmethod
    def tidy_up(self, atol):
        """Zero out values whose magnitude is below *atol*.

        Args:
            atol: Absolute tolerance threshold.

        Returns:
            A new ``QarrayImpl`` with small values zeroed.
        """
        pass

    @abstractmethod
    def kron(self, other: "QarrayImpl") -> "QarrayImpl":
        """Kronecker (tensor) product with another implementation.

        Args:
            other: Right-hand operand.  Mixed-type pairs are handled by
                ``_coerce`` — the result has the higher ``PROMOTION_ORDER``
                type (dense wins over sparse).

        Returns:
            A new ``QarrayImpl`` containing the Kronecker product.
        """
        pass

    @classmethod
    @abstractmethod
    def _eye_data(cls, n: int, dtype=None):
        """Create identity matrix data of size n.

        Args:
            n: Matrix size.
            dtype: Optional data type for the identity entries.

        Returns:
            Raw identity matrix data in the format appropriate for this impl.
        """
        pass

    @classmethod
    @abstractmethod
    def can_handle_data(cls, arr) -> bool:
        """Return True if *arr* is a raw data type natively handled by this impl.

        Used by the module-level :func:`dag_data` dispatcher to route raw
        arrays to the correct backend without any isinstance chain outside the
        impl classes.

        Args:
            arr: Raw array — e.g. ``jnp.ndarray`` for ``DenseImpl`` or
                ``sparse.BCOO`` for ``SparseImpl``.

        Returns:
            True if this impl can operate on *arr* without conversion.
        """
        pass

    @classmethod
    @abstractmethod
    def dag_data(cls, arr):
        """Conjugate transpose of raw data in this impl's native format.

        Implementations must handle batched arrays (last two axes are
        swapped) and must not densify sparse arrays.

        Args:
            arr: Raw array in this impl's native format.

        Returns:
            Conjugate transpose with the last two axes swapped.
        """
        pass

    def _promote_to(self, target_cls: type) -> "QarrayImpl":
        """Convert this impl to *target_cls* by passing through dense.

        Args:
            target_cls: The destination ``QarrayImpl`` subclass.

        Returns:
            An instance of *target_cls* holding equivalent data.
        """
        if isinstance(self, target_cls):
            return self
        return target_cls.from_data(self.to_dense()._data)

    def _coerce(self, other: "QarrayImpl") -> "tuple[QarrayImpl, QarrayImpl]":
        """Coerce *self* and *other* to the same implementation type.

        The impl type with the higher ``PROMOTION_ORDER`` wins; the other side
        is promoted via :meth:`_promote_to`.

        Args:
            other: The other operand.

        Returns:
            A pair ``(a, b)`` of the same ``QarrayImpl`` subclass, suitable
            for a binary operation.
        """
        if type(self) is type(other):
            return self, other
        if self.PROMOTION_ORDER >= other.PROMOTION_ORDER:
            return self, other._promote_to(type(self))
        return self._promote_to(type(other)), other


@struct.dataclass
class DenseImpl(QarrayImpl):
    """Dense implementation using JAX dense arrays.

    Attributes:
        _data: The underlying ``jnp.ndarray``.
    """

    _data: Array

    PROMOTION_ORDER = 1  # noqa: RUF012 — not a struct field; no annotation intentional

    @classmethod
    def from_data(cls, data) -> "DenseImpl":
        """Wrap *data* in a new ``DenseImpl``.

        Args:
            data: Array-like input data.

        Returns:
            A ``DenseImpl`` wrapping ``robust_asarray(data)``.
        """
        return cls(_data=robust_asarray(data))

    def get_data(self) -> Array:
        """Return the underlying dense array."""
        return self._data

    def matmul(self, other: QarrayImpl) -> QarrayImpl:
        """Matrix multiply ``self @ other``, coercing types as needed.

        Args:
            other: Right-hand operand.

        Returns:
            A ``DenseImpl`` containing the matrix product.
        """
        a, b = self._coerce(other)
        if a is not self:
            return a.matmul(b)
        return DenseImpl(self._data @ b._data)

    def add(self, other: QarrayImpl) -> QarrayImpl:
        """Element-wise addition ``self + other``, coercing types as needed.

        Args:
            other: Right-hand operand.

        Returns:
            A ``DenseImpl`` containing the sum.
        """
        a, b = self._coerce(other)
        if a is not self:
            return a.add(b)
        return DenseImpl(self._data + b._data)

    def sub(self, other: QarrayImpl) -> QarrayImpl:
        """Element-wise subtraction ``self - other``, coercing types as needed.

        Args:
            other: Right-hand operand.

        Returns:
            A ``DenseImpl`` containing the difference.
        """
        a, b = self._coerce(other)
        if a is not self:
            return a.sub(b)
        return DenseImpl(self._data - b._data)

    def mul(self, scalar) -> QarrayImpl:
        """Scalar multiplication.

        Args:
            scalar: Scalar value.

        Returns:
            A ``DenseImpl`` with each element multiplied by *scalar*.
        """
        return DenseImpl(scalar * self._data)

    def dag(self) -> QarrayImpl:
        """Conjugate transpose.

        Returns:
            A ``DenseImpl`` containing the conjugate transpose.
        """
        return DenseImpl(jnp.moveaxis(jnp.conj(self._data), -1, -2))

    def to_dense(self) -> "DenseImpl":
        """Return self (already dense).

        Returns:
            This ``DenseImpl`` instance unchanged.
        """
        return self

    def to_sparse(self) -> "SparseImpl":
        """Convert to a ``SparseImpl`` via ``BCOO.fromdense``.

        Returns:
            A ``SparseImpl`` wrapping a BCOO conversion of this array.
        """
        sparse_data = sparse.BCOO.fromdense(self._data)
        return SparseImpl(sparse_data)

    def shape(self) -> tuple:
        """Shape of the underlying dense array.

        Returns:
            Tuple of dimension sizes.
        """
        return self._data.shape

    def dtype(self):
        """Data type of the underlying dense array.

        Returns:
            The dtype of ``_data``.
        """
        return self._data.dtype

    def frobenius_norm(self) -> float:
        """Compute the Frobenius norm.

        Returns:
            The Frobenius norm as a scalar.
        """
        return jnp.sqrt(jnp.sum(jnp.abs(self._data) ** 2))

    def real(self) -> QarrayImpl:
        """Element-wise real part.

        Returns:
            A ``DenseImpl`` containing the real parts.
        """
        return DenseImpl(jnp.real(self._data))

    def imag(self) -> QarrayImpl:
        """Element-wise imaginary part.

        Returns:
            A ``DenseImpl`` containing the imaginary parts.
        """
        return DenseImpl(jnp.imag(self._data))

    def conj(self) -> QarrayImpl:
        """Element-wise complex conjugate.

        Returns:
            A ``DenseImpl`` containing the complex-conjugated values.
        """
        return DenseImpl(jnp.conj(self._data))

    def __deepcopy__(self, memo=None):
        return DenseImpl(
            _data=deepcopy(self._data, memo)
        )

    def tidy_up(self, atol):
        """Zero out real/imaginary parts whose magnitude is below *atol*.

        Args:
            atol: Absolute tolerance threshold.

        Returns:
            A new ``DenseImpl`` with small values zeroed.
        """
        data = self._data
        data_re = jnp.real(data)
        data_im = jnp.imag(data)
        data_re_mask = jnp.abs(data_re) > atol
        data_im_mask = jnp.abs(data_im) > atol
        data_new = data_re * data_re_mask + 1j * data_im * data_im_mask

        return DenseImpl(
            _data=data_new
        )

    def kron(self, other: "QarrayImpl") -> "QarrayImpl":
        """Kronecker product using ``jnp.kron``.

        Args:
            other: Right-hand operand.

        Returns:
            A ``DenseImpl`` containing the Kronecker product.
        """
        a, b = self._coerce(other)
        if a is not self:
            return a.kron(b)
        return DenseImpl(jnp.kron(self._data, b._data))

    @classmethod
    def _eye_data(cls, n: int, dtype=None):
        """Create an ``n x n`` identity matrix as a dense JAX array.

        Args:
            n: Matrix size.
            dtype: Optional data type.

        Returns:
            A ``jnp.ndarray`` identity matrix of shape ``(n, n)``.
        """
        return jnp.eye(n, dtype=dtype)

    @classmethod
    def can_handle_data(cls, arr) -> bool:
        """Return True for any non-BCOO array.

        Args:
            arr: Raw array.

        Returns:
            True when *arr* is not a ``sparse.BCOO`` (dense arrays and
            array-likes are accepted).
        """
        return not isinstance(arr, sparse.BCOO)

    @classmethod
    def dag_data(cls, arr) -> Array:
        """Conjugate transpose for dense arrays.

        Swaps the last two axes via :func:`jnp.moveaxis` and conjugates all
        elements.  For 1-D inputs only conjugation is applied.

        Args:
            arr: Dense array.

        Returns:
            Conjugate transpose with the last two axes swapped.
        """
        if len(arr.shape) == 1:
            return jnp.conj(arr)
        return jnp.moveaxis(jnp.conj(arr), -1, -2)


@struct.dataclass
class SparseImpl(QarrayImpl):
    """Sparse implementation using JAX experimental BCOO sparse arrays.

    Attributes:
        _data: The underlying ``sparse.BCOO`` array.
    """

    _data: sparse.BCOO

    PROMOTION_ORDER = 0  # noqa: RUF012 — not a struct field; no annotation intentional

    @classmethod
    def from_data(cls, data) -> "SparseImpl":
        """Wrap *data* in a new ``SparseImpl``, converting to BCOO if needed.

        Args:
            data: A ``sparse.BCOO`` or array-like input.

        Returns:
            A ``SparseImpl`` wrapping a BCOO representation of *data*.
        """
        return cls(_data=cls._to_sparse(data))

    def get_data(self) -> Array:
        """Return the underlying BCOO sparse array."""
        return self._data

    def matmul(self, other: QarrayImpl) -> QarrayImpl:
        """Matrix multiply ``self @ other``.

        When *other* is a ``DenseImpl``, JAX's native BCOO @ dense path is
        used (no self-densification).  When *other* is also a ``SparseImpl``,
        a sparse @ sparse product is performed.

        Args:
            other: Right-hand operand.

        Returns:
            A ``DenseImpl`` (sparse @ dense) or ``SparseImpl`` (sparse @
            sparse) containing the matrix product.
        """
        if isinstance(other, DenseImpl):
            # Native JAX: BCOO @ dense Array → dense Array (no self-densification)
            return DenseImpl(self._data @ other._data)
        a, b = self._coerce(other)
        if a is not self:
            return a.matmul(b)
        # both SparseImpl
        return SparseImpl(self._data @ b._data)

    def add(self, other: QarrayImpl) -> QarrayImpl:
        """Element-wise addition ``self + other``, coercing types as needed.

        Args:
            other: Right-hand operand.

        Returns:
            A ``SparseImpl`` (both sparse) or ``DenseImpl`` (mixed) sum.
        """
        a, b = self._coerce(other)
        if a is not self:
            return a.add(b)
        # both SparseImpl
        x, y = self._data, b._data
        if x.indices.dtype != y.indices.dtype:
            y = sparse.BCOO((y.data, y.indices.astype(x.indices.dtype)), shape=y.shape)
        return SparseImpl(x + y)

    def sub(self, other: QarrayImpl) -> QarrayImpl:
        """Element-wise subtraction ``self - other``, coercing types as needed.

        Args:
            other: Right-hand operand.

        Returns:
            A ``SparseImpl`` (both sparse) or ``DenseImpl`` (mixed) difference.
        """
        a, b = self._coerce(other)
        if a is not self:
            return a.sub(b)
        # both SparseImpl
        x, y = self._data, b._data
        if x.indices.dtype != y.indices.dtype:
            y = sparse.BCOO((y.data, y.indices.astype(x.indices.dtype)), shape=y.shape)
        return SparseImpl(x - y)

    def mul(self, scalar) -> QarrayImpl:
        """Scalar multiplication.

        Args:
            scalar: Scalar value.

        Returns:
            A ``SparseImpl`` with each stored value multiplied by *scalar*.
        """
        return SparseImpl(scalar * self._data)

    def dag(self) -> QarrayImpl:
        """Conjugate transpose without densifying.

        Transposes the last two dimensions of the BCOO array and conjugates
        the stored values.

        Returns:
            A ``SparseImpl`` containing the conjugate transpose.
        """
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
        """Convert to a ``DenseImpl`` via ``todense()``.

        Returns:
            A ``DenseImpl`` with the same values as this sparse array.
        """
        return DenseImpl(self._data.todense())

    @classmethod
    def _to_sparse(cls, data) -> sparse.BCOO:
        """Convert *data* to a ``sparse.BCOO``, returning it unchanged if already sparse.

        Args:
            data: A ``sparse.BCOO`` or array-like.

        Returns:
            A ``sparse.BCOO`` representation of *data*.
        """
        if isinstance(data, sparse.BCOO):
            return data
        return sparse.BCOO.fromdense(data)

    def to_sparse(self) -> "SparseImpl":
        """Return self (already sparse).

        Returns:
            This ``SparseImpl`` instance unchanged.
        """
        return self

    def shape(self) -> tuple:
        """Shape of the underlying BCOO array.

        Returns:
            Tuple of dimension sizes.
        """
        return self._data.shape

    def dtype(self):
        """Data type of the underlying BCOO array.

        Returns:
            The dtype of ``_data``.
        """
        return self._data.dtype

    def frobenius_norm(self) -> float:
        """Compute the Frobenius norm directly from stored values.

        Avoids converting to dense by computing ``sqrt(sum(|data|^2))``
        over the stored BCOO values.

        Returns:
            The Frobenius norm as a scalar.
        """
        return jnp.sqrt(jnp.sum(jnp.abs(self._data.data) ** 2))

    @classmethod
    def _real(cls, data):
        """Return a BCOO array with only the real parts of the stored values.

        Args:
            data: A ``sparse.BCOO`` array.

        Returns:
            A ``sparse.BCOO`` with ``jnp.real`` applied to stored values.
        """
        return sparse.BCOO(
            (jnp.real(data.data), data.indices),
            shape=data.shape
        )

    def real(self) -> QarrayImpl:
        """Element-wise real part.

        Returns:
            A ``SparseImpl`` containing the real parts of stored values.
        """
        return SparseImpl(SparseImpl._real(self._data))

    @classmethod
    def _imag(cls, data):
        """Return a BCOO array with only the imaginary parts of the stored values.

        Args:
            data: A ``sparse.BCOO`` array.

        Returns:
            A ``sparse.BCOO`` with ``jnp.imag`` applied to stored values.
        """
        return sparse.BCOO(
            (jnp.imag(data.data), data.indices),
            shape=data.shape
        )

    def imag(self) -> QarrayImpl:
        """Element-wise imaginary part.

        Returns:
            A ``SparseImpl`` containing the imaginary parts of stored values.
        """
        return SparseImpl(SparseImpl._imag(self._data))

    @classmethod
    def _conj(cls, data):
        """Return a BCOO array with complex-conjugated stored values.

        Args:
            data: A ``sparse.BCOO`` array.

        Returns:
            A ``sparse.BCOO`` with ``jnp.conj`` applied to stored values.
        """
        return sparse.BCOO(
            (jnp.conj(data.data), data.indices),
            shape=data.shape
        )

    def conj(self) -> QarrayImpl:
        """Element-wise complex conjugate.

        Returns:
            A ``SparseImpl`` containing the complex-conjugated stored values.
        """
        return SparseImpl(SparseImpl._conj(self._data))

    @classmethod
    def _abs(cls, data):
        """Return a BCOO array with absolute values of stored entries.

        Args:
            data: A ``sparse.BCOO`` array.

        Returns:
            A sparsified ``jnp.abs`` of *data*.
        """
        return sparse.sparsify(jnp.abs)(data)

    def abs(self) -> QarrayImpl:
        """Element-wise absolute value.

        Returns:
            A ``SparseImpl`` containing the absolute values of stored entries.
        """
        return SparseImpl(SparseImpl._abs(self._data))

    @classmethod
    def _eye_data(cls, n: int, dtype=None):
        """Create an ``n x n`` identity matrix as a sparse BCOO with O(n) memory.

        No dense intermediate is allocated.

        Args:
            n: Matrix size.
            dtype: Optional data type.

        Returns:
            A ``sparse.BCOO`` identity matrix of shape ``(n, n)``.
        """
        return sparse.eye(n, dtype=dtype)

    @classmethod
    def can_handle_data(cls, arr) -> bool:
        """Return True when *arr* is a ``sparse.BCOO`` array.

        Args:
            arr: Raw array.

        Returns:
            True if *arr* is a ``sparse.BCOO`` instance.
        """
        return isinstance(arr, sparse.BCOO)

    @classmethod
    def dag_data(cls, arr: sparse.BCOO) -> sparse.BCOO:
        """Conjugate transpose for BCOO sparse arrays without densifying.

        Uses :func:`jax.experimental.sparse.bcoo_transpose` to swap the last
        two dimensions and conjugates the stored non-zero values in place.

        Args:
            arr: A ``sparse.BCOO`` array with ``ndim >= 2``.

        Returns:
            A ``sparse.BCOO`` containing the conjugate transpose.
        """
        ndim = arr.ndim
        permutation = tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
        transposed = sparse.bcoo_transpose(arr, permutation=permutation)
        return sparse.BCOO(
            (jnp.conj(transposed.data), transposed.indices),
            shape=transposed.shape,
        )

    def trace(self) -> Array:
        """Compute the trace of the last two matrix dimensions without densifying.

        For a non-batched ``(M, N)`` matrix: returns a scalar.
        For a batched ``(*batch, M, N)`` matrix: returns an array of shape
        ``*batch``.

        Relies on the JAX BCOO ``n_batch=0`` convention used by
        :meth:`_to_sparse`: all dimensions are sparse, so ``indices`` has
        shape ``(nse, ndim)`` and the last two index columns are the
        row/column indices.

        Returns:
            Trace value(s).
        """
        indices = self._data.indices   # (nse, ndim)  — n_batch=0 in JAX BCOO
        values  = self._data.data      # (nse,)
        ndim = indices.shape[-1]

        # Diagonal mask: entries where the row and column index are the same
        is_diag = indices[:, -2] == indices[:, -1]  # (nse,) bool

        if ndim == 2:
            # Non-batched (M, N): simple masked sum
            return jnp.sum(values * is_diag)
        else:
            # Batched (*batch, M, N): scatter-add per batch element
            batch_shape = self._data.shape[:-2]
            B = int(jnp.prod(jnp.array(batch_shape)))
            # Compute cumulative strides for the batch dimensions
            strides = [1]
            for s in reversed(batch_shape[1:]):
                strides.insert(0, strides[0] * s)
            strides = jnp.array(strides, dtype=jnp.int32)  # (n_batch_dims,)
            # indices[:, :-2] are batch indices, shape (nse, n_batch_dims)
            flat_batch_idx = jnp.sum(indices[:, :-2] * strides, axis=-1)  # (nse,)
            result = jnp.zeros(B, dtype=values.dtype).at[flat_batch_idx].add(
                values * is_diag
            )
            return result.reshape(batch_shape)

    def keep_only_diag(self) -> "SparseImpl":
        """Zero out off-diagonal stored entries without densifying.

        Multiplies stored values by a diagonal mask so that only entries
        whose last two index coordinates are equal (i.e. the matrix diagonal)
        survive.  The BCOO index structure is preserved unchanged.

        Returns:
            A ``SparseImpl`` with only diagonal entries non-zero.
        """
        indices = self._data.indices   # (nse, ndim)
        values  = self._data.data      # (nse,)
        is_diag = indices[:, -2] == indices[:, -1]
        new_values = values * is_diag
        return SparseImpl(sparse.BCOO((new_values, indices), shape=self._data.shape))

    def l2_norm_batched(self, bdims: tuple) -> Array:
        """Compute the L2 norm per batch element without densifying.

        For non-batched (``bdims=()``) arrays: returns a scalar equal to
        ``sqrt(sum|x_i|^2)``.  For batched arrays (``(*bdims, ...)``):
        returns an array of shape ``*bdims`` where each entry is the L2 norm
        of the corresponding batch element.

        Uses a scatter-add over the batch indices so that the BCOO values
        array (shape ``(nse,)`` for ``n_batch=0``) is traversed only once.

        Args:
            bdims: Tuple of batch dimension sizes.

        Returns:
            Scalar or array of L2 norms.
        """
        values  = self._data.data      # (nse,) for n_batch=0
        indices = self._data.indices   # (nse, ndim)
        n_batch_dims = len(bdims)
        sq = jnp.abs(values) ** 2

        if n_batch_dims == 0:
            # Non-batched: global sum
            return jnp.sqrt(jnp.sum(sq))
        else:
            B = int(jnp.prod(jnp.array(bdims)))
            strides = [1]
            for s in reversed(bdims[1:]):
                strides.insert(0, strides[0] * s)
            strides = jnp.array(strides, dtype=jnp.int32)  # (n_batch_dims,)
            # indices[:, :n_batch_dims] are the batch coordinate columns
            flat_batch_idx = jnp.sum(
                indices[:, :n_batch_dims] * strides, axis=-1
            )  # (nse,)
            sum_sq = (
                jnp.zeros(B, dtype=jnp.float64)
                .at[flat_batch_idx]
                .add(sq)
            )
            return jnp.sqrt(sum_sq).reshape(bdims)

    def __deepcopy__(self, memo=None):
        return SparseImpl(
            _data=deepcopy(self._data, memo)
        )

    def tidy_up(self, atol):
        """Zero out stored values whose real or imaginary magnitude is below *atol*.

        Zeroes out the real and/or imaginary parts of stored BCOO values that
        fall below *atol*.  Structural zeros (indices) are kept; only the
        values are filtered.  This mirrors the ``DenseImpl`` behaviour but
        avoids converting to a dense array.

        Args:
            atol: Absolute tolerance threshold.

        Returns:
            A new ``SparseImpl`` with small values zeroed.
        """
        values = self._data.data
        re = jnp.real(values)
        im = jnp.imag(values)
        new_values = re * (jnp.abs(re) > atol) + 1j * im * (jnp.abs(im) > atol)
        return SparseImpl(sparse.BCOO((new_values, self._data.indices), shape=self._data.shape))

    def kron(self, other: "QarrayImpl") -> "QarrayImpl":
        """Kronecker product using ``sparsify(jnp.kron)`` — stays sparse.

        Uses JAX's ``sparsify`` transform so the Kronecker product is computed
        natively on BCOO arrays without materialising a dense intermediate.

        Args:
            other: Right-hand operand.

        Returns:
            A ``SparseImpl`` containing the Kronecker product when both operands
            are sparse; a ``DenseImpl`` when types differ (via ``_coerce``).
        """
        a, b = self._coerce(other)
        if a is not self:
            return a.kron(b)
        sparse_kron = sparse.sparsify(jnp.kron)
        return SparseImpl(sparse_kron(self._data, b._data))



# Register implementation classes with the enum registry
QarrayImplType.register(DenseImpl, QarrayImplType.DENSE)
QarrayImplType.register(SparseImpl, QarrayImplType.SPARSE)


@struct.dataclass
class Qarray(Generic[ImplT]):
    """Quantum array with a pluggable storage backend.

    ``Qarray`` wraps a ``QarrayImpl`` together with quantum-mechanical
    dimension metadata (``_qdims``) and optional batch dimensions
    (``_bdims``).  The default backend is dense (``DenseImpl``); pass
    ``implementation="sparse"`` (or ``QarrayImplType.SPARSE``) to store data
    as a JAX BCOO sparse array.

    Attributes:
        _impl: The storage backend holding the raw data.
        _qdims: Quantum dimension metadata (bra/ket structure, Hilbert space
            sizes).
        _bdims: Tuple of batch dimension sizes (empty tuple = non-batched).

    Example:
        >>> import jaxquantum as jqt
        >>> a = jqt.destroy(10, implementation="sparse")
        >>> a.is_sparse
        True
    """

    _impl: ImplT
    _qdims: Qdims = struct.field(pytree_node=False)
    _bdims: tuple[int] = struct.field(pytree_node=False)

    # Initialization ----
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
        """Create a ``Qarray`` from raw data.

        Handles shape normalisation, dimension inference, and tidying of small
        values.

        Args:
            data: Input data array (dense array-like or ``sparse.BCOO``).
            dims: Quantum dimensions as ``((row_dims...), (col_dims...))``.
                Inferred from *data* shape when ``None``.
            bdims: Tuple of batch dimension sizes.  Inferred from the leading
                dimensions of *data* when ``None``.
            implementation: Storage backend — ``QarrayImplType.DENSE``
                (default) or ``QarrayImplType.SPARSE``, or the equivalent
                string ``"dense"`` / ``"sparse"``.

        Returns:
            A new ``Qarray`` backed by the requested implementation.
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

        impl_class = QarrayImplType(implementation).get_impl_class()
        impl = impl_class.from_data(data)
        impl = impl.tidy_up(SETTINGS["auto_tidyup_atol"])

        return cls(impl, qdims, bdims)

    @classmethod
    @overload
    def from_sparse(cls, data, dims=None, bdims=None) -> "Qarray[SparseImpl]":
        ...

    @classmethod
    def from_sparse(cls, data, dims=None, bdims=None):
        """Create a ``Qarray`` directly from a sparse BCOO array without densifying.

        Args:
            data: A ``sparse.BCOO`` or array-like to store as sparse.
            dims: Quantum dimensions.  Inferred when ``None``.
            bdims: Batch dimensions.  Inferred when ``None``.

        Returns:
            A ``Qarray[SparseImpl]``.
        """
        return cls.create(data, dims=dims, bdims=bdims, implementation=QarrayImplType.SPARSE)

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
        """Create a batched ``Qarray`` from a list of same-shaped ``Qarray`` objects.

        The output implementation is determined by the element with the highest
        ``PROMOTION_ORDER``: if all inputs are sparse the result is sparse; if
        any input is dense (or types are mixed) all inputs are promoted to dense
        and the result is dense.

        Args:
            qarr_list: List of ``Qarray`` objects with identical ``dims`` and
                ``bdims``.  May be empty.

        Returns:
            A ``Qarray`` with an extra leading batch dimension of size
            ``len(qarr_list)``.

        Raises:
            ValueError: If the elements have mismatched ``dims`` or ``bdims``.
        """
        if len(qarr_list) == 0:
            dims = ((), ())
            bdims = (0,)
            return cls.create(jnp.array([]), dims=dims, bdims=bdims)

        dims = qarr_list[0].dims
        bdims = qarr_list[0].bdims

        if not all(qarr.dims == dims and qarr.bdims == bdims for qarr in qarr_list):
            raise ValueError("All Qarrays in the list must have the same dimensions.")

        new_bdims = (len(qarr_list),) + bdims

        # Pick the target type: highest PROMOTION_ORDER wins (dense beats sparse).
        target_impl_type = max(
            (q.impl_type for q in qarr_list),
            key=lambda t: t.get_impl_class().PROMOTION_ORDER,
        )

        if target_impl_type == QarrayImplType.SPARSE:
            # All inputs are sparse — stack via dense intermediates then re-sparsify.
            data = jnp.array([q.data.todense() for q in qarr_list])
            return cls.create(data, dims=dims, bdims=new_bdims, implementation=QarrayImplType.SPARSE)

        # Target is dense: promote any sparse inputs before stacking.
        data = jnp.array([q.to_dense().data for q in qarr_list])
        return cls.create(data, dims=dims, bdims=new_bdims, implementation=QarrayImplType.DENSE)

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
        """Create a ``Qarray`` from a (possibly nested) list of ``Qarray`` objects.

        Args:
            qarr_arr: A ``Qarray`` (returned as-is) or a nested list of
                ``Qarray`` objects.

        Returns:
            A ``Qarray`` with batch dimensions matching the nesting structure
            of *qarr_arr*.
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
        """Quantum type of this array (ket, bra, or operator)."""
        return self._qdims.qtype

    @property
    def dtype(self):
        """Data type of the underlying storage array."""
        return self._impl.dtype()

    @property
    def dims(self):
        """Quantum dimensions as ``((row_dims...), (col_dims...))``."""
        return self._qdims.dims

    @property
    def bdims(self):
        """Tuple of batch dimension sizes (empty tuple = non-batched)."""
        return self._bdims

    @property
    def qdims(self):
        """The ``Qdims`` metadata object for this array."""
        return self._qdims

    @property
    def space_dims(self):
        """Hilbert space dimensions for the relevant side (ket row / bra col)."""
        if self.qtype in [Qtypes.oper, Qtypes.ket]:
            return self.dims[0]
        elif self.qtype == Qtypes.bra:
            return self.dims[1]
        else:
            # TODO: not reached for some reason
            raise ValueError("Unsupported qtype.")

    @property
    def data(self):
        """The raw underlying data (dense ``jnp.ndarray`` or ``sparse.BCOO``)."""
        return self._impl.data

    @property
    def shaped_data(self):
        """Data reshaped to ``bdims + dims[0] + dims[1]``."""
        return self.data.reshape(self.bdims + self.dims[0] + self.dims[1])

    @property
    def shape(self):
        """Shape of the underlying data array."""
        return self.data.shape

    @property
    def is_batched(self):
        """True if this array has one or more batch dimensions."""
        return len(self.bdims) > 0

    @property
    def is_sparse(self):
        """True if the storage backend is ``SparseImpl``."""
        return self._impl.impl_type == QarrayImplType.SPARSE

    @property
    def is_dense(self):
        """True if the storage backend is ``DenseImpl``."""
        return self._impl.impl_type == QarrayImplType.DENSE

    @property
    def impl_type(self):
        """The ``QarrayImplType`` member of the current storage backend."""
        return self._impl.impl_type

    def to_sparse(self) -> "Qarray[SparseImpl]":
        """Return a sparse-backed copy of this array.

        If the array is already sparse, returns self unchanged.

        Returns:
            A ``Qarray[SparseImpl]``.
        """
        if self.is_sparse:
            return self
        new_impl = self._impl.to_sparse()
        return Qarray(new_impl, self._qdims, self._bdims)

    def to_dense(self) -> "Qarray[DenseImpl]":
        """Return a dense-backed copy of this array.

        If the array is already dense, returns self unchanged.

        Returns:
            A ``Qarray[DenseImpl]``.
        """
        if self.is_dense:
            return self
        new_impl = self._impl.to_dense()
        return Qarray(new_impl, self._qdims, self._bdims)

    def __getitem__(self, index):
        if len(self.bdims) > 0:
            return Qarray.create(
                self.data[index],
                dims=self.dims,
                implementation=self.impl_type,
            )
        else:
            raise ValueError("Cannot index a non-batched Qarray.")

    def reshape_bdims(self, *args):
        """Reshape the batch dimensions of this ``Qarray``.

        Args:
            *args: New batch dimension sizes.

        Returns:
            A new ``Qarray`` with the requested batch shape.
        """
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
        """Convert Hilbert space dimensions to full quantum dims tuple.

        Args:
            space_dims: Sequence of per-subsystem Hilbert space sizes, or a
                full ``((row_dims), (col_dims))`` tuple (returned unchanged).

        Returns:
            A ``((row_dims...), (col_dims...))`` tuple.

        Raises:
            ValueError: If ``self.qtype`` is not ket, bra, or oper.
        """
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

        Note that this does not take in qdims but rather the new Hilbert space
        dimensions.

        Args:
            *args: New Hilbert dimensions for the Qarray.

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

        Args:
            new_shape: Target shape tuple.

        Returns:
            A new ``Qarray`` with data resized via ``jnp.resize``.
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
        """Length along the first batch dimension.

        Returns:
            Size of the leading batch dimension.

        Raises:
            ValueError: If the array is not batched.
        """
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

        if self.is_sparse and other.is_sparse:
            # Fast structural path: same sparsity pattern → compare values only (no todense)
            if (self.data.indices.shape == other.data.indices.shape
                    and bool(jnp.all(self.data.indices == other.data.indices))):
                return bool(jnp.allclose(self.data.data, other.data.data))
            # Different patterns: fall back to dense comparison (unavoidable)
            return bool(jnp.all(self.data.todense() == other.data.todense()))

        # At least one dense: convert sparse side to dense for comparison
        self_data  = self.data.todense()  if hasattr(self.data,  'todense') else self.data
        other_data = other.data.todense() if hasattr(other.data, 'todense') else other.data
        return bool(jnp.all(self_data == other_data))

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
        """Divide by a scalar.

        Args:
            other: Scalar divisor.

        Returns:
            A new ``Qarray`` with all elements divided by *other*.

        Raises:
            ValueError: If *other* is a ``Qarray``.
        """
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
            eye_data = self._impl._eye_data(self.data.shape[-2], dtype=self.data.dtype)
            other = Qarray.create(
                other * eye_data,
                dims=self.dims,
                implementation=self.impl_type
            )
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
            eye_data = self._impl._eye_data(self.data.shape[-2], dtype=self.data.dtype)
            other = Qarray.create(
                other * eye_data,
                dims=self.dims,
                implementation=self.impl_type
            )
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
        """Build the one-line header string for ``__str__`` and ``__repr__``."""
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
        """One-line header string describing dimensions, shape, and backend."""
        return self._str_header()

    def __repr__(self):
        return self.__str__()

    # Utilities ----
    def copy(self, memo=None):
        """Return a deep copy of this ``Qarray``.

        Args:
            memo: Optional memo dict forwarded to ``deepcopy``.

        Returns:
            A new ``Qarray`` with independent copies of all data.
        """
        return self.__deepcopy__(memo)

    def __deepcopy__(self, memo):
        """Need to override this when defining __getattr__."""

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
        """Conjugate transpose of this array."""
        return dag(self)

    def to_dm(self):
        """Convert a ket to a density matrix via outer product."""
        return ket2dm(self)

    def is_dm(self):
        """Return True if this array is an operator (density-matrix type)."""
        return self.qtype == Qtypes.oper

    def is_vec(self):
        """Return True if this array is a ket or bra."""
        return self.qtype == Qtypes.ket or self.qtype == Qtypes.bra

    def to_ket(self):
        """Convert a bra to a ket (no-op for kets)."""
        return to_ket(self)

    def transpose(self, *args):
        """Transpose subsystem indices."""
        return transpose(self, *args)

    def keep_only_diag_elements(self):
        """Zero out all off-diagonal elements."""
        return keep_only_diag_elements(self)

    # Math Functions ----
    def unit(self):
        """Return the normalised (unit-norm) version of this array."""
        return unit(self)

    def norm(self):
        """Compute the norm of this array."""
        return norm(self)

    def frobenius_norm(self):
        """Compute the Frobenius norm directly from the implementation.

        Returns:
            The Frobenius norm as a scalar.
        """
        return self._impl.frobenius_norm()

    def real(self):
        """Element-wise real part.

        Returns:
            A new ``Qarray`` containing the real parts of each element.
        """
        new_impl = self._impl.real()
        return Qarray.create(
            new_impl.data,
            dims=self.dims,
            implementation=new_impl.impl_type,
        )

    def imag(self):
        """Element-wise imaginary part.

        Returns:
            A new ``Qarray`` containing the imaginary parts of each element.
        """
        new_impl = self._impl.imag()

        return Qarray.create(
            new_impl.data,
            dims=self.dims,
            implementation=new_impl.impl_type,
        )

    def conj(self):
        """Element-wise complex conjugate.

        Returns:
            A new ``Qarray`` containing the complex-conjugated elements.
        """
        new_impl = self._impl.conj()
        return Qarray.create(
            new_impl.data,
            dims=self.dims,
            implementation=new_impl.impl_type,
        )

    def expm(self):
        """Matrix exponential."""
        return expm(self)

    def powm(self, n):
        """Matrix power.

        Args:
            n: Exponent (integer or float).

        Returns:
            This array raised to the *n*-th matrix power.
        """
        return powm(self, n)

    def cosm(self):
        """Matrix cosine."""
        return cosm(self)

    def sinm(self):
        """Matrix sine."""
        return sinm(self)

    def tr(self, **kwargs):
        """Full trace."""
        return tr(self, **kwargs)

    def trace(self, **kwargs):
        """Full trace (alias for :meth:`tr`)."""
        return tr(self, **kwargs)

    def ptrace(self, indx):
        """Partial trace over subsystem *indx*.

        Args:
            indx: Index of the subsystem to trace out.

        Returns:
            Reduced density matrix.
        """
        return ptrace(self, indx)

    def eigenstates(self):
        """Eigenvalues and eigenstates of this operator."""
        return eigenstates(self)

    def eigenenergies(self):
        """Eigenvalues of this operator."""
        return eigenenergies(self)

    def eigenvalues(self):
        """Eigenvalues of this operator (alias for :meth:`eigenenergies`)."""
        return eigenenergies(self)

    def collapse(self, mode="sum"):
        """Collapse batch dimensions.

        Args:
            mode: Collapse strategy — currently only ``"sum"`` is supported.

        Returns:
            A non-batched ``Qarray``.
        """
        return collapse(self, mode=mode)


# Qarray operations ---------------------------------------------------------------------

def concatenate(qarr_list: List[Qarray], axis: int = 0) -> Qarray:
    """Concatenate a list of Qarrays along a specified axis.

    Args:
        qarr_list: List of Qarrays to concatenate.
        axis: Axis along which to concatenate. Default is 0.

    Returns:
        Concatenated Qarray.
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
    """Collapse the batch dimensions of *qarr*.

    Args:
        qarr: Quantum array with optional batch dimensions.
        mode: Collapse strategy.  Only ``"sum"`` is currently supported.

    Returns:
        A non-batched ``Qarray`` obtained by summing over all batch axes.
    """

    if mode == "sum":
        if len(qarr.bdims) == 0:
            return qarr

        batch_axes = list(range(len(qarr.bdims)))

        # Preserve implementation type
        implementation = qarr.impl_type
        return Qarray.create(jnp.sum(qarr.data, axis=batch_axes), dims=qarr.dims, implementation=implementation)


def transpose(qarr: Qarray, indices: List[int]) -> Qarray:
    """Transpose subsystem indices of the quantum array.

    Args:
        qarr: Input quantum array.
        indices: New ordering of subsystem indices.

    Returns:
        Transposed ``Qarray`` (converted to dense first).
    """

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
    """Normalize *qarr* to unit norm.

    Args:
        qarr: Input quantum array.

    Returns:
        Normalized quantum array.
    """
    return qarr / qarr.norm()


def norm(qarr: Qarray) -> float:
    """Compute the norm of a quantum array.

    Sparse paths (no densification):

    * ket / bra — L2 norm via :meth:`SparseImpl.l2_norm_batched` (handles
      batch dimensions).
    * operator — trace norm assuming PSD (nuclear norm = tr(rho) for density
      matrices).  This is exact for density matrices; for general non-PSD
      operators convert to dense first.

    Args:
        qarr: Input quantum array.

    Returns:
        The norm as a scalar (or batched array of scalars).
    """
    if qarr.qtype in [Qtypes.ket, Qtypes.bra] and qarr.is_sparse:
        return qarr._impl.l2_norm_batched(qarr.bdims)

    if qarr.qtype == Qtypes.oper and qarr.is_sparse:
        # Nuclear norm = trace for positive-semidefinite (density matrix) operators.
        # jnp.real strips any floating-point imaginary artefact.
        return jnp.real(qarr._impl.trace())

    qarr = qarr.to_dense()

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
    """Tensor (Kronecker) product of two or more ``Qarray`` objects.

    Args:
        *args: ``Qarray`` objects to tensor together (left to right).
        **kwargs: Optional keyword arguments.  Pass ``parallel=True`` to use
            an einsum-based batched outer product instead of ``jnp.kron``.

    Returns:
        The tensor product as a ``Qarray``.  The output implementation is
        determined by the highest ``PROMOTION_ORDER`` among the inputs: all-sparse
        inputs → sparse output; any dense input → dense output.  This holds for
        both ``parallel=True`` and ``parallel=False``.

    Note:
        ``parallel=True`` uses an einsum-based batched outer product.  The
        einsum is always computed on dense data for efficiency, but the result
        is then wrapped in the appropriate backend (sparse when all inputs are
        sparse, dense otherwise).  For the default (``parallel=False``) path
        each backend's ``kron`` method is used directly.
    """
    parallel = kwargs.pop("parallel", False)

    if parallel:
        # Determine target implementation: highest PROMOTION_ORDER wins.
        # All-sparse → sparse; any dense input → dense (same rule as non-parallel).
        target_impl_type = max(
            (arg.impl_type for arg in args),
            key=lambda t: t.get_impl_class().PROMOTION_ORDER,
        )
        # Einsum-based batched outer product (computed on dense data).
        dense_args = [arg.to_dense() for arg in args]
        data = dense_args[0].data
        dims_0 = dense_args[0].dims[0]
        dims_1 = dense_args[0].dims[1]
        for arg in dense_args[1:]:
            a, b = data, arg.data
            if len(a.shape) > len(b.shape):
                batch_dim = a.shape[:-2]
            elif len(a.shape) == len(b.shape):
                batch_dim = a.shape[:-2] if prod(a.shape[:-2]) > prod(b.shape[:-2]) else b.shape[:-2]
            else:
                batch_dim = b.shape[:-2]

            # NOTE: implementation einsum should be used when available
            data = jnp.einsum("...ij,...kl->...ikjl", a, b).reshape(
                *batch_dim, a.shape[-2] * b.shape[-2], -1
            )
            dims_0 = dims_0 + arg.dims[0]
            dims_1 = dims_1 + arg.dims[1]
        return Qarray.create(data, dims=(dims_0, dims_1), implementation=target_impl_type)

    # Non-parallel: delegate to each impl's kron method.
    # All-sparse inputs stay sparse; mixed inputs promote to dense via _coerce.
    current_impl = args[0]._impl
    dims_0 = args[0].dims[0]
    dims_1 = args[0].dims[1]
    for arg in args[1:]:
        current_impl = current_impl.kron(arg._impl)
        dims_0 = dims_0 + arg.dims[0]
        dims_1 = dims_1 + arg.dims[1]
    return Qarray.create(current_impl.data, dims=(dims_0, dims_1),
                         implementation=current_impl.impl_type)


def tr(qarr: Qarray, **kwargs) -> Array:
    """Full trace of *qarr*.

    For sparse ``Qarray`` objects the trace is computed natively on the BCOO
    data using a masked scatter — no densification.  Custom axis arguments
    are ignored for sparse (the last two dimensions are always the matrix
    dimensions in jaxquantum's convention).

    Args:
        qarr: Input quantum array.
        **kwargs: Forwarded to ``jnp.trace`` for dense arrays (e.g.
            ``axis1``, ``axis2``).

    Returns:
        The trace as a scalar (or batched array of scalars).
    """
    if qarr.is_sparse:
        return qarr._impl.trace()
    axis1 = kwargs.get("axis1", -2)
    axis2 = kwargs.get("axis2", -1)
    return jnp.trace(qarr.data, axis1=axis1, axis2=axis2, **kwargs)


def trace(qarr: Qarray, **kwargs) -> Array:
    """Full trace (alias for :func:`tr`).

    Args:
        qarr: Input quantum array.
        **kwargs: Forwarded to :func:`tr`.

    Returns:
        The trace as a scalar (or batched array of scalars).
    """
    return tr(qarr, **kwargs)


def expm_data(data: Array, **kwargs) -> Array:
    """Matrix exponential of a raw array.

    Args:
        data: Dense matrix array.
        **kwargs: Forwarded to ``jsp.linalg.expm``.

    Returns:
        The matrix exponential.
    """
    return jsp.linalg.expm(data, **kwargs)


def expm(qarr: Qarray, **kwargs) -> Qarray:
    """Matrix exponential of a ``Qarray``.

    Args:
        qarr: Input quantum array (converted to dense internally).
        **kwargs: Forwarded to ``jsp.linalg.expm``.

    Returns:
        A dense ``Qarray`` containing the matrix exponential.
    """
    dims = qarr.dims
    # Convert to dense for expm
    dense_data = qarr.to_dense().data
    data = expm_data(dense_data, **kwargs)
    return Qarray.create(data, dims=dims)


def powm(qarr: Qarray, n: Union[int, float], clip_eigvals=False) -> Qarray:
    """Matrix power of a ``Qarray``.

    Args:
        qarr: Input quantum array.
        n: Exponent.  Integer powers use ``jnp.linalg.matrix_power``; float
            powers diagonalise the matrix.
        clip_eigvals: When ``True``, clip negative eigenvalues to zero before
            applying the float power (useful for nearly-PSD matrices).

    Returns:
        The *n*-th matrix power as a dense ``Qarray``.

    Raises:
        ValueError: If *n* is a float and the matrix has negative eigenvalues
            (and *clip_eigvals* is ``False``).
    """
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
    """Matrix cosine of a raw array.

    Args:
        data: Dense matrix array.
        **kwargs: Unused; kept for API consistency.

    Returns:
        The matrix cosine computed as ``(expm(i*A) + expm(-i*A)) / 2``.
    """
    return (expm_data(1j * data) + expm_data(-1j * data)) / 2


def cosm(qarr: Qarray) -> Qarray:
    """Matrix cosine of a ``Qarray``.

    Args:
        qarr: Input quantum array (converted to dense internally).

    Returns:
        A dense ``Qarray`` containing the matrix cosine.
    """
    dims = qarr.dims
    # Convert to dense for cosm
    dense_data = qarr.to_dense().data
    data = cosm_data(dense_data)
    return Qarray.create(data, dims=dims)


def sinm_data(data: Array, **kwargs) -> Array:
    """Matrix sine of a raw array.

    Args:
        data: Dense matrix array.
        **kwargs: Unused; kept for API consistency.

    Returns:
        The matrix sine computed as ``(expm(i*A) - expm(-i*A)) / (2i)``.
    """
    return (expm_data(1j * data) - expm_data(-1j * data)) / (2j)


def sinm(qarr: Qarray) -> Qarray:
    """Matrix sine of a ``Qarray``.

    Args:
        qarr: Input quantum array (converted to dense internally).

    Returns:
        A dense ``Qarray`` containing the matrix sine.
    """
    dims = qarr.dims
    # Convert to dense for sinm
    dense_data = qarr.to_dense().data
    data = sinm_data(dense_data)
    return Qarray.create(data, dims=dims)


def keep_only_diag_elements(qarr: Qarray) -> Qarray:
    """Zero out all off-diagonal elements of *qarr*.

    For sparse ``Qarray`` objects the off-diagonal stored values are zeroed
    in-place on the BCOO structure — no densification.

    Args:
        qarr: Non-batched input quantum array.

    Returns:
        A ``Qarray`` with only diagonal entries non-zero.

    Raises:
        ValueError: If *qarr* has batch dimensions.
    """
    if len(qarr.bdims) > 0:
        raise ValueError("Cannot keep only diagonal elements of a batched Qarray.")

    dims = qarr.dims
    if qarr.is_sparse:
        new_impl = qarr._impl.keep_only_diag()
        return Qarray.create(new_impl.data, dims=dims, implementation=QarrayImplType.SPARSE)
    data = jnp.diag(jnp.diag(qarr.data))
    return Qarray.create(data, dims=dims)


def to_ket(qarr: Qarray) -> Qarray:
    """Convert *qarr* to a ket.

    Args:
        qarr: A ket (returned as-is) or bra (conjugate-transposed).

    Returns:
        The ket form of *qarr*.

    Raises:
        ValueError: If *qarr* is an operator.
    """
    if qarr.qtype == Qtypes.ket:
        return qarr
    elif qarr.qtype == Qtypes.bra:
        return qarr.dag()
    else:
        raise ValueError("Can only get ket from a ket or bra.")


def eigenstates(qarr: Qarray) -> Qarray:
    """Eigenstates of a quantum array.

    Args:
        qarr: Hermitian operator (converted to dense internally).

    Returns:
        A tuple ``(eigenvalues, eigenstates_qarray)`` where eigenvalues are
        sorted in ascending order.
    """
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
    """Eigenvalues of a quantum array.

    Args:
        qarr: Hermitian operator (converted to dense internally).

    Returns:
        Sorted eigenvalues as a JAX array.
    """
    # Convert to dense for eigenenergies
    dense_qarr = qarr.to_dense()
    evals = jnp.linalg.eigvalsh(dense_qarr.data)
    return evals


def ptrace(qarr: Qarray, indx) -> Qarray:
    """Partial trace over subsystem *indx*.

    Args:
        qarr: Input quantum array (converted to dense internally).
        indx: Index of the subsystem to trace out.

    Returns:
        Reduced density matrix as a ``Qarray``.
    """
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
    """Conjugate transpose of *qarr*.

    Args:
        qarr: Input quantum array.

    Returns:
        The conjugate transpose with swapped ``dims``.
    """
    dims = qarr.dims[::-1]
    new_impl = qarr._impl.dag()
    return Qarray.create(
        new_impl.data,
        dims=dims,
        implementation=new_impl.impl_type,
    )


def dag_data(arr) -> Array:
    """Conjugate transpose of a raw array, dispatching to the right backend.

    Iterates through registered :class:`QarrayImpl` subclasses and delegates
    to the first one whose :meth:`~QarrayImpl.can_handle_data` returns True.
    Adding a new backend automatically extends this function — no changes
    required here.

    Args:
        arr: Input array (``jnp.ndarray``, ``sparse.BCOO``, or any type
            handled by a registered impl).  For 1-D dense arrays only
            conjugation is applied (no transpose).

    Returns:
        Conjugate transpose with the last two axes swapped.

    Raises:
        TypeError: If no registered impl can handle *arr*.
    """
    for impl_class in _IMPL_REGISTRY:
        if impl_class.can_handle_data(arr):
            return impl_class.dag_data(arr)
    raise TypeError(f"dag_data: no registered impl can handle type {type(arr)}")


def ket2dm(qarr: Qarray) -> Qarray:
    """Convert a ket to a density matrix via outer product.

    Args:
        qarr: Ket, bra, or operator.  Operators are returned unchanged.

    Returns:
        Density matrix ``|ψ⟩⟨ψ|``.
    """
    if qarr.qtype == Qtypes.oper:
        return qarr

    if qarr.qtype == Qtypes.bra:
        qarr = qarr.dag()

    return qarr @ qarr.dag()


# Data level operations
def is_dm_data(data: Array) -> bool:
    """Check whether *data* has the shape of a density matrix (square matrix).

    Args:
        data: Array to check.

    Returns:
        True if the last two dimensions are equal.
    """
    return data.shape[-2] == data.shape[-1]


def powm_data(data: Array, n: int) -> Array:
    """Integer matrix power of a raw array.

    Args:
        data: Dense square matrix array.
        n: Integer exponent.

    Returns:
        The *n*-th matrix power.
    """
    return jnp.linalg.matrix_power(data, n)


# Type aliases for readability
DenseQarray = Qarray[DenseImpl]
SparseQarray = Qarray[SparseImpl]

ARRAY_TYPES = (Array, ndarray, Qarray)

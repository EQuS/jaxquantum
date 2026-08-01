"""New Qarray implementation with sparse support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from flax import struct
from jax import Array, config, vmap
from typing import TYPE_CHECKING, List, Union, TypeVar, Generic, overload, Literal

if TYPE_CHECKING:
    from jaxquantum.core.sparse_bcoo import SparseBCOOImpl
import jax.numpy as jnp
import jax.scipy as jsp
from jax.experimental import sparse
from numpy import ndarray
from copy import deepcopy
from math import prod
from enum import Enum

from jaxquantum.core.settings import SETTINGS, _maybe_shard
from jaxquantum.utils.utils import robust_isscalar
from jaxquantum.core.dims import (
    Qtypes,
    Qdims,
    check_dims,
    ket_from_op_dims,
)

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
        SPARSE_BCOO: JAX experimental BCOO sparse array.
        SPARSE_DIA: Diagonal sparse array.
    """

    DENSE = "dense"
    SPARSE_BCOO = "sparse_bcoo"
    SPARSE_DIA = "sparse_dia"

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
        (e.g. ``DenseImpl``, ``SparseBCOOImpl``) that has been registered.

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
        except Exception:  # noqa: BLE001
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
    """Convert *data* to a JAX array, leaving sparse BCOO and SparseDiaData untouched.

    Args:
        data: Input data — any array-like, ``sparse.BCOO``, or ``SparseDiaData``.

    Returns:
        A ``jax.Array``, ``sparse.BCOO``, or ``SparseDiaData``.
    """
    if isinstance(data, sparse.BCOO):
        return data
    # SparseDiaData has a ``_is_sparse_dia`` marker; pass it through unchanged
    if getattr(data, "_is_sparse_dia", False):
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
            "more general" (``DenseImpl = 1``, ``SparseBCOOImpl = 0``).
    """

    PROMOTION_ORDER: int = 0  # override in subclasses; higher = more general
    # Current hierarchy: SparseDiaImpl=0, SparseBCOOImpl=1, DenseImpl=2

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
    def to_sparse_bcoo(self) -> "SparseBCOOImpl":
        """Convert to a ``SparseBCOOImpl`` (BCOO).

        Returns:
            A ``SparseBCOOImpl`` wrapping the same data.
        """
        pass

    def to_sparse_dia(self) -> "QarrayImpl":
        """Convert to a ``SparseDiaImpl``.

        Default implementation goes through dense and auto-detects diagonals.
        Subclasses may override for a more direct path.

        Returns:
            A ``SparseDiaImpl`` wrapping the same data.
        """
        # Import here to avoid circular imports at module load time
        from jaxquantum.core.sparse_dia import SparseDiaImpl
        return SparseDiaImpl.from_data(self.to_dense()._data)

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
                ``sparse.BCOO`` for ``SparseBCOOImpl``.

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

    PROMOTION_ORDER = 2  # noqa: RUF012 — not a struct field; no annotation intentional

    @classmethod
    def _make(cls, data) -> "DenseImpl":
        """Construct a ``DenseImpl``, applying the configured default sharding.

        All internal construction sites route through this so that every
        Qarray (including intermediates produced by ``matmul``, ``kron``,
        etc.) satisfies the user's sharding invariant.
        """
        return cls(_data=_maybe_shard(data))

    @classmethod
    def from_data(cls, data) -> "DenseImpl":
        """Wrap *data* in a new ``DenseImpl``.

        Args:
            data: Array-like input data.

        Returns:
            A ``DenseImpl`` wrapping ``robust_asarray(data)``.
        """
        return cls._make(robust_asarray(data))

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
        return DenseImpl._make(self._data @ b._data)

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
        return DenseImpl._make(self._data + b._data)

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
        return DenseImpl._make(self._data - b._data)

    def mul(self, scalar) -> QarrayImpl:
        """Scalar multiplication.

        Args:
            scalar: Scalar value.

        Returns:
            A ``DenseImpl`` with each element multiplied by *scalar*.
        """
        return DenseImpl._make(scalar * self._data)

    def dag(self) -> QarrayImpl:
        """Conjugate transpose.

        Returns:
            A ``DenseImpl`` containing the conjugate transpose.
        """
        return DenseImpl._make(jnp.moveaxis(jnp.conj(self._data), -1, -2))

    def to_dense(self) -> "DenseImpl":
        """Return self (already dense).

        Returns:
            This ``DenseImpl`` instance unchanged.
        """
        return self

    def to_sparse_bcoo(self) -> "SparseBCOOImpl":
        """Convert to a ``SparseBCOOImpl`` via ``BCOO.fromdense``.

        Returns:
            A ``SparseBCOOImpl`` wrapping a BCOO conversion of this array.
        """
        from jaxquantum.core.sparse_bcoo import SparseBCOOImpl
        return SparseBCOOImpl(sparse.BCOO.fromdense(self._data))

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
        return DenseImpl._make(jnp.real(self._data))

    def imag(self) -> QarrayImpl:
        """Element-wise imaginary part.

        Returns:
            A ``DenseImpl`` containing the imaginary parts.
        """
        return DenseImpl._make(jnp.imag(self._data))

    def conj(self) -> QarrayImpl:
        """Element-wise complex conjugate.

        Returns:
            A ``DenseImpl`` containing the complex-conjugated values.
        """
        return DenseImpl._make(jnp.conj(self._data))

    def __deepcopy__(self, memo=None):
        return DenseImpl._make(deepcopy(self._data, memo))

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

        return DenseImpl._make(data_new)

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
        return DenseImpl._make(jnp.kron(self._data, b._data))

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
        """Return True for any non-BCOO, non-SparseDIA array.

        ``SparseDiaData`` objects carry a ``_is_sparse_dia`` marker so we can
        exclude them without a direct type import (which would be circular).

        Args:
            arr: Raw array.

        Returns:
            True when *arr* is a plain dense array (not BCOO, not SparseDiaData).
        """
        return not isinstance(arr, sparse.BCOO) and not getattr(arr, "_is_sparse_dia", False)

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


# Register implementation classes with the enum registry
# SparseBCOOImpl is registered in sparse_bcoo.py after import
QarrayImplType.register(DenseImpl, QarrayImplType.DENSE)


@struct.dataclass
class Qarray(Generic[ImplT]):
    """Quantum array with a pluggable storage backend.

    ``Qarray`` wraps a ``QarrayImpl`` together with quantum-mechanical
    dimension metadata (``_qdims``) and optional batch dimensions
    (``_bdims``).  The default backend is dense (``DenseImpl``); pass
    ``implementation="sparse_bcoo"`` (or ``QarrayImplType.SPARSE_BCOO``) to
    store data as a JAX BCOO sparse array.

    Attributes:
        _impl: The storage backend holding the raw data.
        _qdims: Quantum dimension metadata (bra/ket structure, Hilbert space
            sizes).
        _bdims: Tuple of batch dimension sizes (empty tuple = non-batched).

    Example:
        >>> import jaxquantum as jqt
        >>> a = jqt.destroy(10, implementation="sparse_bcoo")
        >>> a.is_sparse_bcoo
        True
    """

    _impl: ImplT
    _qdims: Qdims = struct.field(pytree_node=False)
    _bdims: tuple[int] = struct.field(pytree_node=False)

    # Initialization ----
    @classmethod
    def _from_impl(cls, impl, qdims, bdims=None):
        """Wrap a canonical internal result without repeating public validation."""
        if bdims is None:
            shape = impl.shape()
            bdims = shape[:-2] if qdims.qtype == Qtypes.oper else shape[:-1]
        return cls(impl, qdims, tuple(bdims))

    @classmethod
    @overload
    def create(cls, data, dims=None, bdims=None, qtype=None, implementation: Literal[QarrayImplType.DENSE] = QarrayImplType.DENSE) -> "Qarray[DenseImpl]":
        ...

    @classmethod
    @overload
    def create(cls, data, dims=None, bdims=None, qtype=None, implementation: Literal[QarrayImplType.SPARSE_BCOO] = ...) -> "Qarray[SparseBCOOImpl]":
        ...

    @classmethod
    @overload
    def create(cls, data, dims=None, bdims=None, qtype=None, implementation=...) -> "Qarray[DenseImpl]":
        ...

    @classmethod
    def create(cls, data, dims=None, bdims=None, qtype=None, implementation=QarrayImplType.DENSE):
        """Create a ``Qarray`` from raw data.

        Handles shape normalisation, dimension inference, and tidying of small
        values.

        State vectors are stored with their Hilbert space on a **single trailing
        axis** — a ket/bra of dimension ``N`` has data shape ``bdims + (N,)``
        (never ``(N,1)`` / ``(1,N)``).  Operators keep the last two axes:
        ``bdims + (M, N)``.  The ket/bra/oper distinction lives in ``_qdims``,
        not in the data shape.

        Legacy ``(N,1)`` / ``(1,N)`` inputs are still accepted and are squeezed
        to ``(N,)`` on the way in.  Because a 1‑D ``(N,)`` array (or a square
        ``(N,N)`` batch of vectors) is shape-ambiguous, pass ``qtype`` to be
        explicit.

        Args:
            data: Input data array (dense array-like or ``sparse.BCOO``).
            dims: Quantum dimensions as ``((row_dims...), (col_dims...))`` or, as
                a convenience, a flat list of Hilbert space sizes.  Inferred from
                *data* shape when ``None``.
            bdims: Tuple of batch dimension sizes.  Inferred from the leading
                dimensions of *data* when ``None``.
            qtype: Optional quantum type — ``"ket"`` / ``"bra"`` / ``"oper"`` (or
                a :class:`Qtypes` member).  When given it is authoritative and
                resolves shape ambiguity (e.g. a bare ``(N,)`` array with
                ``qtype="ket"`` behaves like a legacy ``(N,1)`` input).
            implementation: Storage backend — ``QarrayImplType.DENSE``
                (default) or ``QarrayImplType.SPARSE_BCOO``, or the equivalent
                string ``"dense"`` / ``"sparse_bcoo"``.

        Returns:
            A new ``Qarray`` backed by the requested implementation.
        """
        data = robust_asarray(data)

        # Whether the caller explicitly requested a qtype (vs. it being derived
        # from a full ``dims`` tuple). An explicit qtype is authoritative and
        # mismatches with the data raise a clear error.
        qtype_explicit = qtype is not None
        if qtype is not None and not isinstance(qtype, Qtypes):
            # Accepts "ket" / "bra" / "oper" (case-sensitive enum value).
            qtype = Qtypes.from_str(qtype)

        # Empty array (e.g. an empty list of operators) — keep legacy behaviour.
        if len(data.shape) == 1 and data.shape[0] == 0:
            dims = ((), ()) if dims is None else (tuple(dims[0]), tuple(dims[1]))
            bdims = (0,) if bdims is None else tuple(bdims)
            check_dims(dims, bdims, data.shape)
            qdims = Qdims(dims)
            impl_class = QarrayImplType(implementation).get_impl_class()
            impl = impl_class.from_data(data)
            impl = impl.tidy_up(SETTINGS["auto_tidyup_atol"])
            return cls(impl, qdims, bdims)

        # Resolve full quantum dims + batch dims.
        dims_is_full = dims is not None and isinstance(dims[0], (list, tuple))
        if qtype is None and dims_is_full:
            qtype = Qtypes.from_dims((tuple(dims[0]), tuple(dims[1])))

        if qtype is not None:
            # Explicit (or dims-derived) qtype: unambiguous layout.
            data, dims, bdims = cls._qtype_layout(data, dims, bdims, qtype)
        else:
            # Auto: legacy shape heuristics (1‑D → ket, square → oper, etc.).
            data, dims, bdims = cls._legacy_layout(data, dims, bdims)

        qdims = Qdims(dims)

        # Store vectors as bdims + (N,); operators as bdims + (M, N).
        if qdims.qtype == Qtypes.oper:
            space = (prod(dims[0]), prod(dims[1]))
        else:
            space = (prod(dims[0]) * prod(dims[1]),)

        target_shape = tuple(bdims) + space

        # An explicitly requested qtype is authoritative: error out if the data
        # cannot be interpreted as that type rather than silently coercing it.
        if qtype_explicit:
            if qdims.qtype != qtype:
                raise ValueError(
                    f"Requested qtype={qtype} is incompatible with the resolved "
                    f"dims {dims} (which is a {qdims.qtype})."
                )
            if prod(data.shape) != prod(target_shape):
                raise ValueError(
                    f"Data of shape {tuple(data.shape)} is incompatible with "
                    f"qtype={qtype} and dims={dims}: cannot reshape to "
                    f"{target_shape}."
                )

        # Reshape to the canonical stored shape only when needed. Use varargs so
        # this works for dense arrays, BCOO, and SparseDiaData alike.
        if tuple(data.shape) != target_shape:
            data = data.reshape(*target_shape)

        check_dims(dims, bdims, data.shape)

        # NOTE: Constantly tidying up on Qarray creation might be a bit overkill.
        # It increases the compilation time, but only very slightly
        # increased the runtime of the jit compiled function.
        # We could instead use this tidy up where we think we need it.

        impl_class = QarrayImplType(implementation).get_impl_class()
        impl = impl_class.from_data(data)
        impl = impl.tidy_up(SETTINGS["auto_tidyup_atol"])

        return cls(impl, qdims, bdims)

    @staticmethod
    def _legacy_layout(data, dims, bdims):
        """Auto-resolve dims/bdims from data shape (no explicit ``qtype``).

        Mirrors the historical shape heuristics: a 1‑D array is a ket, a square
        2‑D array is an operator, a non-square 2‑D array is a batch of kets, and
        legacy ``(N,1)`` / ``(1,N)`` carry the orientation.  Returns
        ``(data, dims, bdims)`` where *data* may still carry a trailing singleton
        for vectors — the caller collapses it to ``(N,)``.
        """
        if len(data.shape) == 1 and data.shape[0] > 0:
            data = data.reshape(data.shape[0], 1)

        if (
            len(data.shape) >= 2
            and data.shape[-2] != data.shape[-1]
            and not (data.shape[-2] == 1 or data.shape[-1] == 1)
        ):
            data = data.reshape(*data.shape[:-1], data.shape[-1], 1)

        if bdims is not None and len(data.shape) - len(bdims) == 1:
            data = data.reshape(*data.shape[:-1], data.shape[-1], 1)

        if bdims is None:
            bdims = tuple(data.shape[:-2])

        if dims is None:
            dims = ((data.shape[-2],), (data.shape[-1],))

        if not isinstance(dims[0], (list, tuple)):
            # Only the Hilbert space dimensions were sent in.
            if data.shape[-1] == 1:
                dims = (tuple(dims), tuple([1 for _ in dims]))
            elif data.shape[-2] == 1:
                dims = (tuple([1 for _ in dims]), tuple(dims))
            else:
                dims = (tuple(dims), tuple(dims))
        else:
            dims = (tuple(dims[0]), tuple(dims[1]))

        return data, dims, tuple(bdims)

    @staticmethod
    def _qtype_layout(data, dims, bdims, qtype):
        """Resolve dims/bdims given an explicit ``qtype`` (no shape guessing).

        Accepts modern ``(...,N)`` vectors as well as legacy ``(N,1)`` / ``(1,N)``
        (and batched variants) and returns ``(data, dims_full, bdims)``.  The
        caller reshapes *data* to the canonical stored shape.
        """
        shape = tuple(data.shape)
        dims_is_full = dims is not None and isinstance(dims[0], (list, tuple))

        if dims_is_full:
            dims_full = (tuple(dims[0]), tuple(dims[1]))
        elif dims is not None:
            sl = tuple(dims)
            if qtype == Qtypes.ket:
                dims_full = (sl, tuple(1 for _ in sl))
            elif qtype == Qtypes.bra:
                dims_full = (tuple(1 for _ in sl), sl)
            else:
                dims_full = (sl, sl)
        else:
            dims_full = None  # inferred from shape below

        if qtype == Qtypes.oper:
            if dims_full is None:
                if len(shape) < 2:
                    raise ValueError(
                        f"qtype='oper' needs 2-D data or explicit dims; "
                        f"got shape {shape}."
                    )
                dims_full = ((shape[-2],), (shape[-1],))
            if bdims is None:
                bdims = shape[:-2]
            return data, dims_full, tuple(bdims)

        # Vector (ket / bra).
        if bdims is not None:
            bdims = tuple(bdims)
            if dims_full is None:
                n = 1
                for d in shape[len(bdims):]:
                    n *= d
                dims_full = ((n,), (1,)) if qtype == Qtypes.ket else ((1,), (n,))
            return data, dims_full, bdims

        # bdims unknown: peel a legacy orientation singleton, then the last axis
        # is the space axis and the leading axes are batch dims.
        s = list(shape)
        if qtype == Qtypes.ket and len(s) >= 2 and s[-1] == 1:
            s = s[:-1]
        elif qtype == Qtypes.bra and len(s) >= 2 and s[-2] == 1:
            s = s[:-2] + s[-1:]
        inferred_n = s[-1] if s else 1
        bdims = tuple(s[:-1])
        if dims_full is None:
            dims_full = ((inferred_n,), (1,)) if qtype == Qtypes.ket else ((1,), (inferred_n,))
        return data, dims_full, bdims

    @classmethod
    @overload
    def from_sparse_bcoo(cls, data, dims=None, bdims=None) -> "Qarray[SparseBCOOImpl]":
        ...

    @classmethod
    def from_sparse_bcoo(cls, data, dims=None, bdims=None):
        """Create a ``Qarray`` directly from a sparse BCOO array without densifying.

        Args:
            data: A ``sparse.BCOO`` or array-like to store as sparse BCOO.
            dims: Quantum dimensions.  Inferred when ``None``.
            bdims: Batch dimensions.  Inferred when ``None``.

        Returns:
            A ``Qarray[SparseBCOOImpl]``.
        """
        return cls.create(data, dims=dims, bdims=bdims, implementation=QarrayImplType.SPARSE_BCOO)

    @classmethod
    def from_sparse_dia(cls, data, dims=None, bdims=None) -> "Qarray":
        """Create a SparseDIA-backed ``Qarray``.

        Accepts either a dense array-like (diagonals are auto-detected) or a
        :class:`~jaxquantum.core.sparse_dia.SparseDiaData` container.

        Args:
            data: Dense array of shape (*batch, n, n) or a ``SparseDiaData``.
            dims: Quantum dimensions ``((row_dims,), (col_dims,))``.
            bdims: Batch dimension sizes.

        Returns:
            A ``Qarray`` backed by ``SparseDiaImpl``.
        """
        return cls.create(data, dims=dims, bdims=bdims, implementation=QarrayImplType.SPARSE_DIA)

    @classmethod
    @overload
    def from_list(cls, qarr_list: List["Qarray[DenseImpl]"], qtype=None) -> "Qarray[DenseImpl]":
        ...

    @classmethod
    @overload
    def from_list(cls, qarr_list: List["Qarray[SparseBCOOImpl]"], qtype=None) -> "Qarray[SparseBCOOImpl]":
        ...

    @classmethod
    def from_list(cls, qarr_list: List[Qarray], qtype=None) -> Qarray:
        """Create a batched ``Qarray`` from a list of same-shaped ``Qarray`` objects.

        The output implementation is determined by the element with the highest
        ``PROMOTION_ORDER``: if all inputs are sparse the result is sparse; if
        any input is dense (or types are mixed) all inputs are promoted to dense
        and the result is dense.

        Works for kets/bras (stacked into ``(len, *bdims, N)``) as well as
        operators, regardless of whether the elements were originally created
        from ``(N,)`` or legacy ``(N,1)`` / ``(1,N)`` arrays — they are all
        stored as ``(N,)`` vectors by the time they reach here.

        Args:
            qarr_list: List of ``Qarray`` objects with identical ``dims`` and
                ``bdims``.  May be empty.
            qtype: Optional quantum type ("ket"/"bra"/"oper" or a ``Qtypes``)
                forwarded to :meth:`create`.  Defaults to the qtype of the
                first element, which is the correct choice in all normal cases;
                pass it explicitly to override or to be defensive.

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

        if qtype is None:
            qtype = qarr_list[0].qtype

        new_bdims = (len(qarr_list),) + bdims

        # Pick the target type: highest PROMOTION_ORDER wins (dense beats sparse).
        target_impl_type = max(
            (q.impl_type for q in qarr_list),
            key=lambda t: t.get_impl_class().PROMOTION_ORDER,
        )

        if target_impl_type == QarrayImplType.SPARSE_DIA:
            # All inputs are SparseDIA — batch without densifying.
            # Compute union of offsets across all operators, then remap each
            # operator's _diags rows into the union shape and stack.
            from jaxquantum.core.sparse_dia import SparseDiaData  # lazy to avoid circular
            union_offsets = tuple(sorted(
                set().union(*[set(q._impl._offsets) for q in qarr_list])
            ))
            union_idx = {k: i for i, k in enumerate(union_offsets)}
            n = qarr_list[0]._impl._diags.shape[-1]
            dtype = jnp.result_type(*[q._impl._diags.dtype for q in qarr_list])
            remapped = []
            for q in qarr_list:
                row = jnp.zeros((len(union_offsets), n), dtype=dtype)
                for i_src, k in enumerate(q._impl._offsets):
                    row = row.at[union_idx[k], :].set(q._impl._diags[i_src, :])
                remapped.append(row)
            stacked = jnp.stack(remapped, axis=0)  # (n_ops, n_union_diags, N)
            raw = SparseDiaData(offsets=union_offsets, diags=stacked)
            return cls.create(raw, dims=dims, bdims=new_bdims, qtype=qtype, implementation=QarrayImplType.SPARSE_DIA)

        if target_impl_type == QarrayImplType.SPARSE_BCOO:
            # All inputs are sparse BCOO — stack via dense intermediates then re-sparsify.
            data = jnp.array([q.data.todense() for q in qarr_list])
            return cls.create(data, dims=dims, bdims=new_bdims, qtype=qtype, implementation=QarrayImplType.SPARSE_BCOO)

        # Target is dense: promote any sparse inputs before stacking.
        data = jnp.array([q.to_dense().data for q in qarr_list])
        return cls.create(data, dims=dims, bdims=new_bdims, qtype=qtype, implementation=QarrayImplType.DENSE)

    @classmethod
    @overload
    def from_array(cls, qarr_arr: "Qarray[DenseImpl]") -> "Qarray[DenseImpl]":
        ...

    @classmethod
    @overload
    def from_array(cls, qarr_arr: "Qarray[SparseBCOOImpl]") -> "Qarray[SparseBCOOImpl]":
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
    def is_sparse_bcoo(self):
        """True if the storage backend is ``SparseBCOOImpl`` (BCOO)."""
        return self._impl.impl_type == QarrayImplType.SPARSE_BCOO

    @property
    def is_dense(self):
        """True if the storage backend is ``DenseImpl``."""
        return self._impl.impl_type == QarrayImplType.DENSE

    @property
    def is_sparse_dia(self):
        """True if the storage backend is ``SparseDiaImpl``."""
        return self._impl.impl_type == QarrayImplType.SPARSE_DIA

    @property
    def impl_type(self):
        """The ``QarrayImplType`` member of the current storage backend."""
        return self._impl.impl_type

    def to_sparse_bcoo(self) -> "Qarray[SparseBCOOImpl]":
        """Return a BCOO-sparse-backed copy of this array.

        If the array is already sparse BCOO, returns self unchanged.

        Returns:
            A ``Qarray[SparseBCOOImpl]``.
        """
        if self.is_sparse_bcoo:
            return self
        new_impl = self._impl.to_sparse_bcoo()
        return Qarray(new_impl, self._qdims, self._bdims)

    def to_sparse_dia(self) -> "Qarray":
        """Return a SparseDIA-backed copy of this array.

        If the array is already SparseDIA, returns self unchanged.

        Returns:
            A ``Qarray[SparseDiaImpl]``.
        """
        if self.is_sparse_dia:
            return self
        new_impl = self._impl.to_sparse_dia()
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
            impl = type(self._impl).from_data(self.data[index])
            return Qarray._from_impl(impl, self._qdims)
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
        elif self.qtype == Qtypes.oper:
            new_shape = new_bdims + (prod(self.dims[0]), -1)
        else:
            # Vectors keep a single trailing space axis (no (N,1)).
            new_shape = new_bdims + (prod(self.dims[0]) * prod(self.dims[1]),)

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
            raise ValueError(  # noqa: TRY004
                "Cannot calculate equality of a Qarray with a non-Qarray."
            )

        if self.dims != other.dims:
            return False

        if self.bdims != other.bdims:
            return False

        if self.is_sparse_bcoo and other.is_sparse_bcoo:
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
        st, ot = self.qtype, other.qtype

        # Operator @ operator: keep the backend-native path (dense or sparse),
        # which preserves the storage implementation. Operators are 2-D-tailed,
        # so create() infers bdims/qtype from the result shape unambiguously.
        if st == Qtypes.oper and ot == Qtypes.oper:
            new_impl = self._impl.matmul(other._impl)
            return Qarray._from_impl(new_impl, _qdims_new)

        # bra @ oper  ==  dag( oper^dag @ bra^dag ).  Reusing the oper @ ket path
        # keeps a sparse operator sparse (no densification of the operator).
        if st == Qtypes.bra and ot == Qtypes.oper:
            return (other.dag() @ self.dag()).dag()

        # oper @ ket: matrix-vector product. Vectors are dense (see migration
        # scope); the operator may be sparse and acts natively on the dense
        # vector. The vector's single space axis is contracted in place — it is
        # never reshaped to (N,1) for storage.
        if st == Qtypes.oper and ot == Qtypes.ket:
            vec = other.to_dense().data  # (..., N)
            if self.is_dense:
                out = jnp.einsum("...ij,...j->...i", self.data, vec)
            else:
                # Transient column only inside the sparse kernel; never stored.
                rhs = vec[..., None]  # (..., N, 1)
                out = self._impl.matmul(DenseImpl._make(rhs)).data[..., 0]
            return Qarray._from_impl(DenseImpl._make(out), _qdims_new)

        # Vector ⊗/· vector — both operands are dense.
        a = self.to_dense().data
        b = other.to_dense().data
        if st == Qtypes.ket and ot == Qtypes.bra:
            # Outer product |a><b| → operator.
            out = jnp.einsum("...i,...j->...ij", a, b)
            return Qarray._from_impl(DenseImpl._make(out), _qdims_new)
        if st == Qtypes.bra and ot == Qtypes.ket:
            # Inner product <a|b> → 1x1 "ket" (qdims ((1,),(1,))).
            out = jnp.einsum("...i,...i->...", a, b)
            out = out.reshape(out.shape + (1,))
            return Qarray._from_impl(DenseImpl._make(out), _qdims_new)

        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Qarray):
            return self.__matmul__(other)

        other = other + 0.0j
        if not robust_isscalar(other) and len(other.shape) > 0:  # not a scalar
            # Broadcast per-batch scalars against the stored data: vectors carry
            # a single trailing space axis, operators carry two.
            extra = (1,) if self.qtype in (Qtypes.ket, Qtypes.bra) else (1, 1)
            other = other.reshape(other.shape + extra)

        new_impl = self._impl.mul(other)
        return Qarray._from_impl(new_impl, self._qdims)

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
            raise ValueError("Cannot divide a Qarray by another Qarray.")  # noqa: TRY004

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
            return Qarray._from_impl(new_impl, self._qdims)

        if robust_isscalar(other) and other == 0:
            return self.copy()

        if self.qtype == Qtypes.oper:
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
            return Qarray._from_impl(new_impl, self._qdims)

        if robust_isscalar(other) and other == 0:
            return self.copy()

        if self.qtype == Qtypes.oper:
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
            if self.is_sparse_bcoo:
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
        return Qarray._from_impl(new_impl, self._qdims)

    def imag(self):
        """Element-wise imaginary part.

        Returns:
            A new ``Qarray`` containing the imaginary parts of each element.
        """
        new_impl = self._impl.imag()
        return Qarray._from_impl(new_impl, self._qdims)

    def conj(self):
        """Element-wise complex conjugate.

        Returns:
            A new ``Qarray`` containing the complex-conjugated elements.
        """
        new_impl = self._impl.conj()
        return Qarray._from_impl(new_impl, self._qdims)

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

    * ket / bra — L2 norm via :meth:`SparseBCOOImpl.l2_norm_batched` (handles
      batch dimensions).
    * operator — trace norm assuming PSD (nuclear norm = tr(rho) for density
      matrices).  This is exact for density matrices; for general non-PSD
      operators convert to dense first.

    Args:
        qarr: Input quantum array.

    Returns:
        The norm as a scalar (or batched array of scalars).
    """
    if qarr.qtype in [Qtypes.ket, Qtypes.bra] and qarr.is_sparse_bcoo:
        return qarr._impl.l2_norm_batched(qarr.bdims)

    if qarr.qtype == Qtypes.oper and qarr.is_sparse_bcoo:
        # Nuclear norm = trace for positive-semidefinite (density matrix) operators.
        # jnp.real strips any floating-point imaginary artefact.
        return jnp.real(qarr._impl.trace())

    if qarr.qtype == Qtypes.oper and qarr.is_sparse_dia:
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
        # Vectors store the Hilbert space on the single trailing axis.
        return jnp.sqrt(jnp.sum(jnp.abs(qdata) ** 2, axis=-1))


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
        # Vectors (kets/bras) carry one trailing space axis; operators carry two.
        # Tensoring requires a consistent qtype across args (mixed is invalid).
        is_vec = dense_args[0].qtype in (Qtypes.ket, Qtypes.bra)
        n_space = 1 if is_vec else 2
        data = dense_args[0].data
        dims_0 = dense_args[0].dims[0]
        dims_1 = dense_args[0].dims[1]
        for arg in dense_args[1:]:
            a, b = data, arg.data
            ba, bb = a.shape[:-n_space], b.shape[:-n_space]
            if len(ba) > len(bb):
                batch_dim = ba
            elif len(ba) == len(bb):
                batch_dim = ba if prod(ba) >= prod(bb) else bb
            else:
                batch_dim = bb

            if is_vec:
                # (..., N) ⊗ (..., M) -> (..., N*M)
                data = jnp.einsum("...i,...j->...ij", a, b).reshape(
                    *batch_dim, a.shape[-1] * b.shape[-1]
                )
            else:
                # (..., M, N) ⊗ (..., K, L) -> (..., M*K, N*L)
                data = jnp.einsum("...ij,...kl->...ikjl", a, b).reshape(
                    *batch_dim, a.shape[-2] * b.shape[-2], -1
                )
            dims_0 = dims_0 + arg.dims[0]
            dims_1 = dims_1 + arg.dims[1]
        impl = target_impl_type.get_impl_class().from_data(data)
        return Qarray._from_impl(impl, Qdims((dims_0, dims_1)))

    # Non-parallel: delegate to each impl's kron method.
    # All-sparse inputs stay sparse; mixed inputs promote to dense via _coerce.
    current_impl = args[0]._impl
    dims_0 = args[0].dims[0]
    dims_1 = args[0].dims[1]
    for arg in args[1:]:
        current_impl = current_impl.kron(arg._impl)
        dims_0 = dims_0 + arg.dims[0]
        dims_1 = dims_1 + arg.dims[1]
    return Qarray._from_impl(current_impl, Qdims((dims_0, dims_1)))


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
    if qarr.is_sparse_bcoo:
        return qarr._impl.trace()
    if qarr.is_sparse_dia:
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
        The *n*-th matrix power as a ``Qarray`` (stays SparseDIA for integer
        non-negative exponents when the input is SparseDIA).

    Raises:
        ValueError: If *n* is a float and the matrix has negative eigenvalues
            (and *clip_eigvals* is ``False``).
    """
    # SparseDIA fast path: binary exponentiation stays in SparseDIA format.
    if qarr.is_sparse_dia and isinstance(n, int) and n >= 0:
        new_impl = qarr._impl.powm(n)
        return Qarray.create(new_impl.data, dims=qarr.dims, implementation=new_impl.impl_type)

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
    if qarr.is_sparse_bcoo:
        new_impl = qarr._impl.keep_only_diag()
        return Qarray.create(new_impl.data, dims=dims, implementation=QarrayImplType.SPARSE_BCOO)
    if qarr.is_sparse_dia:
        from jaxquantum.core.sparse_dia import SparseDiaImpl
        impl = qarr._impl
        n = impl._diags.shape[-1]
        if 0 in impl._offsets:
            i = impl._offsets.index(0)
            main_diag = impl._diags[..., i:i + 1, :]
        else:
            main_diag = jnp.zeros((*impl._diags.shape[:-2], 1, n), dtype=impl._diags.dtype)
        new_impl = SparseDiaImpl(_offsets=(0,), _diags=main_diag)
        return Qarray.create(new_impl.get_data(), dims=dims, implementation=QarrayImplType.SPARSE_DIA)
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
    dims = ket_from_op_dims(qarr.dims)

    # numpy returns [batch, :, i] as the i-th eigenvector
    # we want [batch, i, :] as the i-th eigenvector
    evecs = jnp.swapaxes(evecs, -2, -1)
    evecs = Qarray._from_impl(
        DenseImpl._make(evecs),
        Qdims(dims),
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

    For ket/bra vectors (stored on a single axis) this is just a complex
    conjugate with the ``dims`` reversed — no axis swap, so the data stays
    1‑D.  For operators it is the usual conjugate transpose.

    Args:
        qarr: Input quantum array.

    Returns:
        The conjugate transpose with swapped ``dims``.
    """
    dims = qarr.dims[::-1]
    if qarr.qtype in (Qtypes.ket, Qtypes.bra):
        new_impl = qarr._impl.conj()
    else:
        new_impl = qarr._impl.dag()
    # Pass only the (reversed) dims tuple — not bdims/qtype. ``_bdims`` is a
    # static pytree field and is stale under ``vmap``; let create() infer the
    # batch dims from the actual data shape (which vmap does update).
    return Qarray._from_impl(new_impl, Qdims(dims))


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
# SparseBCOOQarray and SparseDIAQarray are defined lazily (impls imported at runtime)
# Use Qarray[SparseBCOOImpl] / Qarray[SparseDiaImpl] once those modules are imported.

ARRAY_TYPES = (Array, ndarray, Qarray)

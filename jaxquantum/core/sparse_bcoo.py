"""Sparse BCOO backend for Qarray.

Implements the JAX experimental BCOO sparse format as a Qarray storage backend.
"""

from __future__ import annotations

from flax import struct
from jax import Array
from copy import deepcopy
import jax.numpy as jnp
from jax.experimental import sparse

# QarrayImpl and friends are imported below (after qarray.py is fully loaded)
# to match the pattern used by sparse_dia.py and avoid circular imports.
from jaxquantum.core.qarray import QarrayImpl, DenseImpl, QarrayImplType  # noqa: E402


@struct.dataclass
class SparseBCOOImpl(QarrayImpl):
    """Sparse implementation using JAX experimental BCOO sparse arrays.

    Attributes:
        _data: The underlying ``sparse.BCOO`` array.
    """

    _data: sparse.BCOO

    PROMOTION_ORDER = 1  # noqa: RUF012 — not a struct field; no annotation intentional

    @classmethod
    def from_data(cls, data) -> "SparseBCOOImpl":
        """Wrap *data* in a new ``SparseBCOOImpl``, converting to BCOO if needed.

        Args:
            data: A ``sparse.BCOO`` or array-like input.

        Returns:
            A ``SparseBCOOImpl`` wrapping a BCOO representation of *data*.
        """
        return cls(_data=cls._to_sparse(data))

    def get_data(self) -> Array:
        """Return the underlying BCOO sparse array."""
        return self._data

    def matmul(self, other: QarrayImpl) -> QarrayImpl:
        """Matrix multiply ``self @ other``.

        When *other* is a ``DenseImpl``, JAX's native BCOO @ dense path is
        used (no self-densification).  When *other* is also a
        ``SparseBCOOImpl``, a sparse @ sparse product is performed.

        Args:
            other: Right-hand operand.

        Returns:
            A ``DenseImpl`` (sparse @ dense) or ``SparseBCOOImpl`` (sparse @
            sparse) containing the matrix product.
        """
        if isinstance(other, DenseImpl):
            return DenseImpl(self._data @ other._data)
        a, b = self._coerce(other)
        if a is not self:
            return a.matmul(b)
        return SparseBCOOImpl(self._data @ b._data)

    def add(self, other: QarrayImpl) -> QarrayImpl:
        """Element-wise addition ``self + other``, coercing types as needed.

        Args:
            other: Right-hand operand.

        Returns:
            A ``SparseBCOOImpl`` (both sparse) or ``DenseImpl`` (mixed) sum.
        """
        a, b = self._coerce(other)
        if a is not self:
            return a.add(b)
        x, y = self._data, b._data
        if x.indices.dtype != y.indices.dtype:
            y = sparse.BCOO((y.data, y.indices.astype(x.indices.dtype)), shape=y.shape)
        return SparseBCOOImpl(x + y)

    def sub(self, other: QarrayImpl) -> QarrayImpl:
        """Element-wise subtraction ``self - other``, coercing types as needed.

        Args:
            other: Right-hand operand.

        Returns:
            A ``SparseBCOOImpl`` (both sparse) or ``DenseImpl`` (mixed) difference.
        """
        a, b = self._coerce(other)
        if a is not self:
            return a.sub(b)
        x, y = self._data, b._data
        if x.indices.dtype != y.indices.dtype:
            y = sparse.BCOO((y.data, y.indices.astype(x.indices.dtype)), shape=y.shape)
        return SparseBCOOImpl(x - y)

    def mul(self, scalar) -> QarrayImpl:
        """Scalar multiplication.

        Args:
            scalar: Scalar value.

        Returns:
            A ``SparseBCOOImpl`` with each stored value multiplied by *scalar*.
        """
        return SparseBCOOImpl(scalar * self._data)

    def dag(self) -> QarrayImpl:
        """Conjugate transpose without densifying.

        Transposes the last two dimensions of the BCOO array and conjugates
        the stored values.

        Returns:
            A ``SparseBCOOImpl`` containing the conjugate transpose.
        """
        ndim = self._data.ndim
        if ndim >= 2:
            permutation = tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
            transposed_data = sparse.bcoo_transpose(self._data, permutation=permutation)
        else:
            transposed_data = self._data

        conjugated_data = sparse.BCOO(
            (jnp.conj(transposed_data.data), transposed_data.indices),
            shape=transposed_data.shape,
        )
        return SparseBCOOImpl(conjugated_data)

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

    def to_sparse_bcoo(self) -> "SparseBCOOImpl":
        """Return self (already sparse BCOO).

        Returns:
            This ``SparseBCOOImpl`` instance unchanged.
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

        Returns:
            The Frobenius norm as a scalar.
        """
        return jnp.sqrt(jnp.sum(jnp.abs(self._data.data) ** 2))

    @classmethod
    def _real(cls, data):
        """Return a BCOO array with only the real parts of the stored values."""
        return sparse.BCOO((jnp.real(data.data), data.indices), shape=data.shape)

    def real(self) -> QarrayImpl:
        """Element-wise real part.

        Returns:
            A ``SparseBCOOImpl`` containing the real parts of stored values.
        """
        return SparseBCOOImpl(SparseBCOOImpl._real(self._data))

    @classmethod
    def _imag(cls, data):
        """Return a BCOO array with only the imaginary parts of the stored values."""
        return sparse.BCOO((jnp.imag(data.data), data.indices), shape=data.shape)

    def imag(self) -> QarrayImpl:
        """Element-wise imaginary part.

        Returns:
            A ``SparseBCOOImpl`` containing the imaginary parts of stored values.
        """
        return SparseBCOOImpl(SparseBCOOImpl._imag(self._data))

    @classmethod
    def _conj(cls, data):
        """Return a BCOO array with complex-conjugated stored values."""
        return sparse.BCOO((jnp.conj(data.data), data.indices), shape=data.shape)

    def conj(self) -> QarrayImpl:
        """Element-wise complex conjugate.

        Returns:
            A ``SparseBCOOImpl`` containing the complex-conjugated stored values.
        """
        return SparseBCOOImpl(SparseBCOOImpl._conj(self._data))

    @classmethod
    def _abs(cls, data):
        """Return a BCOO array with absolute values of stored entries."""
        return sparse.sparsify(jnp.abs)(data)

    def abs(self) -> QarrayImpl:
        """Element-wise absolute value.

        Returns:
            A ``SparseBCOOImpl`` containing the absolute values of stored entries.
        """
        return SparseBCOOImpl(SparseBCOOImpl._abs(self._data))

    @classmethod
    def _eye_data(cls, n: int, dtype=None):
        """Create an ``n x n`` identity matrix as a sparse BCOO with O(n) memory.

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

        Returns:
            Trace value(s).
        """
        indices = self._data.indices
        values = self._data.data
        ndim = indices.shape[-1]

        is_diag = indices[:, -2] == indices[:, -1]

        if ndim == 2:
            return jnp.sum(values * is_diag)
        else:
            batch_shape = self._data.shape[:-2]
            B = int(jnp.prod(jnp.array(batch_shape)))
            strides = [1]
            for s in reversed(batch_shape[1:]):
                strides.insert(0, strides[0] * s)
            strides = jnp.array(strides, dtype=jnp.int32)
            flat_batch_idx = jnp.sum(indices[:, :-2] * strides, axis=-1)
            result = jnp.zeros(B, dtype=values.dtype).at[flat_batch_idx].add(
                values * is_diag
            )
            return result.reshape(batch_shape)

    def keep_only_diag(self) -> "SparseBCOOImpl":
        """Zero out off-diagonal stored entries without densifying.

        Returns:
            A ``SparseBCOOImpl`` with only diagonal entries non-zero.
        """
        indices = self._data.indices
        values = self._data.data
        is_diag = indices[:, -2] == indices[:, -1]
        new_values = values * is_diag
        return SparseBCOOImpl(sparse.BCOO((new_values, indices), shape=self._data.shape))

    def l2_norm_batched(self, bdims: tuple) -> Array:
        """Compute the L2 norm per batch element without densifying.

        Args:
            bdims: Tuple of batch dimension sizes.

        Returns:
            Scalar or array of L2 norms.
        """
        values = self._data.data
        indices = self._data.indices
        n_batch_dims = len(bdims)
        sq = jnp.abs(values) ** 2

        if n_batch_dims == 0:
            return jnp.sqrt(jnp.sum(sq))
        else:
            B = int(jnp.prod(jnp.array(bdims)))
            strides = [1]
            for s in reversed(bdims[1:]):
                strides.insert(0, strides[0] * s)
            strides = jnp.array(strides, dtype=jnp.int32)
            flat_batch_idx = jnp.sum(indices[:, :n_batch_dims] * strides, axis=-1)
            sum_sq = (
                jnp.zeros(B, dtype=jnp.float64)
                .at[flat_batch_idx]
                .add(sq)
            )
            return jnp.sqrt(sum_sq).reshape(bdims)

    def __deepcopy__(self, memo=None):
        return SparseBCOOImpl(_data=deepcopy(self._data, memo))

    def tidy_up(self, atol) -> "SparseBCOOImpl":
        """Zero out stored values whose real or imaginary magnitude is below *atol*.

        Args:
            atol: Absolute tolerance threshold.

        Returns:
            A new ``SparseBCOOImpl`` with small values zeroed.
        """
        values = self._data.data
        re = jnp.real(values)
        im = jnp.imag(values)
        new_values = re * (jnp.abs(re) > atol) + 1j * im * (jnp.abs(im) > atol)
        return SparseBCOOImpl(
            sparse.BCOO((new_values, self._data.indices), shape=self._data.shape)
        )

    def kron(self, other: "QarrayImpl") -> "QarrayImpl":
        """Kronecker product using ``sparsify(jnp.kron)`` — stays sparse.

        Args:
            other: Right-hand operand.

        Returns:
            A ``SparseBCOOImpl`` containing the Kronecker product when both
            operands are sparse; a ``DenseImpl`` when types differ.
        """
        a, b = self._coerce(other)
        if a is not self:
            return a.kron(b)
        sparse_kron = sparse.sparsify(jnp.kron)
        return SparseBCOOImpl(sparse_kron(self._data, b._data))


# Register with the enum registry
QarrayImplType.register(SparseBCOOImpl, QarrayImplType.SPARSE_BCOO)

__all__ = ["SparseBCOOImpl"]

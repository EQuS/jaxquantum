"""Sparse diagonal (SparseDIA) backend for Qarray.

Stores only the diagonal *values* of a matrix, making quantum operators with
small numbers of non-zero diagonals (annihilation, creation, number, Kerr…)
far cheaper than Dense or BCOO:

  * Memory: O(d * n) where d = number of stored diagonals, n = matrix size
  * No index arrays (unlike BCOO which stores (row, col) per non-zero)
  * ``_offsets`` is *static* Python metadata (pytree_node=False), so JAX
    unrolls all loops over diagonals at compile time — only static slices,
    no dynamic indexing or scatter/gather.

Padding convention (Convention A):

  For diagonal at offset k (k ≥ 0):
      diags[..., i, j] = A[j-k, j]   for j ∈ [k, n-1];  zeros at [0:k]
  For diagonal at offset k (k < 0):
      diags[..., i, j] = A[j-k, j]   for j ∈ [0, n+k-1]; zeros at [n+k:]

Unified access formula (holds for any k, out-of-range slots are zero):
    A[i, i+k]  =  diags[..., diag_idx, i+k]

This makes every matrix operation a set of aligned slice multiplications.

Some improvements (_dia_slice helper, integer matrix power, diagonal-range pruning,
offset detection) were identified by studying the dynamiqs library
(https://github.com/dynamiqs/dynamiqs).
"""

from __future__ import annotations

import numpy as np
from copy import deepcopy
from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import struct
from jax import Array

if TYPE_CHECKING:
    from jaxquantum.core.qarray import DenseImpl, SparseImpl, QarrayImplType


# ---------------------------------------------------------------------------
# Slice helper
# ---------------------------------------------------------------------------

def _dia_slice(k: int) -> slice:
    """Slice selecting the valid data positions for diagonal offset k.

    For k ≥ 0: valid data lives at column indices [k, n), so slice(k, None).
    For k < 0: valid data lives at column indices [0, n+k), so slice(None, k).

    The complementary slice (for the result rows / left-operand columns) is
    always ``_dia_slice(-k)``.
    """
    return slice(k, None) if k >= 0 else slice(None, k)


# ---------------------------------------------------------------------------
# Raw data container
# ---------------------------------------------------------------------------

@struct.dataclass
class SparseDiaData:
    """Lightweight pytree-compatible container for sparse-diagonal raw data.

    Returned by ``SparseDiaImpl.get_data()`` and consumed by
    ``SparseDiaImpl.from_data()``.  Registered as a JAX pytree via Flax's
    ``@struct.dataclass``; ``offsets`` is *not* a pytree leaf (it is static
    compile-time metadata).

    Attributes:
        offsets: Static tuple of diagonal offsets (pytree_node=False).
        diags:   JAX array of shape (*batch, n_diags, n) containing the
                 padded diagonal values.
    """

    offsets: tuple = struct.field(pytree_node=False)
    diags: Array

    # Class-level marker (not a dataclass field — no type annotation).
    # Used by DenseImpl.can_handle_data to exclude SparseDiaData without
    # a direct import (which would be circular).
    _is_sparsedia = True

    @property
    def shape(self) -> tuple:
        """Shape of the represented square matrix (*batch, n, n)."""
        n = self.diags.shape[-1]
        return (*self.diags.shape[:-2], n, n)

    @property
    def dtype(self):
        """Dtype of the stored diagonal values."""
        return self.diags.dtype

    def __mul__(self, scalar):
        return SparseDiaData(offsets=self.offsets, diags=self.diags * scalar)

    def __rmul__(self, scalar):
        return SparseDiaData(offsets=self.offsets, diags=scalar * self.diags)

    def __getitem__(self, index):
        """Index into the batch dimension(s), preserving offsets."""
        return SparseDiaData(offsets=self.offsets, diags=self.diags[index])

    def __len__(self):
        """Number of elements along the leading batch dimension."""
        return self.shape[0]

    def reshape(self, *new_shape):
        """Reshape batch dimensions while preserving diagonal structure.

        ``new_shape`` must end with ``(N, N)`` (the matrix dims are unchanged).
        Only the leading batch dims are reshaped.
        """
        new_batch = new_shape[:-2]
        n = self.diags.shape[-1]
        new_diags = self.diags.reshape(*new_batch, len(self.offsets), n)
        return SparseDiaData(offsets=self.offsets, diags=new_diags)

    def __matmul__(self, other):
        """SparseDIA @ dense → dense (used by mesolve ODE RHS)."""
        # _sparsedia_matmul_dense is defined later in this module; Python
        # resolves the name at call time so forward reference is fine.
        return _sparsedia_matmul_dense(self.offsets, self.diags, other)

    def __rmatmul__(self, other):
        """dense @ SparseDIA → dense (used by mesolve ODE RHS)."""
        return _sparsedia_rmatmul_dense(other, self.offsets, self.diags)


# ---------------------------------------------------------------------------
# Helper: dense → SparseDIA conversion
# ---------------------------------------------------------------------------

def _dense_to_sparsedia(arr: np.ndarray) -> tuple[tuple, np.ndarray]:
    """Extract non-zero diagonal offsets and padded values from a dense array.

    Uses the first batch element (if batched) to detect which diagonals are
    non-zero.  Safe to call outside JIT because *arr* must be a concrete
    numpy / JAX array.

    Args:
        arr: Dense array of shape (*batch, n, n).

    Returns:
        Tuple of (offsets, diags) where:
          - offsets is a sorted tuple of integer offsets
          - diags is a numpy array of shape (*batch, n_diags, n)
    """
    n = arr.shape[-1]
    batch_shape = arr.shape[:-2]

    # Union non-zero diagonals across all batch elements via a single mask + nonzero call.
    arr_np = np.asarray(arr)
    flat_np = arr_np.reshape(-1, n, n)
    union_mask = np.any(flat_np != 0, axis=0)   # (n, n): True where any batch elem is non-zero
    r, c = np.nonzero(union_mask)
    offsets = tuple(sorted(set((c - r).tolist()))) if len(r) > 0 else (0,)

    diags = np.zeros((*batch_shape, len(offsets), n), dtype=arr_np.dtype)
    for i, k in enumerate(offsets):
        # np.diagonal returns shape (*batch, n-|k|)
        d = np.diagonal(arr_np, offset=k, axis1=-2, axis2=-1)
        lo = max(k, 0)
        hi = n - max(-k, 0)
        diags[..., i, lo:hi] = d

    return offsets, diags


# ---------------------------------------------------------------------------
# SparseDiaImpl
# ---------------------------------------------------------------------------

from jaxquantum.core.qarray import QarrayImpl, DenseImpl, SparseImpl, QarrayImplType  # noqa: E402


@struct.dataclass
class SparseDiaImpl(QarrayImpl):
    """Sparse-diagonal backend storing only diagonal values.

    Data layout::

        _offsets  : tuple[int, ...]          — static (pytree_node=False)
        _diags    : Array[*batch, n_diags, n] — JAX-traced values

    For offset k, the convention is:
        * k ≥ 0 : valid data at ``_diags[..., i, k:]``, zeros at ``[0:k]``
        * k < 0 : valid data at ``_diags[..., i, :n+k]``, zeros at ``[n+k:]``

    In both cases: ``A[row, row+k] = _diags[..., i, row+k]``
    """

    _offsets: tuple = struct.field(pytree_node=False)
    _diags: Array

    PROMOTION_ORDER = 0  # noqa: RUF012 — not a struct field

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_data(cls, data) -> "SparseDiaImpl":
        """Wrap *data* in a new ``SparseDiaImpl``.

        Accepts either a :class:`SparseDiaData` container (direct wrap) or
        a dense array-like (auto-detect non-zero diagonals via numpy, safe
        to call before JIT).

        Args:
            data: A :class:`SparseDiaData` or dense array of shape
                (*batch, n, n).

        Returns:
            A new ``SparseDiaImpl`` instance.
        """
        if isinstance(data, SparseDiaData):
            return cls(_offsets=data.offsets, _diags=data.diags)
        offsets, diags_np = _dense_to_sparsedia(np.asarray(data))
        return cls(_offsets=offsets, _diags=jnp.array(diags_np))

    @classmethod
    def from_diags(cls, offsets: tuple, diags: Array) -> "SparseDiaImpl":
        """Directly construct from sorted offsets and padded diagonal array.

        This is the preferred factory when diagonal structure is known in
        advance (e.g., when building ``destroy`` or ``create`` operators).

        Args:
            offsets: Tuple of integer diagonal offsets (need not be sorted;
                will be sorted internally).
            diags:   JAX array of shape (*batch, n_diags, n) with padded
                     diagonal values matching *offsets*.

        Returns:
            A new ``SparseDiaImpl`` instance.
        """
        return cls(_offsets=tuple(sorted(offsets)), _diags=diags)

    # ------------------------------------------------------------------
    # QarrayImpl abstract methods
    # ------------------------------------------------------------------

    def get_data(self) -> SparseDiaData:
        """Return a :class:`SparseDiaData` container with the raw diagonal data."""
        return SparseDiaData(offsets=self._offsets, diags=self._diags)

    def shape(self) -> tuple:
        """Shape of the represented square matrix (including batch dims)."""
        n = self._diags.shape[-1]
        return (*self._diags.shape[:-2], n, n)

    def dtype(self):
        """Dtype of the stored diagonal values."""
        return self._diags.dtype

    def __deepcopy__(self, memo=None):
        return SparseDiaImpl(
            _offsets=deepcopy(self._offsets),
            _diags=self._diags,
        )

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def mul(self, scalar) -> "SparseDiaImpl":
        """Scalar multiplication — scales all diagonal values."""
        return SparseDiaImpl(_offsets=self._offsets, _diags=scalar * self._diags)

    def neg(self) -> "SparseDiaImpl":
        """Negation."""
        return SparseDiaImpl(_offsets=self._offsets, _diags=-self._diags)

    def add(self, other: QarrayImpl) -> QarrayImpl:
        """Element-wise addition.

        SparseDIA + SparseDIA stays SparseDIA (union of offsets, static).
        Otherwise coerces to the higher-order type.
        """
        if isinstance(other, SparseDiaImpl):
            return _sparsedia_add(self, other)
        a, b = self._coerce(other)
        if a is not self:
            return a.add(b)
        return a.add(b)

    def sub(self, other: QarrayImpl) -> QarrayImpl:
        """Element-wise subtraction."""
        if isinstance(other, SparseDiaImpl):
            return _sparsedia_add(self, other, subtract=True)
        a, b = self._coerce(other)
        if a is not self:
            return a.sub(b)
        return a.sub(b)

    def matmul(self, other: QarrayImpl) -> QarrayImpl:
        """Matrix multiplication.

        * SparseDIA @ SparseDIA → SparseDIA  (O(d₁·d₂·n))
        * SparseDIA @ Dense    → Dense       (O(d·n²), no densification of self)
        * Others               → coerce then delegate
        """
        if isinstance(other, DenseImpl):
            return DenseImpl(_sparsedia_matmul_dense(
                self._offsets, self._diags, other._data
            ))
        if isinstance(other, SparseDiaImpl):
            offsets, diags = _sparsedia_matmul_sparsedia(
                self._offsets, self._diags,
                other._offsets, other._diags,
            )
            return SparseDiaImpl(_offsets=offsets, _diags=diags)
        a, b = self._coerce(other)
        if a is not self:
            return a.matmul(b)
        return a.matmul(b)

    def dag(self) -> "SparseDiaImpl":
        """Conjugate transpose without densification.

        Negates every offset and rearranges the stored values so that the
        padding convention remains consistent.
        """
        new_offsets = tuple(-k for k in self._offsets)
        new_diags = jnp.zeros_like(self._diags)
        for i, k in enumerate(self._offsets):
            s = _dia_slice(k)    # valid data slice for offset k
            sm = _dia_slice(-k)  # valid data slice for offset -k (the new position)
            new_diags = new_diags.at[..., i, sm].set(jnp.conj(self._diags[..., i, s]))
        return SparseDiaImpl(_offsets=new_offsets, _diags=new_diags)

    def kron(self, other: QarrayImpl) -> QarrayImpl:
        """Kronecker product.

        SparseDIA ⊗ SparseDIA stays SparseDIA: output offset for pair
        (kA, kB) is ``kA * m + kB`` where m = dim(B).  Fully vectorised —
        no loops at JAX level.
        """
        if isinstance(other, SparseDiaImpl):
            return _sparsedia_kron(self, other)
        a, b = self._coerce(other)
        if a is not self:
            return a.kron(b)
        return a.kron(b)

    def tidy_up(self, atol) -> "SparseDiaImpl":
        """Zero diagonal values whose magnitude is below *atol*."""
        diags = self._diags
        real_part = jnp.where(jnp.abs(jnp.real(diags)) < atol, 0.0, jnp.real(diags))
        imag_part = jnp.where(jnp.abs(jnp.imag(diags)) < atol, 0.0, jnp.imag(diags))
        new_diags = (real_part + 1j * imag_part).astype(diags.dtype)
        return SparseDiaImpl(_offsets=self._offsets, _diags=new_diags)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_dense(self) -> "DenseImpl":
        """Convert to a ``DenseImpl`` by summing diagonal contributions."""
        n = self._diags.shape[-1]
        batch_shape = self._diags.shape[:-2]
        result = jnp.zeros((*batch_shape, n, n), dtype=self._diags.dtype)
        for i, k in enumerate(self._offsets):
            s = _dia_slice(k)
            length = n - abs(k)
            if length <= 0:
                continue
            vals = self._diags[..., i, s]
            row_idx = jnp.arange(length) + max(-k, 0)
            col_idx = row_idx + k
            result = result.at[..., row_idx, col_idx].set(vals)
        return DenseImpl(result)

    def to_sparse(self) -> "SparseImpl":
        """Convert to a ``SparseImpl`` (BCOO) via dense."""
        return self.to_dense().to_sparse()

    def to_sparse_dia(self) -> "SparseDiaImpl":
        """Return self (already SparseDIA)."""
        return self

    # ------------------------------------------------------------------
    # Class-method interface
    # ------------------------------------------------------------------

    @classmethod
    def _eye_data(cls, n: int, dtype=None):
        """Return an n×n identity as a dense JAX array.

        ``from_data`` will automatically convert it to SparseDIA format
        when the implementation type is ``SPARSE_DIA``.
        """
        return jnp.eye(n, dtype=dtype)

    @classmethod
    def can_handle_data(cls, arr) -> bool:
        """Return True only for :class:`SparseDiaData` objects."""
        return isinstance(arr, SparseDiaData)

    @classmethod
    def dag_data(cls, arr: SparseDiaData) -> SparseDiaData:
        """Conjugate transpose of raw :class:`SparseDiaData` without densification."""
        impl = SparseDiaImpl(_offsets=arr.offsets, _diags=arr.diags)
        result = impl.dag()
        return result.get_data()

    # ------------------------------------------------------------------
    # Extra sparse-native methods (no densification)
    # ------------------------------------------------------------------

    def trace(self):
        """Compute trace directly from the main diagonal (offset 0).

        Returns:
            Scalar trace (sum of main diagonal values).
        """
        if 0 in self._offsets:
            i = self._offsets.index(0)
            return jnp.sum(self._diags[..., i, :], axis=-1)
        return jnp.zeros(self._diags.shape[:-2], dtype=self._diags.dtype)

    def frobenius_norm(self):
        """Frobenius norm computed directly from stored diagonal values."""
        return jnp.sqrt(jnp.sum(jnp.abs(self._diags) ** 2))

    def real(self) -> "SparseDiaImpl":
        """Element-wise real part of stored values."""
        return SparseDiaImpl(
            _offsets=self._offsets,
            _diags=jnp.real(self._diags).astype(self._diags.dtype),
        )

    def imag(self) -> "SparseDiaImpl":
        """Element-wise imaginary part of stored values."""
        return SparseDiaImpl(
            _offsets=self._offsets,
            _diags=jnp.imag(self._diags).astype(self._diags.dtype),
        )

    def conj(self) -> "SparseDiaImpl":
        """Element-wise complex conjugate of stored values."""
        return SparseDiaImpl(_offsets=self._offsets, _diags=jnp.conj(self._diags))

    def powm(self, n: int) -> "SparseDiaImpl":
        """Integer matrix power staying SparseDIA via binary exponentiation.

        Uses O(log n) SparseDIA @ SparseDIA multiplications rather than
        densifying.  A^0 returns the identity operator.

        Args:
            n: Non-negative integer exponent.

        Returns:
            A ``SparseDiaImpl`` equal to this matrix raised to the *n*-th power.

        Raises:
            ValueError: If *n* is negative.
        """
        if n < 0:
            raise ValueError("powm requires n >= 0")
        if n == 0:
            size = self._diags.shape[-1]
            eye_diags = jnp.ones((*self._diags.shape[:-2], 1, size), dtype=self._diags.dtype)
            return SparseDiaImpl(_offsets=(0,), _diags=eye_diags)
        if n == 1:
            return self
        half = self.powm(n // 2)
        squared = half.matmul(half)  # SparseDIA @ SparseDIA → SparseDIA
        return squared if n % 2 == 0 else self.matmul(squared)


# ---------------------------------------------------------------------------
# Pure-function helpers (operate on raw arrays, no QarrayImpl wrapping)
# ---------------------------------------------------------------------------

def _sparsedia_add(
    a: SparseDiaImpl,
    b: SparseDiaImpl,
    subtract: bool = False,
) -> SparseDiaImpl:
    """Add (or subtract) two SparseDiaImpl matrices, preserving SparseDIA format.

    Computes the union of offsets at Python level (static), then copies /
    sums the corresponding diagonal arrays.
    """
    n = a._diags.shape[-1]
    batch_shape = jnp.broadcast_shapes(a._diags.shape[:-2], b._diags.shape[:-2])

    out_offsets = tuple(sorted(set(a._offsets) | set(b._offsets)))
    out_diags = jnp.zeros((*batch_shape, len(out_offsets), n), dtype=a._diags.dtype)

    a_idx = {k: i for i, k in enumerate(a._offsets)}
    b_idx = {k: i for i, k in enumerate(b._offsets)}

    for oi, k in enumerate(out_offsets):
        val = jnp.zeros((*batch_shape, n), dtype=a._diags.dtype)
        if k in a_idx:
            val = val + a._diags[..., a_idx[k], :]
        if k in b_idx:
            sign = -1 if subtract else 1
            val = val + sign * b._diags[..., b_idx[k], :]
        out_diags = out_diags.at[..., oi, :].set(val)

    return SparseDiaImpl(_offsets=out_offsets, _diags=out_diags)


def _sparsedia_matmul_dense(
    offsets: tuple,
    diags: Array,
    B: Array,
) -> Array:
    """Compute (SparseDIA) @ (dense matrix) → dense, without densifying the LHS.

    For each stored diagonal at offset k:
        result[..., row_range, :] += diags[..., i, valid_slice, None] * B[..., valid_slice, :]

    Complexity: O(d * n * m) where n×n is the operator and n×m is B.

    Args:
        offsets: Static tuple of diagonal offsets for the LHS.
        diags:   JAX array of shape (*batch, n_diags, n).
        B:       Dense right-hand side of shape (*batch, n, m).

    Returns:
        Dense product of shape (*batch, n, m).
    """
    # n = diags.shape[-1]
    batch_shape = jnp.broadcast_shapes(diags.shape[:-2], B.shape[:-2])
    result = jnp.zeros(
        (*batch_shape, B.shape[-2], B.shape[-1]),
        dtype=jnp.result_type(diags.dtype, B.dtype),
    )
    for i, k in enumerate(offsets):
        s = _dia_slice(k)    # valid column slice for diagonal k
        sm = _dia_slice(-k)  # corresponding row slice for the result
        result = result.at[..., sm, :].add(diags[..., i, s, None] * B[..., s, :])
    return result


def _sparsedia_rmatmul_dense(
    B: Array,
    offsets: tuple,
    diags: Array,
) -> Array:
    """Compute (dense matrix) @ (SparseDIA) → dense.

    For each stored diagonal at offset k:
        C[..., :, k:] += B[..., :, :n-k] * diags[..., i, k:][..., None, :]  (k ≥ 0)
        C[..., :, :n-m] += B[..., :, m:] * diags[..., i, :n-m][..., None, :]  (k < 0)

    Complexity: O(d * n * p) where n×n is the operator and p×n is B.

    Args:
        B:       Dense left-hand side of shape (*batch, p, n).
        offsets: Static tuple of diagonal offsets for the RHS.
        diags:   JAX array of shape (*batch, n_diags, n).

    Returns:
        Dense product of shape (*batch, p, n).
    """
    # n = diags.shape[-1]
    batch_shape = jnp.broadcast_shapes(diags.shape[:-2], B.shape[:-2])
    result = jnp.zeros(
        (*batch_shape, B.shape[-2], B.shape[-1]),
        dtype=jnp.result_type(diags.dtype, B.dtype),
    )
    for i, k in enumerate(offsets):
        s = _dia_slice(k)    # valid column slice for diagonal k
        sm = _dia_slice(-k)  # complementary slice for B columns / result columns
        result = result.at[..., :, s].add(B[..., :, sm] * diags[..., i, s][..., None, :])
    return result


def _sparsedia_matmul_sparsedia(
    left_offsets: tuple,
    left_diags: Array,
    right_offsets: tuple,
    right_diags: Array,
) -> tuple[tuple, Array]:
    """Compute (SparseDIA) @ (SparseDIA) → SparseDIA.

    Derivation uses the unified access formula A[i, i+k] = diags[i+k].
    For the contribution of diagonal pair (k1, k2) to output at kout = k1+k2:

        out_diag[j + k2] += left_diag[j] * right_diag[j + k2]

    This is an aligned slice multiply, handled by the sign of k2:
        k2 ≥ 0: out.at[kout:].add( left[:n-k2] * right[k2:] )
        k2 < 0: out.at[:n+k2].add( left[-k2:]  * right[:n+k2] )

    Complexity: O(d1 * d2 * n).

    Args:
        left_offsets:  Static offsets for the LHS matrix.
        left_diags:    JAX array (*batch, d1, n).
        right_offsets: Static offsets for the RHS matrix.
        right_diags:   JAX array (*batch, d2, n).

    Returns:
        Tuple of (out_offsets, out_diags).
    """
    n = left_diags.shape[-1]
    batch_shape = jnp.broadcast_shapes(
        left_diags.shape[:-2], right_diags.shape[:-2]
    )

    # Pre-filter output offsets: diagonal pairs where |k1+k2| >= n are zero.
    out_offset_set = sorted(
        {k1 + k2 for k1 in left_offsets for k2 in right_offsets if abs(k1 + k2) < n}
    )
    if not out_offset_set:
        out_offset_set = [0]
    out_offset_idx = {k: i for i, k in enumerate(out_offset_set)}
    out_diags = jnp.zeros(
        (*batch_shape, len(out_offset_set), n), dtype=left_diags.dtype
    )

    for li, k1 in enumerate(left_offsets):
        for ri, k2 in enumerate(right_offsets):
            kout = k1 + k2
            if abs(kout) >= n:
                continue
            oi = out_offset_idx[kout]
            s = _dia_slice(k2)    # valid column slice for right diagonal k2
            sm = _dia_slice(-k2)  # complementary slice for left diagonal
            contribution = left_diags[..., li, sm] * right_diags[..., ri, s]
            out_diags = out_diags.at[..., oi, s].add(contribution)

    return tuple(out_offset_set), out_diags


def _sparsedia_kron(a: SparseDiaImpl, b: SparseDiaImpl) -> SparseDiaImpl:
    """Kronecker product of two SparseDiaImpl matrices → SparseDiaImpl.

    For operands A (n_A × n_A) and B (m × m), the output has dimension
    (n_A*m) × (n_A*m).  Each diagonal pair (kA, kB) contributes to the
    output diagonal at offset ``kout = kA * m + kB``.

    Key insight — the full output diagonal can be constructed without any
    scatter or dynamic indexing::

        out_diag_arr[s]  =  left_padded[ s // m ] * right_padded[ s % m ]

    which equals::

        jnp.repeat(left_padded, m, axis=-1) * jnp.tile(right_padded, n_A)

    Complexity: O(d_A * d_B * N) where N = n_A * m.

    Args:
        a: Left SparseDiaImpl of shape (*batch, n_A, n_A).
        b: Right SparseDiaImpl of shape (*batch, m, m).

    Returns:
        SparseDiaImpl of shape (*batch, N, N).
    """
    n_A = a._diags.shape[-1]
    m = b._diags.shape[-1]
    
    # N = n_A * m
    # batch_shape = jnp.broadcast_shapes(a._diags.shape[:-2], b._diags.shape[:-2])

    # Accumulate contributions per output offset
    out_accum: dict[int, Array] = {}

    for li, kA in enumerate(a._offsets):
        for ri, kB in enumerate(b._offsets):
            kout = kA * m + kB
            # Full output diagonal (length N) via repeat/tile — fully vectorised
            left_rep = jnp.repeat(a._diags[..., li, :], m, axis=-1)   # (*batch, N)
            right_tiled = jnp.tile(b._diags[..., ri, :], n_A)          # (*batch, N)
            contrib = left_rep * right_tiled
            if kout in out_accum:
                out_accum[kout] = out_accum[kout] + contrib
            else:
                out_accum[kout] = contrib

    out_offsets = tuple(sorted(out_accum.keys()))
    out_diags = jnp.stack([out_accum[k] for k in out_offsets], axis=-2)
    return SparseDiaImpl(_offsets=out_offsets, _diags=out_diags)


# ---------------------------------------------------------------------------
# Register with the QarrayImplType enum
# ---------------------------------------------------------------------------

QarrayImplType.register(SparseDiaImpl, QarrayImplType.SPARSE_DIA)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "SparseDiaData",
    "SparseDiaImpl",
]

"""cuQuantum backend for ``Qarray``.

Wraps ``cuquantum.densitymat.jax.OperatorTerm`` directly.  Arithmetic on
``Qarray[CuquantumImpl]`` delegates to ``OperatorTerm``'s native operator
overloads (``__add__``, ``__matmul__``, ``__and__``, ``dag()``...), which
build new ``OperatorTerm``s without ever materialising the dense
tensor-product matrix.

This module imports ``cuquantum.densitymat.jax`` at load time and raises
``ImportError`` when cuquantum isn't installed; ``core/__init__.py`` catches
that so the rest of the package keeps working on CPU-only installs.

Identity convention
-------------------
An *empty* ``OperatorTerm`` (zero products) on ``dims`` represents the
**identity superoperator** on that Hilbert space.  This convention is enforced
on the jaxquantum side only — ``OperatorTerm`` itself keeps its mathematical
"empty sum = zero" semantics.

- ``@``, ``&`` (matmul, kron): an empty operand naturally acts as identity.
  Short-circuited here.
- ``+``, ``-``, scalar ``*``: an empty operand needs a coefficient-bearing
  identity, which ``OperatorTerm`` can't represent natively (``append``
  rejects an empty product tuple).  ``_materialize_if_empty`` swaps the empty
  ``OperatorTerm`` for one with a single ``ElementaryOperator(jnp.eye(d_0))``
  factor on mode 0 (cuQuantum's elementary product treats unspecified modes
  as implicit identity, so a single eye is sufficient).
- ``dag()``: empty stays empty, which is correct since ``I† = I``.
"""

from __future__ import annotations

from copy import deepcopy
from math import prod

import jax.numpy as jnp
from flax import struct

# This import is what gates the entire backend — it raises ImportError when
# cuquantum is not installed; ``core/__init__.py`` catches that.
from cuquantum.densitymat.jax import (  # noqa: E402
    ElementaryOperator,
    MatrixOperator,
)

from jaxquantum.utils.cuquantum_util import OperatorTerm 

from jaxquantum.core.qarray import (  # noqa: E402
    DenseImpl,
    QarrayImpl,
    QarrayImplType,
)





# ---------------------------------------------------------------------------
# Free-function arithmetic on ``OperatorTerm``
# ---------------------------------------------------------------------------
# These mirror the ``__add__`` / ``__sub__`` / ``__matmul__`` / ``__and__`` /
# ``dag()`` methods that exist on the development version of
# ``cuquantum.densitymat.jax.OperatorTerm`` but are not in any released
# ``cuquantum-python``.  Implementing them here against the public
# ``OperatorTerm`` surface (``dims``, ``op_prods``, ``modes``, ``conjs``,
# ``duals``, ``coeffs``, ``append``) lets jaxquantum work with any cuquantum
# release that ships a basic ``OperatorTerm``.

def _cuqnt_copy_base_op(op):
    """Return a fresh ``ElementaryOperator`` / ``MatrixOperator`` wrapping the same data.

    The original instance may have an associated cuDensityMat ``_ptr`` (after
    its parent ``OperatorTerm`` has been used in an ``operator_action``) and we
    don't want the new ``OperatorTerm`` to alias that pointer.  Constructing a
    new wrapper around ``op.data`` produces an equivalent operator with
    ``_ptr=None``.  Underlying JAX array storage is shared (immutable), which
    is fine.
    """
    if isinstance(op, ElementaryOperator):
        return ElementaryOperator(op.data, diag_offsets=op.diag_offsets)
    return MatrixOperator(op.data)


def _cuqnt_check_dims(left: OperatorTerm, right: OperatorTerm, op_name: str) -> None:
    if left.dims != right.dims:
        raise ValueError(
            f"Cannot perform {op_name} between OperatorTerms with different dims: "
            f"{left.dims} vs {right.dims}"
        )


def _cuqnt_append_copied_product(out, op_prod, modes, conjs, duals, coeff):
    """Helper: dispatch on Elementary vs Matrix and append a copy of ``op_prod`` to ``out``."""
    copied = tuple(_cuqnt_copy_base_op(op) for op in op_prod)
    if isinstance(op_prod[0], ElementaryOperator):
        out.append(copied, modes=modes, duals=duals, coeff=coeff)
    else:
        out.append(copied, conjs=conjs, duals=duals, coeff=coeff)


def _cuqnt_add(left: OperatorTerm, right: OperatorTerm) -> OperatorTerm:
    """Sum of two ``OperatorTerm``s — concatenate their products into a fresh ``OperatorTerm``."""
    _cuqnt_check_dims(left, right, "addition")
    out = OperatorTerm(left.dims)
    for op_term in (left, right):
        for op_prod, modes, conjs, duals, coeff in zip(
            op_term.op_prods, op_term.modes, op_term.conjs, op_term.duals, op_term.coeffs
        ):
            _cuqnt_append_copied_product(out, op_prod, modes, conjs, duals, coeff)
    return out


def _cuqnt_sub(left: OperatorTerm, right: OperatorTerm) -> OperatorTerm:
    """Difference of two ``OperatorTerm``s — append left's products, then right's with negated coeffs."""
    _cuqnt_check_dims(left, right, "subtraction")
    out = OperatorTerm(left.dims)
    for op_term, sign in ((left, 1), (right, -1)):
        for op_prod, modes, conjs, duals, coeff in zip(
            op_term.op_prods, op_term.modes, op_term.conjs, op_term.duals, op_term.coeffs
        ):
            _cuqnt_append_copied_product(out, op_prod, modes, conjs, duals, sign * coeff)
    return out


def _cuqnt_scalar_mul(scalar, ot: OperatorTerm) -> OperatorTerm:
    """Scale every product's coefficient by ``scalar`` (Python or JAX scalar)."""
    out = OperatorTerm(ot.dims)
    
    for op_prod, modes, conjs, duals, coeff in zip(
        ot.op_prods, ot.modes, ot.conjs, ot.duals, ot.coeffs
    ):
        if jnp.iscomplexobj(scalar):
            op_prod = [ElementaryOperator(op.data + 0.0j) for op in op_prod]

        _cuqnt_append_copied_product(out, op_prod, modes, conjs, duals, scalar * coeff)
    return out


def _cuqnt_matmul(left: OperatorTerm, right: OperatorTerm) -> OperatorTerm:
    """Matrix product of two ``OperatorTerm``s — Cartesian product of their products.

    The new product is ``(*left_prod, *right_prod)``: cuDensityMat composes the
    operators in the order they appear in the product tuple, so a product
    ``[A, B]`` on the same mode acts as the matrix ``A @ B``.  Hence
    ``left @ right`` lays out ``left``'s factors before ``right``'s.
    """
    _cuqnt_check_dims(left, right, "matrix multiplication")
    out = OperatorTerm(left.dims)
    for left_prod, left_modes, left_conjs, left_duals, left_coeff in zip(
        left.op_prods, left.modes, left.conjs, left.duals, left.coeffs
    ):
        for right_prod, right_modes, right_conjs, right_duals, right_coeff in zip(
            right.op_prods, right.modes, right.conjs, right.duals, right.coeffs
        ):
            left_type = type(left_prod[0])
            right_type = type(right_prod[0])
            if left_type is not right_type:
                raise TypeError(
                    "Cannot matmul OperatorTerms with mixed ElementaryOperator and MatrixOperator products."
                )
            joined = tuple(_cuqnt_copy_base_op(op) for op in (*left_prod, *right_prod))
            coeff = left_coeff * right_coeff
            if left_type is ElementaryOperator:
                out.append(
                    joined,
                    modes=(*left_modes, *right_modes),
                    duals=(*left_duals, *right_duals),
                    coeff=coeff,
                )
            else:
                out.append(
                    joined,
                    conjs=(*left_conjs, *right_conjs),
                    duals=(*left_duals, *right_duals),
                    coeff=coeff,
                )
    return out


def _cuqnt_kron(left: OperatorTerm, right: OperatorTerm) -> OperatorTerm:
    """Tensor product of two ``OperatorTerm``s — only ``ElementaryOperator`` products supported.

    The result acts on the combined Hilbert space ``left.dims + right.dims``;
    ``right``'s mode indices are shifted by ``len(left.dims)``.
    """
    combined_dims = tuple(left.dims) + tuple(right.dims)
    n_left = len(left.dims)

    left_padded = OperatorTerm(combined_dims)
    for op_prod, modes, _conjs, duals, coeff in zip(
        left.op_prods, left.modes, left.conjs, left.duals, left.coeffs
    ):
        if not isinstance(op_prod[0], ElementaryOperator):
            raise NotImplementedError("kron is not supported for MatrixOperator products.")
        left_padded.append(
            tuple(_cuqnt_copy_base_op(op) for op in op_prod),
            modes=modes,
            duals=duals,
            coeff=coeff,
        )

    right_padded = OperatorTerm(combined_dims)
    for op_prod, modes, _conjs, duals, coeff in zip(
        right.op_prods, right.modes, right.conjs, right.duals, right.coeffs
    ):
        if not isinstance(op_prod[0], ElementaryOperator):
            raise NotImplementedError("kron is not supported for MatrixOperator products.")
        right_padded.append(
            tuple(_cuqnt_copy_base_op(op) for op in op_prod),
            modes=tuple(m + n_left for m in modes),
            duals=duals,
            coeff=coeff,
        )

    return _cuqnt_matmul(left_padded, right_padded)


def _cuqnt_dag_dense_data(data):
    """Conjugate-transpose a dense operator data tensor of shape ``[batch, *modes, *modes]``."""
    num_modes = (data.ndim - 1) // 2
    perm = (
        (0,)
        + tuple(range(num_modes + 1, 2 * num_modes + 1))
        + tuple(range(1, num_modes + 1))
    )
    return jnp.conj(jnp.transpose(data, perm))


def _cuqnt_dag(ot: OperatorTerm) -> OperatorTerm:
    """Hermitian conjugate of an ``OperatorTerm``.

    For ``ElementaryOperator`` products the operator order is preserved
    (elementary products are tensor products on disjoint modes — or commuting
    ket/bra placements via ``duals`` — so they commute).  For
    ``MatrixOperator`` products the order is reversed (matrix products are
    sequential).  Multidiagonal ``ElementaryOperator`` products raise
    ``NotImplementedError``.
    """
    out = OperatorTerm(ot.dims)
    for op_prod, modes, conjs, duals, coeff in zip(
        ot.op_prods, ot.modes, ot.conjs, ot.duals, ot.coeffs
    ):
        if isinstance(op_prod[0], ElementaryOperator):
            for base_op in op_prod:
                if base_op.diag_offsets != ():
                    raise NotImplementedError(
                        "_cuqnt_dag is not supported for multidiagonal ElementaryOperator."
                    )
            dagged = tuple(
                ElementaryOperator(_cuqnt_dag_dense_data(base_op.data)) for base_op in op_prod
            )
            out.append(dagged, modes=modes, duals=duals, coeff=coeff.conj())
        else:  # MatrixOperator: sequential, must reverse
            dagged = tuple(
                MatrixOperator(_cuqnt_dag_dense_data(mat_op.data)) for mat_op in reversed(op_prod)
            )
            out.append(
                dagged,
                conjs=tuple(reversed(conjs)),
                duals=tuple(reversed(duals)),
                coeff=coeff.conj(),
            )
    return out


# ---------------------------------------------------------------------------
# Local helpers used by ``CuquantumImpl``
# ---------------------------------------------------------------------------

def _materialize_if_empty(ot: OperatorTerm, dtype=None) -> OperatorTerm:
    """Replace an identity-encoded (empty) ``OperatorTerm`` with an explicit eye factor.

    ``OperatorTerm.append`` rejects an empty ``op_prod`` tuple, so the empty
    ``OperatorTerm`` cannot be scaled or added to anything via the native
    arithmetic.  This helper materialises it as a single-factor product
    ``[ElementaryOperator(eye(dims[0]))]`` on mode 0 — cuQuantum's elementary
    product convention treats unspecified modes as implicit identity, so this
    one factor is enough to act as the full-space identity.
    """
    if ot.op_prods:
        return ot
    out = OperatorTerm(ot.dims)
    out.append(
        [ElementaryOperator(jnp.eye(ot.dims[0], dtype=dtype or jnp.complex128))],
        modes=(0,),
        coeff=1.0,
    )
    return out


def _lift_with_mode_shift(ot: OperatorTerm, shift: int, combined_dims) -> OperatorTerm:
    """Embed ``ot`` into a larger Hilbert space with all elementary modes shifted by ``shift``."""
    out = OperatorTerm(combined_dims)
    for op_prod, modes, conjs, duals, coeff in zip(
        ot.op_prods, ot.modes, ot.conjs, ot.duals, ot.coeffs
    ):
        copied = tuple(_cuqnt_copy_base_op(op) for op in op_prod)
        if isinstance(op_prod[0], ElementaryOperator):
            out.append(copied, modes=tuple(m + shift for m in modes), duals=duals, coeff=coeff)
        else:
            out.append(copied, conjs=conjs, duals=duals, coeff=coeff)
    return out


# ---------------------------------------------------------------------------
# CuquantumImpl
# ---------------------------------------------------------------------------

@struct.dataclass
class CuquantumImpl(QarrayImpl):
    """``Qarray`` backend wrapping a cuQuantum ``OperatorTerm`` directly.

    Promotion order is the highest of all backends — mixed-type binary
    operations promote dense / sparse operands up into cuquantum.  In
    practice, mixed operations should be avoided because promoting a bare
    dense matrix into an ``OperatorTerm`` discards mode structure (see
    ``_promote_to``).
    """

    _data: OperatorTerm = struct.field(pytree_node=False)

    PROMOTION_ORDER = 3  # noqa: RUF012 — class attribute, not a struct field

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_data(cls, data) -> "CuquantumImpl":
        """Wrap *data* in a new ``CuquantumImpl``.

        Accepts:
            - ``OperatorTerm`` — used as-is.
            - 2-D dense JAX array of shape ``(n, n)`` — wrapped as a
              single-mode ``OperatorTerm`` on dims ``(n,)``.
            - Higher-rank dense input — wrapped as a single virtual mode of
              size ``n`` (the matrix dimension).  This loses mode structure
              and defeats the purpose of the backend; user code should
              build operators via ``destroy`` / ``tensor`` instead.
        """
        if isinstance(data, OperatorTerm):
            return cls(_data=data)

        arr = jnp.asarray(data)
        if arr.ndim < 2:
            raise ValueError(
                f"CuquantumImpl.from_data expects at least 2D data, got shape {arr.shape}"
            )
        n = int(arr.shape[-1])
        ot = OperatorTerm((n,))
        ot.append([ElementaryOperator(arr)], modes=(0,), coeff=1.0)
        return cls(_data=ot)

    @classmethod
    def identity_term(cls, n: int, dtype=None) -> "CuquantumImpl":
        """Identity on a single mode of size ``n`` — encoded as an empty ``OperatorTerm``."""
        return cls(_data=OperatorTerm((int(n),)))

    @classmethod
    def single_site(cls, matrix, n: int) -> "CuquantumImpl":
        """Wrap a single-site ``(n, n)`` matrix as a one-mode ``OperatorTerm``."""
        return cls.from_data(jnp.asarray(matrix))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_data(self) -> OperatorTerm:
        return self._data

    @property
    def dims(self) -> tuple:
        return self._data.dims

    def shape(self) -> tuple:
        n = prod(self._data.dims) if self._data.dims else 1
        return (n, n)

    def dtype(self):
        return self._data.dtype or jnp.complex128

    # ------------------------------------------------------------------
    # cuquantum boundary
    # ------------------------------------------------------------------

    def to_operator_term(self) -> "OperatorTerm":
        """Return the underlying ``OperatorTerm`` directly — no rebuild needed."""
        return self._data

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def add(self, other: QarrayImpl) -> QarrayImpl:
        a, b = self._coerce(other)
        if a is not self:
            return a.add(b)
        a_data = _materialize_if_empty(a._data, dtype=a.dtype())
        b_data = _materialize_if_empty(b._data, dtype=b.dtype())
        return CuquantumImpl(_data=_cuqnt_add(a_data, b_data))

    def sub(self, other: QarrayImpl) -> QarrayImpl:
        a, b = self._coerce(other)
        if a is not self:
            return a.sub(b)
        a_data = _materialize_if_empty(a._data, dtype=a.dtype())
        b_data = _materialize_if_empty(b._data, dtype=b.dtype())
        return CuquantumImpl(_data=_cuqnt_sub(a_data, b_data))

    def mul(self, scalar) -> QarrayImpl:
        data = _materialize_if_empty(self._data, dtype=self.dtype())
        return CuquantumImpl(_data=_cuqnt_scalar_mul(scalar, data))

    def matmul(self, other: QarrayImpl) -> QarrayImpl:
        a, b = self._coerce(other)
        if a is not self:
            return a.matmul(b)
        if not a._data.op_prods:  # I @ B = B
            return CuquantumImpl(_data=_lift_with_mode_shift(b._data, 0, b._data.dims))
        if not b._data.op_prods:  # A @ I = A
            return CuquantumImpl(_data=_lift_with_mode_shift(a._data, 0, a._data.dims))
        return CuquantumImpl(_data=_cuqnt_matmul(a._data, b._data))

    def kron(self, other: QarrayImpl) -> QarrayImpl:
        a, b = self._coerce(other)
        if a is not self:
            return a.kron(b)
        combined_dims = tuple(a._data.dims) + tuple(b._data.dims)
        if not a._data.op_prods:  # I_a ⊗ B: lift B with shifted modes
            return CuquantumImpl(
                _data=_lift_with_mode_shift(b._data, len(a._data.dims), combined_dims)
            )
        if not b._data.op_prods:  # A ⊗ I_b: lift A keeping modes
            return CuquantumImpl(_data=_lift_with_mode_shift(a._data, 0, combined_dims))
        return CuquantumImpl(_data=_cuqnt_kron(a._data, b._data))

    def dag(self) -> QarrayImpl:
        # Empty OperatorTerm stays empty under dag(), which matches I† = I.
        return CuquantumImpl(_data=_cuqnt_dag(self._data))

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_dense(self) -> "DenseImpl":
        """Build the dense matrix term-by-term.

        Used for testing / debugging.  Defeats the backend's purpose — emit
        each term as a Kronecker product of single-site matrices on the
        right modes (identities elsewhere), multiply by the coefficient,
        and sum.
        """
        full_dim = prod(self._data.dims) if self._data.dims else 1
        dtype = self.dtype()

        # Empty OperatorTerm represents identity per the jaxquantum convention.
        if not self._data.op_prods:
            return DenseImpl(_data=jnp.eye(full_dim, dtype=dtype))

        result = jnp.zeros((full_dim, full_dim), dtype=dtype)

        for op_prod, modes, coeff_arr in zip(
            self._data.op_prods, self._data.modes, self._data.coeffs
        ):
            # cuquantum's ``append([A, B], modes=[i, i], ...)`` composes A and
            # B *in order* — equivalent to A @ B on that mode.  Pre-compose
            # any per-mode multi-factor product into per-mode matrices.
            mode_to_matrix = {}
            idx = 0
            for elem_op in op_prod:
                # ElementaryOperator data is shape ``[batch=1, *mode_extents,
                # *mode_extents]``; squeeze the batch axis for the dense case.
                mat = jnp.squeeze(elem_op.data, axis=0)
                # Each elementary op may itself act on multiple modes; for our
                # dense round-trip we only support the single-mode case (which
                # is what the rest of jaxquantum builds anyway).
                if elem_op.num_modes != 1:
                    raise NotImplementedError(
                        "to_dense() supports only single-mode ElementaryOperators."
                    )
                m = int(modes[idx])
                if m in mode_to_matrix:
                    mode_to_matrix[m] = mode_to_matrix[m] @ mat
                else:
                    mode_to_matrix[m] = mat
                idx += 1

            # Build Kronecker product over all modes (identity where unused).
            term_matrix = None
            for i, dim in enumerate(self._data.dims):
                factor = mode_to_matrix.get(i, jnp.eye(dim, dtype=dtype))
                term_matrix = (
                    factor if term_matrix is None
                    else jnp.kron(term_matrix, factor)
                )
            if term_matrix is None:  # zero-mode degenerate case
                term_matrix = jnp.eye(1, dtype=dtype)

            # Coefficient is a length-1 jax array post-``append``; squeeze.
            coeff = coeff_arr[0] if coeff_arr.ndim else coeff_arr
            result = result + coeff * term_matrix

        return DenseImpl(_data=result)

    def to_sparse_bcoo(self):
        from jaxquantum.core.sparse_bcoo import SparseBCOOImpl
        return SparseBCOOImpl.from_data(self.to_dense()._data)

    def to_sparse_dia(self):
        from jaxquantum.core.sparse_dia import SparseDiaImpl
        return SparseDiaImpl.from_data(self.to_dense()._data)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __deepcopy__(self, memo=None):
        # Rebuild an equivalent OperatorTerm using the public surface so we
        # don't depend on ``OperatorTerm._copy()`` (private API).
        return CuquantumImpl(
            _data=_lift_with_mode_shift(self._data, 0, self._data.dims)
        )

    def tidy_up(self, atol):
        # Coefficient-level thresholding without densifying would change
        # the term list ambiguously (which terms count as "small"?).  The
        # safe choice is a no-op — densify first if you need to clean up.
        return self

    @classmethod
    def _eye_data(cls, n: int, dtype=None) -> "OperatorTerm":
        """Identity-on-mode-of-size-``n`` data: an empty ``OperatorTerm((n,))``."""
        return OperatorTerm((int(n),))

    @classmethod
    def can_handle_data(cls, arr) -> bool:
        return isinstance(arr, OperatorTerm) # or getattr(arr, "_is_cuquantum_op", False)

    @classmethod
    def dag_data(cls, arr) -> "OperatorTerm":
        """Conjugate transpose of a raw ``OperatorTerm``.

        Defensive — the cuquantum solver path doesn't go through ``dag_data``
        because ``c_ops_dag`` is computed only inside ``_mesolve_data``.
        """
        if not isinstance(arr, OperatorTerm):
            raise TypeError(
                f"CuquantumImpl.dag_data expects OperatorTerm, got {type(arr)}"
            )
        return _cuqnt_dag(arr)

    def _promote_to(self, target_cls):
        """Refuse silent promotion *into* ``CuquantumImpl`` from a bare matrix.

        Promoting a dense or sparse impl into cuquantum without ``dims``
        information would wrap a potentially huge matrix as a single-mode
        ``ElementaryOperator``, which defeats the backend's purpose.  We
        densify on the way out (so cuquantum results can be consumed by
        dense/sparse callers) but rely on user code to build cuquantum
        operators via ``destroy(d, implementation="cuquantum")`` + ``tensor``
        when they want the structured representation.
        """
        if isinstance(self, target_cls):
            return self
        if target_cls is CuquantumImpl:
            return target_cls.from_data(self.to_dense()._data)
        # Promoting *out* of cuquantum: densify and re-wrap.
        return target_cls.from_data(self.to_dense()._data)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

QarrayImplType.register(CuquantumImpl, QarrayImplType.CUQUANTUM)


__all__ = ["CuquantumImpl"]

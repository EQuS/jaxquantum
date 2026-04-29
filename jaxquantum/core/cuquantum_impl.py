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
    OperatorTerm,
)

from jaxquantum.core.qarray import (  # noqa: E402
    DenseImpl,
    QarrayImpl,
    QarrayImplType,
)


# ---------------------------------------------------------------------------
# Helpers
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
    for op_prod, modes, conjs, duals, coeff, op_type in zip(
        ot.op_prods, ot.modes, ot.conjs, ot.duals, ot.coeffs, ot._op_prod_types
    ):
        copied = tuple(op._copy() for op in op_prod)
        if op_type is ElementaryOperator:
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
        if a._data.dims != b._data.dims:
            raise ValueError(
                f"Cannot add CuquantumImpls with different dims: "
                f"{a._data.dims} vs {b._data.dims}"
            )
        a_data = _materialize_if_empty(a._data, dtype=a.dtype())
        b_data = _materialize_if_empty(b._data, dtype=b.dtype())
        return CuquantumImpl(_data=a_data + b_data)

    def sub(self, other: QarrayImpl) -> QarrayImpl:
        a, b = self._coerce(other)
        if a is not self:
            return a.sub(b)
        if a._data.dims != b._data.dims:
            raise ValueError(
                f"Cannot subtract CuquantumImpls with different dims: "
                f"{a._data.dims} vs {b._data.dims}"
            )
        a_data = _materialize_if_empty(a._data, dtype=a.dtype())
        b_data = _materialize_if_empty(b._data, dtype=b.dtype())
        return CuquantumImpl(_data=a_data - b_data)

    def mul(self, scalar) -> QarrayImpl:
        data = _materialize_if_empty(self._data, dtype=self.dtype())
        return CuquantumImpl(_data=scalar * data)

    def matmul(self, other: QarrayImpl) -> QarrayImpl:
        a, b = self._coerce(other)
        if a is not self:
            return a.matmul(b)
        if a._data.dims != b._data.dims:
            raise ValueError(
                f"Cannot matmul CuquantumImpls with different dims: "
                f"{a._data.dims} vs {b._data.dims}"
            )
        if not a._data.op_prods:  # I @ B = B
            return CuquantumImpl(_data=b._data._copy())
        if not b._data.op_prods:  # A @ I = A
            return CuquantumImpl(_data=a._data._copy())
        return CuquantumImpl(_data=a._data @ b._data)

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
        return CuquantumImpl(_data=a._data & b._data)

    def dag(self) -> QarrayImpl:
        # Empty OperatorTerm stays empty under dag(), which matches I† = I.
        return CuquantumImpl(_data=self._data.dag())

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
        # OperatorTerm has its own _copy() that handles the JAX-array bookkeeping.
        return CuquantumImpl(_data=self._data._copy())

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
        return isinstance(arr, OperatorTerm) or getattr(arr, "_is_cuquantum_op", False)

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
        return arr.dag()

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

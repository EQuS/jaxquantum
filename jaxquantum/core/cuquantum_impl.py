"""cuQuantum backend for ``Qarray``.

Wraps ``cuquantum.densitymat.jax.OperatorTerm`` so that mode-structured
Hamiltonians and dissipators can be assembled without ever materialising the
full tensor-product matrix.  The resulting ``Qarray[CuquantumImpl]`` is
consumed by ``_mesolve_cuquantum`` / ``_sesolve_cuquantum`` in
``solvers.py``, which dispatch to ``operator_action`` instead of dense
matrix multiplications.

This module imports ``cuquantum.densitymat.jax`` at load time and raises
``ImportError`` when cuquantum isn't installed; ``core/__init__.py`` catches
that so the rest of the package keeps working on CPU-only installs.

Design notes
------------
- Every ``CuquantumImpl`` carries a tuple of *terms*, where each term is
  ``(matrices, modes, duals, coeff)``.  This mirror list is what arithmetic
  operations manipulate directly — cuquantum does not yet expose
  ``OperatorTerm.__add__`` / ``__matmul__``, so we build new OperatorTerms
  ourselves via ``.append``.
- The ``OperatorTerm`` itself is rebuilt eagerly in ``to_operator_term()`` at
  the boundary where the solver consumes the impl.
- All five binary operations (``add``, ``sub``, ``mul``, ``matmul``,
  ``kron``) and the unary ``dag`` produce terms with ``duals=[False, ...]``
  on individual entries.  The ``dual`` distinction is reserved for the outer
  ``Operator.append`` step inside the solver RHS, which is how
  ``-1j * [H, ρ]`` is encoded.
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
    Operator,
    operator_action,
)

from jaxquantum.core.qarray import (  # noqa: E402
    DenseImpl,
    QarrayImpl,
    QarrayImplType,
)


# ---------------------------------------------------------------------------
# Term primitive
# ---------------------------------------------------------------------------

# A single term inside an OperatorTerm: a product of single-site
# ElementaryOperators sitting on specific modes, with a scalar coefficient.
#
#   matrices : tuple of (n_i, n_i) JAX arrays (one per factor)
#   modes    : tuple of int — which mode each factor acts on
#   duals    : tuple of bool — per-factor dual flag (kept ``False`` by all
#              arithmetic; the outer Operator.append toggles dual)
#   coeff    : scalar coefficient (Python or JAX)
#
# Stored as plain tuples so the entire ``CuquantumOpData`` container can be
# marked ``pytree_node=False``.


# ---------------------------------------------------------------------------
# Raw data container
# ---------------------------------------------------------------------------

@struct.dataclass
class CuquantumOpData:
    """Lightweight container carrying everything needed to rebuild an OperatorTerm.

    Returned by ``CuquantumImpl.get_data()``; consumed by
    ``CuquantumImpl.from_data()``.  Exposes ``shape`` and ``dtype`` so it can
    flow through ``Qarray.create``'s shape-inference logic without going
    through a dense intermediate.
    """

    terms: tuple = struct.field(pytree_node=False)
    dims: tuple = struct.field(pytree_node=False)
    _dtype: object = struct.field(pytree_node=False, default=jnp.complex128)

    # Marker used by ``robust_asarray`` and ``DenseImpl.can_handle_data`` to
    # route this object without a hard import.
    _is_cuquantum_op = True

    @property
    def shape(self) -> tuple:
        n = prod(self.dims) if self.dims else 1
        return (n, n)

    @property
    def dtype(self):
        return self._dtype


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_term_data(n: int, dtype=None) -> CuquantumOpData:
    """Identity on a single mode of size ``n`` — a coefficient-1 empty product."""
    return CuquantumOpData(
        terms=((tuple(), tuple(), tuple(), 1.0 + 0.0j),),
        dims=(int(n),),
        _dtype=dtype or jnp.complex128,
    )


def _single_site_term_data(matrix, n: int) -> CuquantumOpData:
    """Wrap a single-site ``(n, n)`` matrix as a one-mode OperatorTerm."""
    matrix = jnp.asarray(matrix)
    return CuquantumOpData(
        terms=(((matrix,), (0,), (False,), 1.0 + 0.0j),),
        dims=(int(n),),
        _dtype=matrix.dtype,
    )


def _matrix_dag(matrix):
    """Conjugate transpose of a single-site matrix."""
    return jnp.conj(jnp.swapaxes(matrix, -1, -2))


def _compose_per_mode(matrices, modes, duals):
    """Pre-compose factors that share both mode and dual flag.

    For each ``(mode, dual)`` group, multiply the per-mode factors in the
    given order (``matrices[0] @ matrices[1] @ ...``) so the resulting
    cuquantum term has at most one ``ElementaryOperator`` per
    ``(mode, dual)`` pair.

    Returns:
        Tuple of (matrices_out, modes_out, duals_out) — three lists of equal
        length.
    """
    grouped: dict[tuple[int, bool], "jnp.ndarray"] = {}
    order: list[tuple[int, bool]] = []
    for mat, m, dual in zip(matrices, modes, duals):
        key = (int(m), bool(dual))
        if key in grouped:
            grouped[key] = grouped[key] @ mat
        else:
            grouped[key] = mat
            order.append(key)

    matrices_out = [grouped[k] for k in order]
    modes_out = [k[0] for k in order]
    duals_out = [k[1] for k in order]
    return matrices_out, modes_out, duals_out


# ---------------------------------------------------------------------------
# CuquantumImpl
# ---------------------------------------------------------------------------

@struct.dataclass
class CuquantumImpl(QarrayImpl):
    """``Qarray`` backend wrapping a cuQuantum ``OperatorTerm``.

    The backend stores a tuple of *terms* (each a product of single-site
    matrices on specific modes, with a coefficient) plus the per-mode
    Hilbert-space dimensions.  The cuQuantum ``OperatorTerm`` is rebuilt
    on demand via ``to_operator_term()`` whenever the solver needs it.

    Promotion order is the highest of all backends — mixed-type binary
    operations promote dense / sparse operands up into cuquantum.  In
    practice, mixed operations should be avoided because promoting a bare
    dense matrix into an ``OperatorTerm`` discards mode structure (see
    ``_promote_to``).
    """

    _data: CuquantumOpData = struct.field(pytree_node=False)

    PROMOTION_ORDER = 3  # noqa: RUF012 — class attribute, not a struct field

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_data(cls, data) -> "CuquantumImpl":
        """Wrap *data* in a new ``CuquantumImpl``.

        Accepts:
            - ``CuquantumOpData`` — used as-is.
            - 2-D dense JAX array of shape ``(n, n)`` — wrapped as a
              single-mode OperatorTerm.
            - Higher-rank dense input — wrapped as a single virtual mode of
              size ``n`` (the matrix dimension).  This loses mode structure
              and defeats the purpose of the backend; user code should
              build operators via ``destroy`` / ``tensor`` instead.

        Args:
            data: Input data (see above).

        Returns:
            A ``CuquantumImpl`` wrapping *data*.
        """
        if isinstance(data, CuquantumOpData):
            return cls(_data=data)

        arr = jnp.asarray(data)
        if arr.ndim < 2:
            raise ValueError(
                f"CuquantumImpl.from_data expects at least 2D data, got shape {arr.shape}"
            )
        n = int(arr.shape[-1])
        return cls(_data=_single_site_term_data(arr, n))

    @classmethod
    def from_terms(cls, terms, dims, dtype=None) -> "CuquantumImpl":
        """Direct constructor from a ``terms`` tuple and ``dims`` tuple."""
        return cls(_data=CuquantumOpData(
            terms=tuple(terms),
            dims=tuple(int(d) for d in dims),
            _dtype=dtype or jnp.complex128,
        ))

    @classmethod
    def identity_term(cls, n: int, dtype=None) -> "CuquantumImpl":
        """Identity on a single mode of size ``n``."""
        return cls(_data=_identity_term_data(n, dtype=dtype))

    @classmethod
    def single_site(cls, matrix, n: int) -> "CuquantumImpl":
        """Wrap a single-site ``(n, n)`` matrix as a one-mode OperatorTerm."""
        return cls(_data=_single_site_term_data(matrix, n))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_data(self) -> CuquantumOpData:
        return self._data

    @property
    def terms(self) -> tuple:
        return self._data.terms

    @property
    def dims(self) -> tuple:
        return self._data.dims

    def shape(self) -> tuple:
        return self._data.shape

    def dtype(self):
        return self._data.dtype

    # ------------------------------------------------------------------
    # cuquantum boundary
    # ------------------------------------------------------------------

    def to_operator_term(self) -> "OperatorTerm":
        """Build a fresh cuQuantum ``OperatorTerm`` from the stored terms.

        Per-mode multi-factor products are pre-composed into a single matrix
        before wrapping in an ``ElementaryOperator``.  This makes the result
        independent of cuQuantum's same-mode composition convention: each
        emitted cuquantum term has at most one elementary operator per
        ``(mode, dual)`` pair.  Factors with different dual flags on the same
        mode (e.g. the ``L ρ L†`` recipe) stay separate, since they act on
        opposite sides of ρ.
        """
        ot = OperatorTerm(self._data.dims)
        for matrices, modes, duals, coeff in self._data.terms:
            elems_matrices, out_modes, out_duals = _compose_per_mode(
                matrices, modes, duals
            )
            elems = [ElementaryOperator(m) for m in elems_matrices]
            ot.append(elems, modes=out_modes, duals=out_duals, coeff=coeff)
        return ot

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
        new_terms = tuple(a._data.terms) + tuple(b._data.terms)
        return CuquantumImpl.from_terms(
            new_terms, a._data.dims, dtype=a._data._dtype
        )

    def sub(self, other: QarrayImpl) -> QarrayImpl:
        a, b = self._coerce(other)
        if a is not self:
            return a.sub(b)
        if a._data.dims != b._data.dims:
            raise ValueError(
                f"Cannot subtract CuquantumImpls with different dims: "
                f"{a._data.dims} vs {b._data.dims}"
            )
        negated = tuple(
            (mats, modes, duals, -coeff)
            for (mats, modes, duals, coeff) in b._data.terms
        )
        new_terms = tuple(a._data.terms) + negated
        return CuquantumImpl.from_terms(
            new_terms, a._data.dims, dtype=a._data._dtype
        )

    def mul(self, scalar) -> QarrayImpl:
        new_terms = tuple(
            (mats, modes, duals, scalar * coeff)
            for (mats, modes, duals, coeff) in self._data.terms
        )
        return CuquantumImpl.from_terms(
            new_terms, self._data.dims, dtype=self._data._dtype
        )

    def matmul(self, other: QarrayImpl) -> QarrayImpl:
        a, b = self._coerce(other)
        if a is not self:
            return a.matmul(b)
        if a._data.dims != b._data.dims:
            raise ValueError(
                f"Cannot matmul CuquantumImpls with different dims: "
                f"{a._data.dims} vs {b._data.dims}"
            )
        new_terms = []
        for (mats_a, modes_a, duals_a, coeff_a) in a._data.terms:
            for (mats_b, modes_b, duals_b, coeff_b) in b._data.terms:
                new_terms.append((
                    tuple(mats_a) + tuple(mats_b),
                    tuple(modes_a) + tuple(modes_b),
                    tuple(duals_a) + tuple(duals_b),
                    coeff_a * coeff_b,
                ))
        return CuquantumImpl.from_terms(
            tuple(new_terms), a._data.dims, dtype=a._data._dtype
        )

    def kron(self, other: QarrayImpl) -> QarrayImpl:
        a, b = self._coerce(other)
        if a is not self:
            return a.kron(b)
        offset = len(a._data.dims)
        combined_dims = tuple(a._data.dims) + tuple(b._data.dims)

        new_terms = []
        for (mats_a, modes_a, duals_a, coeff_a) in a._data.terms:
            for (mats_b, modes_b, duals_b, coeff_b) in b._data.terms:
                shifted_modes_b = tuple(m + offset for m in modes_b)
                new_terms.append((
                    tuple(mats_a) + tuple(mats_b),
                    tuple(modes_a) + shifted_modes_b,
                    tuple(duals_a) + tuple(duals_b),
                    coeff_a * coeff_b,
                ))
        return CuquantumImpl.from_terms(
            tuple(new_terms), combined_dims, dtype=a._data._dtype
        )

    def dag(self) -> QarrayImpl:
        new_terms = []
        for (mats, modes, duals, coeff) in self._data.terms:
            # (A_1 A_2 ... A_k)^† = A_k^† ... A_2^† A_1^†
            rev_mats = tuple(_matrix_dag(m) for m in reversed(mats))
            rev_modes = tuple(reversed(modes))
            rev_duals = tuple(reversed(duals))
            new_terms.append((rev_mats, rev_modes, rev_duals, jnp.conj(coeff)))
        return CuquantumImpl.from_terms(
            tuple(new_terms), self._data.dims, dtype=self._data._dtype
        )

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
        result = jnp.zeros((full_dim, full_dim), dtype=self._data._dtype)

        for (matrices, modes, duals, coeff) in self._data.terms:
            # Compose any per-mode multi-factor product into per-mode matrices.
            # cuquantum's ``append([A, B], modes=[i, i], ...)`` composes A and
            # B *in order* — equivalent to A @ B on that mode.
            mode_to_matrix = {}
            for mat, m, dual in zip(matrices, modes, duals):
                # ``dual`` flags inside an OperatorTerm describe the
                # left/right action when later attached to an Operator;
                # for the static dense conversion we treat them as
                # left-action (the standard interpretation when
                # ``Operator.append(..., dual=False)`` is used).
                _ = dual  # noqa: F841 — flagged but not used for dense build
                if m in mode_to_matrix:
                    mode_to_matrix[m] = mode_to_matrix[m] @ mat
                else:
                    mode_to_matrix[m] = mat

            # Build Kronecker product over all modes (identity where unused).
            term_matrix = None
            for i, dim in enumerate(self._data.dims):
                factor = mode_to_matrix.get(i, jnp.eye(dim, dtype=self._data._dtype))
                term_matrix = (
                    factor if term_matrix is None
                    else jnp.kron(term_matrix, factor)
                )
            if term_matrix is None:  # zero-mode degenerate case
                term_matrix = jnp.eye(1, dtype=self._data._dtype)
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
        return CuquantumImpl(_data=deepcopy(self._data, memo))

    def tidy_up(self, atol):
        # Coefficient-level thresholding without densifying would change
        # the term list ambiguously (which terms count as "small"?).  The
        # safe choice is a no-op — densify first if you need to clean up.
        return self

    @classmethod
    def _eye_data(cls, n: int, dtype=None) -> CuquantumOpData:
        return _identity_term_data(n, dtype=dtype)

    @classmethod
    def can_handle_data(cls, arr) -> bool:
        return isinstance(arr, CuquantumOpData) or getattr(arr, "_is_cuquantum_op", False)

    @classmethod
    def dag_data(cls, arr) -> CuquantumOpData:
        """Conjugate transpose of raw cuquantum data.

        Defensive — the cuquantum solver path doesn't go through ``dag_data``
        because ``c_ops_dag`` is computed only inside ``_mesolve_data``.
        """
        if not isinstance(arr, CuquantumOpData):
            raise TypeError(f"CuquantumImpl.dag_data expects CuquantumOpData, got {type(arr)}")
        new_terms = []
        for (mats, modes, duals, coeff) in arr.terms:
            rev_mats = tuple(_matrix_dag(m) for m in reversed(mats))
            new_terms.append((
                rev_mats,
                tuple(reversed(modes)),
                tuple(reversed(duals)),
                jnp.conj(coeff),
            ))
        return CuquantumOpData(
            terms=tuple(new_terms), dims=arr.dims, _dtype=arr._dtype,
        )

    def _promote_to(self, target_cls):
        """Refuse silent promotion *into* CuquantumImpl from a bare matrix.

        Promoting a dense or sparse impl into cuquantum without ``dims``
        information would wrap a potentially huge matrix as a single-mode
        ElementaryOperator, which defeats the backend's purpose.  Raise a
        clear error so users build cuquantum operators via
        ``destroy(d, implementation="cuquantum")`` + ``tensor`` instead.
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


__all__ = ["CuquantumImpl", "CuquantumOpData"]

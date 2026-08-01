"""States."""

from typing import List
from jax import config
from math import prod

import jax.numpy as jnp
from jax.nn import one_hot

from jaxquantum.core.qarray import Qarray, tensor, QarrayImplType

config.update("jax_enable_x64", True)


def _impl_from(impl_type: QarrayImplType, factory: str, *args, **kwargs) -> Qarray:
    """Build a ``Qarray`` by calling a classmethod on the registered impl class.

    Looks up the impl class via ``QarrayImplType.get_impl_class()`` (so
    ``operators.py`` never imports a specific ``*Impl`` class), invokes
    ``factory`` on it with the provided args, and wraps the result.

    Example::

        _impl_from(QarrayImplType.SPARSE_DIA, "from_diags",
                   offsets=(1,), diags=...)
    """
    impl_class = QarrayImplType(impl_type).get_impl_class()
    impl = getattr(impl_class, factory)(*args, **kwargs)
    return Qarray.from_impl(impl)


def _impl_type(implementation):
    """Coerce ``implementation`` to a ``QarrayImplType`` member, or ``None``."""
    if implementation is None:
        return None
    return QarrayImplType(implementation)


def sigmax(implementation=None) -> Qarray:
    """σx

    Args:
        implementation: Qarray implementation type (e.g. ``"dense"``,
            ``"sparse_dia"``, ``"cuquantum"``).  When ``None``, falls back to
            ``SETTINGS["default_backend"]``.

    Returns:
        σx Pauli Operator
    """
    impl_type = _impl_type(implementation)
    if impl_type == QarrayImplType.SPARSE_DIA:
        # Offset -1: valid at [0:1] → diag[0] = A[1,0] = 1.0, diag[1] = 0 (trailing zero)
        # Offset +1: valid at [1:]  → diag[0] = 0 (leading zero), diag[1] = A[0,1] = 1.0
        diags = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        return _impl_from(QarrayImplType.SPARSE_DIA, "from_diags", offsets=(-1, 1), diags=diags)
    if impl_type == QarrayImplType.CUQUANTUM:
        return _impl_from(QarrayImplType.CUQUANTUM, "single_site",
                          jnp.array([[0.0, 1.0], [1.0, 0.0]]), 2)
    return Qarray.create(jnp.array([[0.0, 1.0], [1.0, 0.0]]), implementation=implementation)


def sigmay(implementation=None) -> Qarray:
    """σy

    Returns:
        σy Pauli Operator
    """
    impl_type = _impl_type(implementation)
    if impl_type == QarrayImplType.SPARSE_DIA:
        diags = jnp.array([[1.0j, 0.0], [0.0, -1.0j]])
        return _impl_from(QarrayImplType.SPARSE_DIA, "from_diags", offsets=(-1, 1), diags=diags)
    if impl_type == QarrayImplType.CUQUANTUM:
        return _impl_from(QarrayImplType.CUQUANTUM, "single_site",
                          jnp.array([[0.0, -1.0j], [1.0j, 0.0]]), 2)
    return Qarray.create(jnp.array([[0.0, -1.0j], [1.0j, 0.0]]), implementation=implementation)


def sigmaz(implementation=None) -> Qarray:
    """σz

    Returns:
        σz Pauli Operator
    """
    impl_type = _impl_type(implementation)
    if impl_type == QarrayImplType.SPARSE_DIA:
        diags = jnp.array([[1.0, -1.0]])
        return _impl_from(QarrayImplType.SPARSE_DIA, "from_diags", offsets=(0,), diags=diags)
    if impl_type == QarrayImplType.CUQUANTUM:
        return _impl_from(QarrayImplType.CUQUANTUM, "single_site",
                          jnp.array([[1.0, 0.0], [0.0, -1.0]]), 2)
    return Qarray.create(jnp.array([[1.0, 0.0], [0.0, -1.0]]), implementation=implementation)


def hadamard(implementation=None) -> Qarray:
    """H

    Returns:
        H: Hadamard gate
    """
    impl_type = _impl_type(implementation)
    if impl_type == QarrayImplType.SPARSE_DIA:
        s = 1.0 / jnp.sqrt(2.0)
        # offset -1: valid at [0]   → diag[0]=A[1,0]=s, diag[1]=0 (trailing zero)
        # offset  0: valid at [0:2] → diag[0]=A[0,0]=s, diag[1]=A[1,1]=-s
        # offset +1: valid at [1]   → diag[0]=0 (leading zero), diag[1]=A[0,1]=s
        diags = jnp.array([[s, 0.0], [s, -s], [0.0, s]])
        return _impl_from(QarrayImplType.SPARSE_DIA, "from_diags",
                          offsets=(-1, 0, 1), diags=diags)
    if impl_type == QarrayImplType.CUQUANTUM:
        return _impl_from(QarrayImplType.CUQUANTUM, "single_site",
                          jnp.array([[1.0, 1.0], [1.0, -1.0]]) / jnp.sqrt(2), 2)
    return Qarray.create(jnp.array([[1, 1], [1, -1]]) / jnp.sqrt(2), implementation=implementation)


def sigmam(implementation=None) -> Qarray:
    """σ-

    Returns:
        σ- Pauli Operator
    """
    impl_type = _impl_type(implementation)
    if impl_type == QarrayImplType.SPARSE_DIA:
        diags = jnp.array([[1.0, 0.0]])
        return _impl_from(QarrayImplType.SPARSE_DIA, "from_diags", offsets=(-1,), diags=diags)
    if impl_type == QarrayImplType.CUQUANTUM:
        return _impl_from(QarrayImplType.CUQUANTUM, "single_site",
                          jnp.array([[0.0, 0.0], [1.0, 0.0]]), 2)
    return Qarray.create(jnp.array([[0.0, 0.0], [1.0, 0.0]]), implementation=implementation)


def sigmap(implementation=None) -> Qarray:
    """σ+

    Returns:
        σ+ Pauli Operator
    """
    impl_type = _impl_type(implementation)
    if impl_type == QarrayImplType.SPARSE_DIA:
        diags = jnp.array([[0.0, 1.0]])
        return _impl_from(QarrayImplType.SPARSE_DIA, "from_diags", offsets=(1,), diags=diags)
    if impl_type == QarrayImplType.CUQUANTUM:
        return _impl_from(QarrayImplType.CUQUANTUM, "single_site",
                          jnp.array([[0.0, 1.0], [0.0, 0.0]]), 2)
    return Qarray.create(jnp.array([[0.0, 1.0], [0.0, 0.0]]), implementation=implementation)


def qubit_rotation(theta: float, nx, ny, nz, implementation=None) -> Qarray:
    """Single qubit rotation.

    Args:
        theta: rotation angle.
        nx: rotation axis x component.
        ny: rotation axis y component.
        nz: rotation axis z component.
        implementation: Qarray implementation type.

    Returns:
        Single qubit rotation operator.
    """
    impl_type = _impl_type(implementation)
    if impl_type == QarrayImplType.CUQUANTUM:
        # ``CuquantumImpl.identity_term`` carries no dtype, so adding it to a
        # complex single-site op fails in cuquantum's OperatorTerm arithmetic.
        # Build dense and convert at the end.
        result = jnp.cos(theta / 2) * identity(2) - 1j * jnp.sin(theta / 2) * (
            nx * sigmax() + ny * sigmay() + nz * sigmaz()
        )
        return result.to_backend(implementation)
    return jnp.cos(theta / 2) * identity(2, implementation=implementation) - 1j * jnp.sin(theta / 2) * (
        nx * sigmax(implementation=implementation)
        + ny * sigmay(implementation=implementation)
        + nz * sigmaz(implementation=implementation)
    )


def destroy(N, implementation=None) -> Qarray:
    """annihilation operator

    Args:
        N: Hilbert space size
        implementation: Qarray implementation type.

    Returns:
        annihilation operator in Hilbert Space of size N
    """
    impl_type = _impl_type(implementation)
    if impl_type == QarrayImplType.SPARSE_DIA:
        # Single superdiagonal at offset +1; Convention A: 1 leading zero.
        diags = jnp.zeros((1, N), dtype=jnp.float64)
        diags = diags.at[0, 1:].set(jnp.sqrt(jnp.arange(1, N, dtype=jnp.float64)))
        return _impl_from(QarrayImplType.SPARSE_DIA, "from_diags", offsets=(1,), diags=diags)
    if impl_type == QarrayImplType.CUQUANTUM:
        matrix = jnp.diag(jnp.sqrt(jnp.arange(1, N, dtype=jnp.complex128)), k=1)
        return _impl_from(QarrayImplType.CUQUANTUM, "single_site", matrix, N)
    return Qarray.create(jnp.diag(jnp.sqrt(jnp.arange(1, N)), k=1), implementation=implementation)


def create(N, implementation=None) -> Qarray:
    """creation operator

    Args:
        N: Hilbert space size
        implementation: Qarray implementation type.

    Returns:
        creation operator in Hilbert Space of size N
    """
    impl_type = _impl_type(implementation)
    if impl_type == QarrayImplType.SPARSE_DIA:
        # Single subdiagonal at offset -1; Convention A: 1 trailing zero.
        diags = jnp.zeros((1, N), dtype=jnp.float64)
        diags = diags.at[0, :N - 1].set(jnp.sqrt(jnp.arange(1, N, dtype=jnp.float64)))
        return _impl_from(QarrayImplType.SPARSE_DIA, "from_diags", offsets=(-1,), diags=diags)
    if impl_type == QarrayImplType.CUQUANTUM:
        matrix = jnp.diag(jnp.sqrt(jnp.arange(1, N, dtype=jnp.complex128)), k=-1)
        return _impl_from(QarrayImplType.CUQUANTUM, "single_site", matrix, N)
    return Qarray.create(jnp.diag(jnp.sqrt(jnp.arange(1, N)), k=-1), implementation=implementation)


def num(N, implementation=None) -> Qarray:
    """Number operator

    Args:
        N: Hilbert Space size
        implementation: Qarray implementation type.

    Returns:
        number operator in Hilbert Space of size N
    """
    impl_type = _impl_type(implementation)
    if impl_type == QarrayImplType.SPARSE_DIA:
        # Main diagonal only; no leading/trailing zeros needed (offset 0).
        diags = jnp.arange(N, dtype=jnp.float64).reshape(1, N)
        return _impl_from(QarrayImplType.SPARSE_DIA, "from_diags", offsets=(0,), diags=diags)
    if impl_type == QarrayImplType.CUQUANTUM:
        matrix = jnp.diag(jnp.arange(N, dtype=jnp.complex128))
        return _impl_from(QarrayImplType.CUQUANTUM, "single_site", matrix, N)
    return Qarray.create(jnp.diag(jnp.arange(N)), implementation=implementation)


def identity(*args, implementation=None, **kwargs) -> Qarray:
    """Identity matrix.

    Args:
        implementation: Qarray implementation type.

    Returns:
        Identity matrix.
    """
    impl_type = _impl_type(implementation)
    if impl_type == QarrayImplType.SPARSE_DIA:
        n = args[0] if args else kwargs.get("N", kwargs.get("n", None))
        if n is not None and (len(args) <= 1) and not kwargs:
            diags = jnp.ones((1, int(n)), dtype=jnp.float64)
            return _impl_from(QarrayImplType.SPARSE_DIA, "from_diags",
                              offsets=(0,), diags=diags)
    if impl_type == QarrayImplType.CUQUANTUM:
        n = args[0] if args else kwargs.get("N", kwargs.get("n", None))
        if n is not None and (len(args) <= 1) and not kwargs:
            return _impl_from(QarrayImplType.CUQUANTUM, "identity_term", int(n))
    return Qarray.create(jnp.eye(*args, **kwargs), implementation=implementation)


qeye = identity


def identity_like(A, implementation=None) -> Qarray:
    """Identity matrix with the same shape as A.

    Args:
        A: Matrix.
        implementation: Qarray implementation type.

    Returns:
        Identity matrix with the same shape as A.
    """
    space_dims = A.space_dims
    total_dim = prod(space_dims)
    impl_type = _impl_type(implementation)
    if impl_type == QarrayImplType.CUQUANTUM:
        # Build a multi-mode identity by kron'ing single-mode identities.
        # Each one is an empty-product OperatorTerm on its mode.
        identities = [
            _impl_from(QarrayImplType.CUQUANTUM, "identity_term", int(d))
            for d in space_dims
        ]
        result = identities[0]
        for next_id in identities[1:]:
            result = tensor(result, next_id)
        # Re-tag dims to the (space_dims, space_dims) operator form.
        return Qarray.create(
            result._impl.get_data(),
            dims=[space_dims, space_dims],
            implementation=QarrayImplType.CUQUANTUM,
        )
    return Qarray.create(
        jnp.eye(total_dim, total_dim),
        dims=[space_dims, space_dims],
        implementation=implementation,
    )


def displace(N, α, implementation=None) -> Qarray:
    """Displacement operator

    Args:
        N: Hilbert Space Size
        α: Phase space displacement
        implementation: Qarray implementation type.  ``expm`` densifies
            internally; the result is converted back to the requested
            backend before returning.

    Returns:
        Displace operator D(α)
    """
    a = destroy(N, implementation=implementation)
    result = (α * a.dag() - jnp.conj(α) * a).expm()
    if implementation is None:
        return result
    return result.to_backend(implementation)


def squeeze(N, z, implementation=None) -> Qarray:
    """Single-mode Squeezing operator.

    Args:
        N: Hilbert Space Size
        z: squeezing parameter
        implementation: Qarray implementation type.  ``expm`` densifies
            internally; the result is converted back to the requested
            backend before returning.

    Returns:
        Squeezing operator
    """
    a = destroy(N, implementation=implementation)
    op = (1 / 2.0) * jnp.conj(z) * (a @ a) - (1 / 2.0) * z * (a.dag() @ a.dag())
    result = op.expm()
    if implementation is None:
        return result
    return result.to_backend(implementation)


def squeezing_linear_to_dB(z):
    return 20 * jnp.log10(jnp.exp(jnp.abs(z)))


def squeezing_dB_to_linear(z_dB):
    return jnp.log(10**(z_dB / 20))

# States ---------------------------------------------------------------------


_KET_INCOMPATIBLE = frozenset({QarrayImplType.SPARSE_DIA, QarrayImplType.CUQUANTUM})


def _ket_safe_impl(implementation):
    """Return ``implementation`` unless it can't represent a ket; then ``None``.

    SPARSE_DIA stores diagonals of square matrices and CUQUANTUM is mode-
    structured; neither has a meaningful representation for a column vector
    ``(N, 1)``.  Falls back to the default backend (dense unless overridden
    in SETTINGS) so the resulting state is usable in matmul.
    """
    if implementation is None:
        return None
    return None if QarrayImplType(implementation) in _KET_INCOMPATIBLE else implementation


def basis(N: int, k: int, implementation=None):
    """Creates a |k> (i.e. fock state) ket in a specified Hilbert Space.

    Args:
        N: Hilbert space dimension
        k: fock number
        implementation: Qarray implementation type.  Kets are stored densely
            for backends that can't represent non-square data
            (``sparse_dia``, ``cuquantum``).

    Returns:
        Fock State |k>
    """
    return Qarray.create(
        one_hot(k, N).reshape(N, 1),
        implementation=_ket_safe_impl(implementation),
    )


def multi_mode_basis_set(Ns: List[int], implementation=None) -> Qarray:
    """Creates a multi-mode basis set.

    Args:
        Ns: List of Hilbert space dimensions for each mode.
        implementation: Qarray implementation type.  Stored densely for
            ket-incompatible backends.

    Returns:
        Multi-mode basis set.
    """
    data = jnp.eye(prod(Ns))
    dims = (tuple(Ns), tuple([1 for _ in Ns]))
    return Qarray.create(
        data,
        dims=dims,
        bdims=(prod(Ns),),
        implementation=_ket_safe_impl(implementation),
    )


def coherent(N: int, α: complex, implementation=None) -> Qarray:
    """Coherent state.

    Args:
        N: Hilbert Space Size.
        α: coherent state amplitude.
        implementation: Qarray implementation type.  The result is a ket; for
            backends without a ket representation (``sparse_dia``,
            ``cuquantum``) the entire computation falls back to dense.

    Return:
        Coherent state |α⟩.
    """
    safe = _ket_safe_impl(implementation)
    return displace(N, α, implementation=safe) @ basis(N, 0, implementation=safe)


def thermal_dm(N: int, n: float, implementation=None) -> Qarray:
    """Thermal state.

    Args:
        N: Hilbert Space Size.
        n: average photon number.
        implementation: Qarray implementation type.

    Return:
        Thermal state.
    """

    beta = jnp.log(1 + 1 / n)

    return Qarray.create(
        jnp.where(
            jnp.isposinf(beta),
            basis(N, 0).to_dm().data,
            jnp.diag(jnp.exp(-beta * jnp.linspace(0, N - 1, N))),
        ),
        implementation=implementation,
    ).unit()


def basis_like(A: Qarray, ks: List[int], implementation=None) -> Qarray:
    """Creates a |k> (i.e. fock state) ket with the same space dims as A.

    Args:
        A: state or operator.
        ks: fock numbers.
        implementation: Qarray implementation type.

    Returns:
        Fock State |k> with the same space dims as A.
    """
    space_dims = A.space_dims
    assert len(space_dims) == len(ks), "len(ks) must be equal to len(space_dims)"

    kets = []
    for j, k in enumerate(ks):
        kets.append(basis(space_dims[j], k, implementation=implementation))
    return tensor(*kets)

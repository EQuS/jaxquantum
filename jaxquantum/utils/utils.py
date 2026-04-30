"""
JAX Utils
"""

from numbers import Number
from typing import Dict

from jax import lax, Array, device_put, config
from jax._src.scipy.special import gammaln
import jax.numpy as jnp
import numpy as np

from typing import Literal

config.update("jax_enable_x64", True)


def device_put_params(params: Dict, non_device_params=None):
    non_device_params = [] if non_device_params is None else non_device_params
    for param in params:
        if param in non_device_params:
            continue
        if isinstance(params[param], Number) or isinstance(params[param], np.ndarray):
            params[param] = device_put(params[param])
    return params


def comb(N, k):
    """
    NCk

    #TODO: replace with jsp.special.comb once issue is closed:
    https://github.com/google/jax/issues/9709

    Args:
        N: total items
        k: # of items to choose

    Returns:
        NCk: N choose k
    """
    one = 1
    N_plus_1 = lax.add(N, one)
    k_plus_1 = lax.add(k, one)
    return lax.exp(
        lax.sub(
            gammaln(N_plus_1), lax.add(gammaln(k_plus_1), gammaln(lax.sub(N_plus_1, k)))
        )
    )


def complex_to_real_iso_matrix(A):
    return jnp.block([[jnp.real(A), -jnp.imag(A)], [jnp.imag(A), jnp.real(A)]])


def real_to_complex_iso_matrix(A):
    N = A.shape[0]
    return A[: N // 2, : N // 2] + 1j * A[N // 2 :, : N // 2]


def complex_to_real_iso_vector(v):
    return jnp.block([[jnp.real(v)], [jnp.imag(v)]])


def real_to_complex_iso_vector(v):
    N = v.shape[0]
    return v[: N // 2, :] + 1j * v[N // 2 :, :]


def imag_times_iso_vector(v):
    N = v.shape[0]
    return jnp.block([[-v[N // 2 :, :]], [v[: N // 2, :]]])


def imag_times_iso_matrix(A):
    N = A.shape[0]
    Ar = A[: N // 2, : N // 2]
    Ai = A[N // 2 :, : N // 2]
    return jnp.block([[-Ai, -Ar], [Ar, -Ai]])


def conj_transpose_iso_matrix(A):
    N = A.shape[0]
    Ar = A[: N // 2, : N // 2].T
    Ai = A[N // 2 :, : N // 2].T
    return jnp.block([[Ar, Ai], [-Ai, Ar]])


def robust_isscalar(val):
    is_scalar = isinstance(val, Number) or np.isscalar(val)
    if isinstance(val, Array):
        is_scalar = is_scalar or (len(val.shape) == 0)
    return is_scalar


# =====================================================

# Precision

def set_precision(precision: Literal["single", "double"]):
    """
    Set the precision of JAX operations.

    Args:
        precision: 'single' or 'double'

    Raises:
        ValueError: if precision is not 'single' or 'double'
    """
    if precision == "single":
        config.update("jax_enable_x64", False)
    elif precision == "double":
        config.update("jax_enable_x64", True)
    else:
        raise ValueError("precision must be 'single' or 'double'")


# =====================================================

# Sharding


def set_default_sharding(sharding):
    """Configure the global default ``Sharding`` applied to every Qarray.

    Once set, every ``DenseImpl`` and ``SparseDiaImpl`` construction routes
    its underlying ``jnp.ndarray`` through ``jax.lax.with_sharding_constraint``
    using *sharding*. ``SparseBCOOImpl`` is unsupported under sharding and
    will raise from ``Qarray.create(..., implementation=SPARSE_BCOO)``.

    Args:
        sharding: Either a ``jax.sharding.Sharding`` (typically
            ``NamedSharding(mesh, PartitionSpec(...))``) applied to every
            array regardless of rank, or a callable ``(arr) -> Sharding``
            for rank-adaptive partitioning. Pass ``None`` to disable
            (equivalent to ``clear_default_sharding()``).
    """
    from jaxquantum.core.settings import SETTINGS
    SETTINGS["default_sharding"] = sharding


def get_default_sharding():
    """Return the configured default sharding, or ``None`` if unset."""
    from jaxquantum.core.settings import SETTINGS
    return SETTINGS["default_sharding"]


def clear_default_sharding():
    """Disable default sharding (return to single-device behaviour)."""
    from jaxquantum.core.settings import SETTINGS
    SETTINGS["default_sharding"] = None


def set_device_mesh(shape, axis_names, partition_spec=None, devices=None):
    """Configure default sharding from a high-level mesh description.

    Convenience wrapper around :func:`set_default_sharding` that builds a
    ``Mesh`` and ``NamedSharding`` for you. Mirrors the pattern used in
    ``experiments/distributed/1-demo.ipynb``.

    Args:
        shape: Tuple of mesh dimensions, e.g. ``(2,)`` for a 2-device 1D mesh
            or ``(2, 4)`` for a 2x4 2D mesh.
        axis_names: Tuple of mesh axis names, same length as *shape*, e.g.
            ``('dp',)`` or ``('dp', 'mp')``.
        partition_spec: Optional ``jax.sharding.PartitionSpec``. If ``None``,
            stores a rank-adaptive callable that picks the partition for
            each array based on the *name* of each mesh axis:

            * Names starting with ``'dp'`` / ``'data'`` → data-parallel:
              prefer the leading batch axes ``[0, 1, ..., rank-3]``, then
              fall through to matrix axes ``[-2, -1]``.
            * Names starting with ``'mp'`` / ``'model'`` → model-parallel:
              prefer matrix axes ``[-2, -1]``, then fall through to leading
              batch axes.
            * Anything else behaves like ``'mp'``.

            Each mesh axis is greedy-bound to the first un-claimed array
            axis (in priority order) whose size is divisible by the mesh
            axis size. Mesh axes that find no binding are unused (the array
            replicates along them). This produces e.g.

            * ``('dp',)`` on ``(B, N, N)`` → ``P('dp', None, None)``
              (parameter sweep — each device gets a slice of B).
            * ``('mp',)`` on ``(N, N)`` → ``P('mp', None)`` (single large
              system, matrix-row sharded).
            * ``('dp', 'mp')`` on ``(B, N, N)`` → ``P('dp', 'mp', None)``
              (both modes simultaneously).
        devices: Optional explicit list of devices. Defaults to
            ``jax.devices()``.

    Raises:
        ValueError: if ``len(shape) != len(axis_names)``.
    """
    if len(shape) != len(axis_names):
        raise ValueError(
            f"shape ({shape}) and axis_names ({axis_names}) must have the "
            "same length"
        )

    import jax
    from math import prod
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    if devices is None:
        # Slice to the first prod(shape) devices so a 1D mesh works on a
        # host with extra devices (e.g. tests run with XLA_FLAGS spoofing 8
        # CPUs but the user wants a (2,) mesh).
        all_devices = jax.devices()
        needed = prod(shape)
        if len(all_devices) < needed:
            raise ValueError(
                f"set_device_mesh(shape={shape}) needs {needed} devices but "
                f"only {len(all_devices)} are available."
            )
        devices = all_devices[:needed]

    mesh_devices = mesh_utils.create_device_mesh(shape, devices=devices)
    mesh = Mesh(mesh_devices, axis_names)

    if partition_spec is None:
        # Rank-adaptive: pick partition per array based on each mesh axis's
        # name. 'dp'/'data' prefers leading batch axes; 'mp'/'model' (and
        # anything else) prefers matrix axes. Greedy first-fit binding;
        # unbound mesh axes leave the array replicated along them. See the
        # docstring for the full priority table.
        mesh_axis_priorities = [
            (name, _array_axis_priority(name)) for name in axis_names
        ]

        def _adaptive(arr):
            rank = arr.ndim
            if rank == 0:
                return NamedSharding(mesh, PartitionSpec())
            spec_parts = [None] * rank
            used_array_axes: set = set()
            for mesh_axis, priority_fn in mesh_axis_priorities:
                mesh_size = mesh.shape[mesh_axis]
                for array_axis in priority_fn(rank):
                    if array_axis in used_array_axes:
                        continue
                    if arr.shape[array_axis] % mesh_size != 0:
                        continue
                    spec_parts[array_axis] = mesh_axis
                    used_array_axes.add(array_axis)
                    break
            return NamedSharding(mesh, PartitionSpec(*spec_parts))

        set_default_sharding(_adaptive)
    else:
        set_default_sharding(NamedSharding(mesh, partition_spec))


def _array_axis_priority(mesh_axis_name: str):
    """Return a function ``rank -> list[int]`` giving the array-axis
    priority order for the given mesh-axis name.

    'dp'/'data' → batch dims first (leading axes), then matrix dims.
    'mp'/'model' (or anything else) → matrix dims first, then batch.
    """
    name = mesh_axis_name.lower()
    is_dp = name.startswith("dp") or name.startswith("data")

    def priority(rank: int) -> list:
        if rank >= 2:
            matrix = [rank - 2, rank - 1]
            batch = list(range(rank - 2))
        else:
            matrix = list(range(rank))
            batch = []
        return (batch + matrix) if is_dp else (matrix + batch)

    return priority
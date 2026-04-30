"""Core settings.

Holds global runtime configuration read by the rest of ``jaxquantum``.

How sharding works
------------------

A single global ``SETTINGS["default_sharding"]`` (either a
``jax.sharding.Sharding`` *or* a callable ``arr -> Sharding``) is consumed
by every ``QarrayImpl._make`` classmethod via :func:`_maybe_shard`. When
the entry is ``None`` (the default), every ``_make`` call is a pure
pass-through and behaviour matches the single-device codebase exactly.

Why a callable: lets the rank-adaptive default produced by
``set_device_mesh`` return a different ``NamedSharding`` per array shape
(matrices vs kets vs batched arrays vs SparseDIA ``_diags``).

Two parallelism modes are first-class, picked by mesh-axis name:

* **Data-parallel** — name a mesh axis ``'dp'`` or ``'data'``. Shards the
  leading batch dim of ``(B, *)`` arrays (parameter sweeps, vmap'd
  trajectories).
* **Model-parallel** — name a mesh axis ``'mp'`` or ``'model'`` (or
  anything else). Shards the matrix dim of ``(*, n, n)`` operators.
* **Both** — use a 2D mesh ``axis_names=('dp', 'mp')``.

Chokepoint: every impl construction goes through ``_make`` →
``_maybe_shard`` → ``jax.lax.with_sharding_constraint``. The constraint is
a no-op when the array is already correctly sharded, so it's safe to apply
on every op (matmul, kron, conversions). Pure JAX ops between calls
propagate the sharding through XLA without further help.

Not sharded: ``SparseBCOO`` (variable nnz per shard — raises in
``from_data``); ``_offsets`` on ``SparseDIA`` (static Python metadata,
not a JAX array).

User knobs (in :mod:`jaxquantum.utils.utils`, next to ``set_precision``):
``set_device_mesh``, ``set_default_sharding``, ``get_default_sharding``,
``clear_default_sharding``.
"""

from __future__ import annotations

from typing import Any, Optional


SETTINGS: dict = {
    "auto_tidyup_atol": 1e-14,
    # Optional[jax.sharding.Sharding | Callable[[Array], jax.sharding.Sharding]]
    "default_sharding": None,
}


def _maybe_shard(arr: Any) -> Any:
    """Apply ``SETTINGS['default_sharding']`` to *arr* if one is configured.

    Uses ``jax.lax.with_sharding_constraint`` (works inside and outside
    ``jit``; near-free no-op when the array is already correctly sharded).
    Short-circuits and returns *arr* unchanged when no default sharding is
    set — this is the common single-device path and must stay zero-cost.
    """
    s = SETTINGS["default_sharding"]
    if s is None:
        return arr

    # Lazy imports keep this module import-cheap.
    from jax.lax import with_sharding_constraint

    sharding = s(arr) if callable(s) else s
    _validate_shape(arr, sharding)
    return with_sharding_constraint(arr, sharding)


def _validate_shape(arr: Any, sharding: Any) -> None:
    """Raise ``ValueError`` if *arr*'s shape can't be partitioned per *sharding*.

    Walks the ``PartitionSpec`` against ``arr.shape`` and checks that each
    sharded axis is divisible by the product of the named mesh-axis sizes.
    Pure-Python on shape tuples — runs only when sharding is enabled.

    Only attempts validation for ``NamedSharding`` (the common case from
    ``set_device_mesh``). Other ``Sharding`` types are forwarded to JAX
    untouched and any mismatch surfaces from JAX itself.
    """
    try:
        from jax.sharding import NamedSharding
    except Exception:  # pragma: no cover — jax always has this
        return

    if not isinstance(sharding, NamedSharding):
        return

    spec = sharding.spec
    mesh = sharding.mesh
    shape = getattr(arr, "shape", None)
    if shape is None:
        return

    for axis_idx, axis_spec in enumerate(spec):
        if axis_spec is None or axis_idx >= len(shape):
            continue
        names = axis_spec if isinstance(axis_spec, tuple) else (axis_spec,)
        mesh_size = 1
        for name in names:
            mesh_size *= mesh.shape[name]
        dim = shape[axis_idx]
        if dim % mesh_size != 0:
            raise ValueError(
                f"Cannot apply default sharding {sharding} to array of "
                f"shape {tuple(shape)}: axis {axis_idx} (size {dim}) is not "
                f"divisible by mesh axis {names} (size {mesh_size}). "
                "Pass an explicit `sharding=` per call, or call "
                "`jqt.clear_default_sharding()` to disable."
            )

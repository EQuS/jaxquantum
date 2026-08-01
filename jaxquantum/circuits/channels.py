"""Functional direct-channel constructors and reusable kernels."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from jax import lax
import jax.numpy as jnp

from jaxquantum.circuits.gates import Gate
from jaxquantum.core.qarray import Qarray


def apply_elementwise_channel(rho, params):
    """Multiply trailing density-matrix axes by a channel factor."""
    return rho * params["_factor"][..., None, :, :]


def apply_shifted_channel(rho, params):
    """Apply output-indexed shifted Kraus branches."""
    coefficients = params["_coefficients"]
    shifts = params["_shifts"]
    indices = jnp.arange(rho.shape[-1])

    def branch(index):
        source = indices + shifts[index]
        valid = (source >= 0) & (source < rho.shape[-1])
        source = jnp.clip(source, 0, rho.shape[-1] - 1)
        shifted = rho[..., source[:, None], source[None, :]]
        coefficient = jnp.where(
            valid,
            coefficients[..., index, None, :],
            0,
        )
        return coefficient[..., :, None] * shifted * jnp.conj(coefficient[..., None, :])

    return lax.fori_loop(
        1,
        coefficients.shape[-2],
        lambda index, total: total + branch(index),
        branch(0),
    )


def apply_kraus_map(kraus, rho):
    """Apply a dense Kraus stack whose leading axis indexes branches."""
    kraus = kraus.to_dense().data if isinstance(kraus, Qarray) else kraus
    if kraus.shape[0] == 0:
        return rho

    def branch(index):
        matrix = lax.dynamic_index_in_dim(kraus, index, 0, False)
        return matrix @ rho @ jnp.swapaxes(jnp.conj(matrix), -1, -2)

    return lax.fori_loop(
        1,
        kraus.shape[0],
        lambda index, total: total + branch(index),
        branch(0),
    )


def apply_channel(channel, rho, axes=(-2, -1)):
    """Apply a channel to density-matrix axes, with Kraus fallback."""
    input_ndim = rho.ndim
    rho = jnp.moveaxis(rho, axes, (-2, -1))
    if channel._channel_apply is not None:
        result = channel._channel_apply(rho[..., None, :, :], channel.params)
        result = jnp.squeeze(result, axis=-3)
    else:
        result = apply_kraus_map(channel.KM, rho)
    extra_dims = result.ndim - input_ndim
    output_axes = tuple(axis + extra_dims if axis >= 0 else axis for axis in axes)
    return jnp.moveaxis(result, (-2, -1), output_axes)


def _kraus_generator(kraus):
    if kraus is None:
        return None
    if callable(kraus):

        def generate(params):
            result = kraus(params)
            return result if isinstance(result, Qarray) else Qarray.from_list(result)

        return generate
    kraus = kraus if isinstance(kraus, Qarray) else Qarray.from_list(kraus)
    return lambda _: kraus


def Channel(
    dims,
    apply: Callable,
    *,
    params=None,
    kraus=None,
    name="Channel",
):
    """Create a channel from a pure density-matrix kernel."""
    num_modes = 1 if isinstance(dims, int) else len(dims)
    return Gate.create(
        dims,
        name=name,
        params={} if params is None else params,
        gen_KM=_kraus_generator(kraus),
        channel_apply=apply,
        lazy_kraus=kraus is not None,
        num_modes=num_modes,
    )


def ElementwiseChannel(
    dims,
    factor,
    *,
    params=None,
    kraus=None,
    name="ElementwiseChannel",
):
    """Create ``rho[m,n] *= factor[m,n]`` channel."""
    params = dict(params or {})
    params["_factor"] = jnp.asarray(factor)
    return Channel(
        dims,
        apply_elementwise_channel,
        params=params,
        kraus=kraus,
        name=name,
    )


def ShiftedChannel(
    dimension,
    coefficients,
    shifts: Sequence[int],
    *,
    params=None,
    kraus=None,
    name="ShiftedChannel",
):
    """Create a channel from output coefficients and input-index shifts."""
    coefficients = jnp.asarray(coefficients)
    shifts = jnp.asarray(shifts)
    if coefficients.shape[-2:] != (shifts.shape[0], dimension):
        raise ValueError("coefficients must end in (num_shifts, dimension)")
    params = dict(params or {})
    params.update(
        {
            "_coefficients": coefficients,
            "_shifts": shifts,
        }
    )
    return Channel(
        dimension,
        apply_shifted_channel,
        params=params,
        kraus=kraus,
        name=name,
    )

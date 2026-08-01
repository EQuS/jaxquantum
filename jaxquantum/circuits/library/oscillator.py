"""Oscillator gates."""

from jaxquantum.core.operators import (displace, basis, destroy, create, num)
from jaxquantum.circuits.gates import Gate
from jaxquantum.circuits.channels import (
    apply_elementwise_channel,
    apply_shifted_channel,
)
from jax.scipy.special import gammaln
import jax.numpy as jnp
from jaxquantum import Qarray
from jaxquantum.utils import hermgauss
from functools import partial
import jax

def diag_expm(diag_matrix):
    """Computes expm of a diagonal matrix efficiently (O(N) instead of O(N^3))."""
    # Extract diagonal, exponentiate elements, put back on diagonal
    return jnp.diag(jnp.exp(jnp.diagonal(diag_matrix)))



def D(N, alpha, ts=None, c_ops=None):
    """Displacement gate.

    Args:
        N: Hilbert space dimension.
        alpha: Displacement amplitude.
        ts: Optional time array for hamiltonian simulation.
        c_ops: Optional collapse operators.

    Returns:
        Displacement gate.
    """
    gen_Ht = None
    if ts is not None:
        delta_t = ts[-1] - ts[0]
        amp = 1j * alpha / delta_t
        a = destroy(N)
        gen_Ht = lambda params: (lambda t: jnp.conj(amp) * a + amp * a.dag())

    return Gate.create(
        N,
        name="D",
        params={"alpha": alpha},
        gen_U=lambda params: displace(N, params["alpha"]),
        gen_Ht=gen_Ht,
        ts=ts,
        gen_c_ops=lambda params: Qarray.from_list([]) if c_ops is None else c_ops,
        num_modes=1,
    )


def _conditional_displacement(N, beta, echoed=False):
    displacement = displace(N, beta / 2).data
    inverse = jnp.swapaxes(jnp.conj(displacement), -1, -2)
    blocks = jnp.zeros(
        displacement.shape[:-2] + (2, N, 2, N),
        dtype=displacement.dtype,
    )
    if echoed:
        blocks = blocks.at[..., 1, :, 0, :].set(displacement)
        blocks = blocks.at[..., 0, :, 1, :].set(inverse)
    else:
        blocks = blocks.at[..., 0, :, 0, :].set(displacement)
        blocks = blocks.at[..., 1, :, 1, :].set(inverse)
    return Qarray.create(blocks.reshape(blocks.shape[:-4] + (2 * N, 2 * N)),
                         dims=[[2, N], [2, N]])


def CD(N, beta, ts=None):
    """Conditional displacement gate.

    Args:
        N: Hilbert space dimension.
        beta: Conditional displacement amplitude.
        ts: Optional time sequence for hamiltonian simulation.

    Returns:
        Conditional displacement gate.
    """
    gen_Ht = None
    if ts is not None:
        g = basis(2, 0)
        e = basis(2, 1)
        gg = g @ g.dag()
        ee = e @ e.dag()
        delta_t = ts[-1] - ts[0]
        amp = 1j * beta / delta_t / 2
        a = destroy(N)
        gen_Ht = lambda params: lambda t: (
            gg
            ^ (jnp.conj(amp) * a + amp * a.dag()) + ee
            ^ (jnp.conj(-amp) * a + (-amp) * a.dag())
        )

    return Gate.create(
        [2, N],
        name="CD",
        params={"beta": beta},
        gen_U=lambda params: _conditional_displacement(N, params["beta"]),
        gen_Ht=gen_Ht,
        ts=ts,
        num_modes=2,
    )


def ECD(N, beta, ts=None):
    """Echoed conditional displacement gate.

    Args:
        N: Hilbert space dimension.
        beta: Conditional displacement amplitude.
        ts: Optional time sequence for hamiltonian simulation.

    Returns:
        Echoed conditional displacement gate.
    """
    return Gate.create(
        [2, N],
        name="ECD",
        params={"beta": beta},
        gen_U=lambda params: _conditional_displacement(
            N,
            params["beta"],
            echoed=True,
        ),
        gen_Ht=None,
        ts=ts,
        num_modes=2,
    )

def CR(N, theta):
    """Conditional rotation gate.

    Args:
        N: Hilbert space dimension.
        theta: Conditional rotation angle.

    Returns:
        Conditional rotation gate.
    """
    g = basis(2, 0)
    e = basis(2, 1)

    gg = g @ g.dag()
    ee = e @ e.dag()


    return Gate.create(
        [2, N],
        name="CR",
        params={"theta": theta},
        gen_U=lambda params: (gg ^ (-1.j*theta/2*create(N)@destroy(N)).expm())
        + (ee ^ (1.j*theta/2*create(N)@destroy(N)).expm()),
        num_modes=2,
    )


# --- 2. Optimized Kernels (Using diag_expm) ---

@partial(jax.jit, static_argnames=["N", "max_l"])
def _Amp_Damp_Kraus_Map_JIT(N, err_prob, max_l):
    coefficients = _amp_damp_coefficients(
        N,
        err_prob,
        max_l,
        truncate=False,
    )
    indices = jnp.arange(N)
    sources = jnp.minimum(
        indices + jnp.arange(coefficients.shape[-2])[:, None],
        N - 1,
    )
    return _shifted_kraus_maps(coefficients, sources)


def _amp_damp_coefficients(
    dimension,
    probability,
    max_l,
    truncate=True,
):
    probability = jnp.asarray(probability)
    indices = jnp.arange(dimension)
    order = min(max_l, dimension - 1) if truncate else max_l
    losses = jnp.arange(order + 1)
    retention = jnp.sqrt(1.0 - probability)
    log_binomial = (
        gammaln(indices + losses[:, None] + 1)
        - gammaln(indices + 1)
        - gammaln(losses[:, None] + 1)
    )
    coefficients = (
        jnp.exp(0.5 * log_binomial)
        * jnp.power(probability[..., None, None], 0.5 * losses[:, None])
        * jnp.power(retention[..., None, None], indices)
    )
    return jnp.where(
        indices + losses[:, None] < dimension,
        coefficients,
        0,
    )


def _shifted_kraus_maps(coefficients, sources):
    coefficients = jnp.moveaxis(coefficients, -2, 0)
    indices = jnp.arange(coefficients.shape[-1])

    def build(coefficient, source):
        shape = coefficient.shape[:-1] + (coefficient.shape[-1],) * 2
        return jnp.zeros(shape, coefficient.dtype).at[..., indices, source].set(
            coefficient
        )

    return jax.vmap(build)(coefficients, sources)


def _amp_damp_apply(rho, params, dimension, max_l):
    coefficients = _amp_damp_coefficients(
        dimension,
        params["err_prob"],
        max_l,
    )
    return apply_shifted_channel(
        rho,
        {
            "_coefficients": coefficients,
            "_shifts": jnp.arange(coefficients.shape[-2]),
        },
    )


def Amp_Damp(N, err_prob, max_l):
    if max_l < 0:
        raise ValueError("max_l must be non-negative")

    def kmap(params):
        data = _Amp_Damp_Kraus_Map_JIT(
            params["N"],
            params["err_prob"],
            params["max_l"],
        )
        return Qarray.create(data, dims=[[N], [N]], bdims=data.shape[:-2])

    return Gate.create(
        N,
        name="Amp_Damp",
        params={"err_prob": err_prob, "max_l": max_l, "N": N},
        gen_KM=kmap,
        channel_apply=partial(
            _amp_damp_apply,
            dimension=N,
            max_l=max_l,
        ),
        lazy_kraus=True,
        num_modes=1,
    )


@partial(jax.jit, static_argnames=["N", "max_l"])
def _Amp_Gain_Kraus_Map_JIT(N, err_prob, max_l):
    coefficients = _amp_gain_coefficients(
        N,
        err_prob,
        max_l,
        truncate=False,
    )
    indices = jnp.arange(N)
    sources = jnp.maximum(
        indices - jnp.arange(coefficients.shape[-2])[:, None],
        0,
    )
    return _shifted_kraus_maps(coefficients, sources)


def _amp_gain_coefficients(
    dimension,
    probability,
    max_l,
    truncate=True,
):
    probability = jnp.asarray(probability)
    indices = jnp.arange(dimension)
    order = min(max_l, dimension - 1) if truncate else max_l
    gains = jnp.arange(order + 1)
    source = indices - gains[:, None]
    log_binomial = (
        gammaln(indices + 1)
        - gammaln(jnp.maximum(source, 0) + 1)
        - gammaln(gains[:, None] + 1)
    )
    coefficients = (
        jnp.exp(0.5 * log_binomial)
        * jnp.power(probability[..., None, None], 0.5 * gains[:, None])
        * jnp.power(
            jnp.sqrt(1.0 - probability)[..., None, None],
            jnp.maximum(source, 0),
        )
    )
    return jnp.where(source >= 0, coefficients, 0)


def _amp_gain_apply(rho, params, dimension, max_l):
    coefficients = _amp_gain_coefficients(
        dimension,
        params["err_prob"],
        max_l,
    )
    return apply_shifted_channel(
        rho,
        {
            "_coefficients": coefficients,
            "_shifts": -jnp.arange(coefficients.shape[-2]),
        },
    )


def Amp_Gain(N, err_prob, max_l):
    if max_l < 0:
        raise ValueError("max_l must be non-negative")

    def kmap(params):
        data = _Amp_Gain_Kraus_Map_JIT(
            params["N"],
            params["err_prob"],
            params["max_l"],
        )
        return Qarray.create(data, dims=[[N], [N]], bdims=data.shape[:-2])

    return Gate.create(
        N,
        name="Amp_Gain",
        params={"err_prob": err_prob, "max_l": max_l, "N": N},
        gen_KM=kmap,
        channel_apply=partial(
            _amp_gain_apply,
            dimension=N,
            max_l=max_l,
        ),
        lazy_kraus=True,
        num_modes=1,
    )


@partial(jax.jit, static_argnames=["N", "max_l"])
def _Thermal_Ch_Kraus_Map_JIT(N, err_prob, n_bar, max_l):
    coefficients, gains, losses = _thermal_coefficients(
        N,
        err_prob,
        n_bar,
        max_l,
        truncate=False,
    )
    indices = jnp.arange(N)
    sources = jnp.clip(
        indices + losses[:, None] - gains[:, None],
        0,
        N - 1,
    )
    return _shifted_kraus_maps(coefficients, sources)


def _thermal_coefficients(
    dimension,
    probability,
    n_bar,
    max_l,
    truncate=True,
):
    probability = jnp.asarray(probability)
    n_bar = jnp.asarray(n_bar)
    order = (
        min(max_l, dimension - 1) + 1
        if truncate
        else max_l + 1
    )
    pair_indices = jnp.arange(order**2)
    gains = pair_indices // order
    losses = pair_indices % order
    output = jnp.arange(dimension)
    source = output + losses[:, None] - gains[:, None]
    valid = (source >= 0) & (source < dimension) & (
        output + losses[:, None] < dimension
    )
    source_safe = jnp.clip(source, 0, dimension - 1)
    prefactor = jnp.sqrt(
        jnp.power(
            probability[..., None, None] * (1.0 + n_bar[..., None, None]),
            losses[:, None],
        )
        * jnp.power(
            probability[..., None, None] * n_bar[..., None, None],
            gains[:, None],
        )
        / (
            jnp.exp(gammaln(losses[:, None] + 1))
            * jnp.exp(gammaln(gains[:, None] + 1))
        )
    )
    ratio = jnp.exp(
        gammaln(output + losses[:, None] + 1)
        - 0.5
        * (
            gammaln(source_safe + 1)
            + gammaln(output + 1)
        )
    )
    coefficients = (
        prefactor
        * ratio
        * jnp.power(
            jnp.sqrt(1.0 - probability)[..., None, None],
            output,
        )
    )
    return jnp.where(valid, coefficients, 0), gains, losses


def _thermal_apply(rho, params, dimension, max_l):
    coefficients, gains, losses = _thermal_coefficients(
        dimension,
        params["err_prob"],
        params["n_bar"],
        max_l,
    )
    return apply_shifted_channel(
        rho,
        {
            "_coefficients": coefficients,
            "_shifts": losses - gains,
        },
    )


def Thermal_Ch(N, err_prob, n_bar, max_l):
    if max_l < 0:
        raise ValueError("max_l must be non-negative")

    def kmap(params):
        data = _Thermal_Ch_Kraus_Map_JIT(
            params["N"],
            params["err_prob"],
            params["n_bar"],
            params["max_l"],
        )
        return Qarray.create(data, dims=[[N], [N]], bdims=data.shape[:-2])

    return Gate.create(
        N,
        name="Thermal_Ch",
        params={
            "err_prob": err_prob,
            "n_bar": n_bar,
            "max_l": max_l,
            "N": N,
        },
        gen_KM=kmap,
        channel_apply=partial(
            _thermal_apply,
            dimension=N,
            max_l=max_l,
        ),
        lazy_kraus=True,
        num_modes=1,
    )


@partial(jax.jit, static_argnames=["N", "max_l"])
def _Dephasing_Ch_Kraus_Map_JIT(N, ws, phis, max_l):
    def compute_op(w, phi):
        diagonal = jnp.exp(1.0j * phi[..., None] * jnp.arange(N))
        return jnp.sqrt(w) * diagonal[..., None, :] * jnp.eye(N)
    return jax.vmap(compute_op)(ws, phis)


def _dephasing_factor(dimension, probability, nodes, weights):
    indices = jnp.arange(dimension)
    delta = indices[:, None] - indices[None, :]
    phases = jnp.sqrt(2 * jnp.asarray(probability)[..., None]) * nodes
    return jnp.sum(
        weights[:, None, None]
        * jnp.exp(1.0j * phases[..., :, None, None] * delta),
        axis=-3,
    )


def _dephasing_apply(rho, params, dimension):
    return apply_elementwise_channel(
        rho,
        {
            "_factor": _dephasing_factor(
                dimension,
                params["err_prob"],
                params["_nodes"],
                params["_weights"],
            )
        },
    )


def Dephasing_Ch(N, err_prob, max_l):
    if max_l < 1:
        raise ValueError("max_l must be positive")
    xs, ws_raw = hermgauss(max_l)
    ws = 1/jnp.sqrt(jnp.pi)*ws_raw

    def kmap(params):
        phases = (
            jnp.sqrt(2 * params["err_prob"])[..., None]
            * params["_nodes"]
        )
        data = _Dephasing_Ch_Kraus_Map_JIT(
            params["N"],
            params["_weights"],
            jnp.moveaxis(phases, -1, 0),
            params["max_l"],
        )
        return Qarray.create(data, dims=[[N], [N]], bdims=data.shape[:-2])

    return Gate.create(
        N,
        name="Dephasing_Ch",
        params={
            "err_prob": err_prob,
            "max_l": max_l,
            "N": N,
            "_nodes": xs,
            "_weights": ws,
        },
        gen_KM=kmap,
        channel_apply=partial(_dephasing_apply, dimension=N),
        lazy_kraus=True,
        num_modes=1,
    )


def selfKerr(N, K):
    a = destroy(N)
    return Gate.create(
        N,
        name="selfKerr",
        params={"Kerr": K},
        gen_U=lambda params: (-1.0j * K / 2 * (a.dag() @ a.dag() @ a @ a)).expm(),
        num_modes=1,
    )


@partial(jax.jit, static_argnames=["N", "max_l"])
def _Dephasing_Reset_Kraus_Map_JIT(N, p, t_rst, chi, max_l):
    p = jnp.asarray(p)
    g = basis(2, 0).data
    e = basis(2, 1).data
    gg = jnp.outer(g, jnp.conj(g))
    ee = jnp.outer(e, jnp.conj(e))
    ge = jnp.outer(g, jnp.conj(e))

    n_op = num(N).data
    I_N = jnp.eye(N)

    exponents = jnp.arange(max_l - 1) / (max_l - 1)
    raw_weights = jnp.power(p[..., None], exponents)
    weights = (
        (1 - p[..., None])
        * raw_weights
        / jnp.sum(raw_weights, axis=-1, keepdims=True)
    )

    def compute_op(l):
        def branch_0(_):
            matrix = jnp.kron(gg, I_N)
            return jnp.broadcast_to(matrix, p.shape + matrix.shape)

        def branch_1(_):
            op_osc = diag_expm(-1.0j * chi * t_rst * n_op)
            return jnp.sqrt(p)[..., None, None] * jnp.kron(ee, op_osc)

        def branch_rest(_):
            exponent = -1.0j * chi * t_rst * (l - 2) / (max_l - 1)
            op_osc = diag_expm(exponent * n_op)
            return (
                jnp.sqrt(weights[..., l - 2])[..., None, None]
                * jnp.kron(ge, op_osc)
            )

        return jax.lax.cond(
            l == 0,
            branch_0,
            lambda _: jax.lax.cond(l == 1, branch_1, branch_rest, operand=None),
            operand=None
        )

    ls = jnp.arange(max_l+1)
    return jax.vmap(compute_op)(ls)


def _dephasing_reset_factors(N, p, t_rst, chi, max_l):
    indices = jnp.arange(N)
    delta = indices[:, None] - indices[None, :]
    transfer_indices = jnp.arange(2, max_l + 1)
    p = jnp.asarray(p)
    raw_weights = jnp.power(
        p[..., None],
        (transfer_indices - 2) / (max_l - 1),
    )
    weights = (
        (1 - p[..., None])
        * raw_weights
        / jnp.sum(raw_weights, axis=-1, keepdims=True)
    )
    angles = chi * t_rst * (transfer_indices - 2) / (max_l - 1)
    transfer = jnp.sum(
        weights[..., :, None, None]
        * jnp.exp(-1.0j * angles[:, None, None] * delta),
        axis=-3,
    )
    excited = p[..., None, None] * jnp.exp(-1.0j * chi * t_rst * delta)
    return transfer, excited


def _dephasing_reset_apply(rho, params, dimension, max_l):
    """Apply structured ancilla reset directly to density-matrix blocks."""
    transfer_factor, excited_factor = _dephasing_reset_factors(
        dimension,
        params["p"],
        params["t_rst"],
        params["chi"],
        max_l,
    )
    blocks = rho.reshape(rho.shape[:-2] + (2, dimension, 2, dimension))
    ground = blocks[..., 0, :, 0, :]
    excited = blocks[..., 1, :, 1, :]
    transfer = transfer_factor[..., None, :, :]
    excited_factor = excited_factor[..., None, :, :]
    ground = ground + transfer * excited
    excited = excited_factor * excited
    zero = jnp.zeros_like(ground)
    output = jnp.stack(
        (
            jnp.stack((ground, zero), axis=-2),
            jnp.stack((zero, excited), axis=-2),
        ),
        axis=-4,
    )
    return output.reshape(output.shape[:-4] + (2 * dimension,) * 2)


def Dephasing_Reset(N, p, t_rst, chi, max_l):
    if max_l < 2:
        raise ValueError("max_l must be at least two")

    def kmap(params):
        data = _Dephasing_Reset_Kraus_Map_JIT(
            params["N"],
            params["p"],
            params["t_rst"],
            params["chi"],
            params["max_l"],
        )
        return Qarray.create(
            data,
            dims=[[2, N], [2, N]],
            bdims=data.shape[:-2],
        )

    return Gate.create(
        [2, N],
        name="Dephasing_Reset",
        params={
            "p": p,
            "t_rst": t_rst,
            "chi": chi,
            "max_l": max_l,
            "N": N,
        },
        gen_KM=kmap,
        channel_apply=partial(
            _dephasing_reset_apply,
            dimension=N,
            max_l=max_l,
        ),
        lazy_kraus=True,
        num_modes=2,
    )

"""qubit gates."""

from jaxquantum.core.operators import (
    identity,
    sigmax,
    sigmay,
    sigmaz,
    basis,
    hadamard,
    qubit_rotation,
)
from jaxquantum.circuits.gates import Gate
from jaxquantum.circuits.channels import apply_elementwise_channel
from jaxquantum.core.qarray import Qarray
import jax.numpy as jnp


def _reset_apply(rho, params):
    output = jnp.zeros_like(rho)
    return output.at[..., 0, 0].set(jnp.trace(rho, axis1=-2, axis2=-1))


def _imperfect_reset_apply(rho, params):
    p_eg = jnp.asarray(params["p_eg"])[..., None]
    p_ee = jnp.asarray(params["p_ge"])[..., None]
    ground = (1 - p_eg) * rho[..., 0, 0] + (1 - p_ee) * rho[..., 1, 1]
    excited = p_eg * rho[..., 0, 0] + p_ee * rho[..., 1, 1]
    zero = jnp.zeros_like(ground)
    return jnp.stack(
        (jnp.stack((ground, zero), -1), jnp.stack((zero, excited), -1)),
        -2,
    )


def _thermal_qubit_apply(rho, params):
    probability = jnp.asarray(params["err_prob"])[..., None]
    n_bar = jnp.asarray(params["n_bar"])[..., None]
    p0 = (n_bar + 1) / (2 * n_bar + 1)
    p1 = n_bar / (2 * n_bar + 1)
    coherence = jnp.sqrt(1 - probability)
    ground = (1 - p1 * probability) * rho[..., 0, 0] + p0 * probability * rho[..., 1, 1]
    excited = (
        p1 * probability * rho[..., 0, 0] + (1 - p0 * probability) * rho[..., 1, 1]
    )
    return jnp.stack(
        (
            jnp.stack((ground, coherence * rho[..., 0, 1]), -1),
            jnp.stack((coherence * rho[..., 1, 0], excited), -1),
        ),
        -2,
    )


def _measure_x_apply(rho, params):
    sign = params["_sign"]
    weight = 0.25 * (
        rho[..., 0, 0] + rho[..., 1, 1] + sign * (rho[..., 0, 1] + rho[..., 1, 0])
    )
    return jnp.stack(
        (
            jnp.stack((weight, sign * weight), -1),
            jnp.stack((sign * weight, weight), -1),
        ),
        -2,
    )


def _dephase_x_apply(rho, params):
    return 0.5 * (rho + rho[..., ::-1, ::-1])


def _dephase_z_apply(rho, params):
    factor = (1 - 2 * jnp.asarray(params["err_prob"]))[..., None]
    one = jnp.ones_like(factor)
    return jnp.stack(
        (
            jnp.stack((one * rho[..., 0, 0], factor * rho[..., 0, 1]), -1),
            jnp.stack((factor * rho[..., 1, 0], one * rho[..., 1, 1]), -1),
        ),
        -2,
    )


def X():
    return Gate.create(2, name="X", gen_U=lambda params: sigmax(), num_modes=1)


def Y():
    return Gate.create(2, name="Y", gen_U=lambda params: sigmay(), num_modes=1)


def Z():
    return Gate.create(2, name="Z", gen_U=lambda params: sigmaz(), num_modes=1)


def H():
    return Gate.create(2, name="H", gen_U=lambda params: hadamard(), num_modes=1)


def Rx(theta, ts=None):
    gen_Ht = None
    if ts is not None:
        delta_t = ts[-1] - ts[0]
        amp = theta / delta_t
        gen_Ht = lambda params: (lambda t: amp / 2 * sigmax())

    return Gate.create(
        2,
        name="Rx",
        params={"theta": theta},
        gen_U=lambda params: qubit_rotation(params["theta"], 1, 0, 0),
        gen_Ht=gen_Ht,
        ts=ts,
        num_modes=1,
    )


def Ry(theta, ts=None):
    gen_Ht = None
    if ts is not None:
        delta_t = ts[-1] - ts[0]
        amp = theta / delta_t
        gen_Ht = lambda params: (lambda t: amp / 2 * sigmay())
    return Gate.create(
        2,
        name="Ry",
        params={"theta": theta},
        gen_U=lambda params: qubit_rotation(params["theta"], 0, 1, 0),
        gen_Ht=gen_Ht,
        ts=ts,
        num_modes=1,
    )


def Rz(theta, ts=None):
    gen_Ht = None
    if ts is not None:
        delta_t = ts[-1] - ts[0]
        amp = theta / delta_t
        gen_Ht = lambda params: (lambda t: amp / 2 * sigmaz())
    return Gate.create(
        2,
        name="Rz",
        params={"theta": theta},
        gen_U=lambda params: qubit_rotation(params["theta"], 0, 0, 1),
        gen_Ht=gen_Ht,
        ts=ts,
        num_modes=1,
    )


def MZ(measure=None):
    if measure is None:
        gate_name = "MZ"
        factor = jnp.eye(2)
    elif measure == +1:
        gate_name = "MZ_plus"
        factor = jnp.array([[1, 0], [0, 0]])
    elif measure == -1:
        gate_name = "MZ_minus"
        factor = jnp.array([[0, 0], [0, 1]])
    else:
        raise ValueError("measure should be None, +1 or -1")

    def kmap(params):
        g, e = basis(2, 0), basis(2, 1)
        if measure is None:
            return Qarray.from_list([g @ g.dag(), e @ e.dag()])
        state = g if measure == 1 else e
        return Qarray.from_list([state @ state.dag()])

    return Gate.create(
        2,
        name=gate_name,
        params={"_factor": factor},
        gen_KM=kmap,
        channel_apply=apply_elementwise_channel,
        lazy_kraus=True,
        num_modes=1,
    )


def MX(measure=None):
    if measure is None:
        gate_name = "MX"
        channel_apply = _dephase_x_apply
        sign = 0
    elif measure == +1:
        gate_name = "MX_plus"
        channel_apply = _measure_x_apply
        sign = 1
    elif measure == -1:
        gate_name = "MX_minus"
        channel_apply = _measure_x_apply
        sign = -1
    else:
        raise ValueError("measure should be None, +1 or -1")

    def kmap(params):
        g, e = basis(2, 0), basis(2, 1)
        plus, minus = (g + e).unit(), (g - e).unit()
        if measure is None:
            return Qarray.from_list([plus @ plus.dag(), minus @ minus.dag()])
        state = plus if measure == 1 else minus
        return Qarray.from_list([state @ state.dag()])

    return Gate.create(
        2,
        name=gate_name,
        params={"_sign": sign},
        gen_KM=kmap,
        channel_apply=channel_apply,
        lazy_kraus=True,
        num_modes=1,
    )


def Reset():
    def kmap(params):
        g, e = basis(2, 0), basis(2, 1)
        return Qarray.from_list([g @ g.dag(), g @ e.dag()])

    return Gate.create(
        2,
        name="Reset",
        gen_KM=kmap,
        channel_apply=_reset_apply,
        lazy_kraus=True,
        num_modes=1,
    )


def IP_Reset(p_eg, p_ee):
    def kmap(params):
        g, e = basis(2, 0), basis(2, 1)
        gg, ge = g @ g.dag(), g @ e.dag()
        eg, ee = e @ g.dag(), e @ e.dag()
        return Qarray.from_list(
            [
                jnp.sqrt(1 - p_eg) * gg,
                jnp.sqrt(p_ee) * ee,
                jnp.sqrt(p_eg) * eg,
                jnp.sqrt(1 - p_ee) * ge,
            ]
        )

    return Gate.create(
        2,
        name="IP_Reset",
        params={"p_eg": p_eg, "p_ge": p_ee},
        gen_KM=kmap,
        channel_apply=_imperfect_reset_apply,
        lazy_kraus=True,
        num_modes=1,
    )


def CX():
    g = basis(2, 0)
    e = basis(2, 1)

    gg = g @ g.dag()
    ee = e @ e.dag()

    op = (gg ^ identity(2)) + (ee ^ sigmax())

    return Gate.create([2, 2], name="CX", gen_U=lambda params: op, num_modes=2)


def _Thermal_Kraus_Ops_Qb(err_prob, n_bar):
    """ " Returns the Kraus Operators for a thermal channel with probability
    err_prob and average photon number n_bar in a Hilbert Space of size 2"""
    p0 = (n_bar + 1) / (2 * n_bar + 1)
    p1 = n_bar / (2 * n_bar + 1)
    return [
        Qarray.create(jnp.sqrt(p0) * jnp.array([[1, 0], [0, jnp.sqrt(1 - err_prob)]])),
        Qarray.create(jnp.sqrt(p0) * jnp.array([[0, jnp.sqrt(err_prob)], [0, 0]])),
        Qarray.create(jnp.sqrt(p1) * jnp.array([[0, 0], [jnp.sqrt(err_prob), 0]])),
        Qarray.create(jnp.sqrt(p1) * jnp.array([[jnp.sqrt(1 - err_prob), 0], [0, 1]])),
    ]


def Thermal_Ch_Qb(err_prob, n_bar):
    kmap = lambda params: Qarray.from_list(_Thermal_Kraus_Ops_Qb(err_prob, n_bar))
    return Gate.create(
        2,
        name="Thermal_Ch_Qb",
        params={"err_prob": err_prob, "n_bar": n_bar},
        gen_KM=kmap,
        channel_apply=_thermal_qubit_apply,
        lazy_kraus=True,
        num_modes=1,
    )


def _Pure_Dephasing_Ops_Qb(err_prob):
    """ " Returns the Kraus Operators for a thermal channel with probability
    err_prob and average photon number n_bar in a Hilbert Space of size 2"""
    return [jnp.sqrt(1 - err_prob) * identity(2), jnp.sqrt(err_prob) * sigmaz()]


def Dephasing_Ch_Qb(err_prob):
    kmap = lambda params: Qarray.from_list(_Pure_Dephasing_Ops_Qb(err_prob))
    return Gate.create(
        2,
        name="Dephasing_Ch_Qb",
        params={"err_prob": err_prob},
        gen_KM=kmap,
        channel_apply=_dephase_z_apply,
        lazy_kraus=True,
        num_modes=1,
    )

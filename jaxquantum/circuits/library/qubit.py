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
from jaxquantum.core.qarray import Qarray
import jax.numpy as jnp


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
    g = basis(2, 0)
    e = basis(2, 1)

    gg = g @ g.dag()
    ee = e @ e.dag()

    if measure is None:
        kmap = Qarray.from_list([gg, ee])
        gate_name = "MZ"
    elif measure == +1:
        kmap = Qarray.from_list([gg])
        gate_name = "MZ_plus"
    elif measure == -1:
        kmap = Qarray.from_list([ee])
        gate_name = "MZ_minus"
    else:
        raise ValueError("measure should be None, +1 or -1")

    return Gate.create(2, name=gate_name, gen_KM=lambda params: kmap, num_modes=1)


def MX(measure=None):
    g = basis(2, 0)
    e = basis(2, 1)

    plus = (g + e).unit()
    minus = (g - e).unit()

    pp = plus @ plus.dag()
    mm = minus @ minus.dag()

    if measure is None:
        kmap = Qarray.from_list([pp, mm])
        gate_name = "MX"
    elif measure == +1:
        kmap = Qarray.from_list([pp])
        gate_name = "MX_plus"
    elif measure == -1:
        kmap = Qarray.from_list([mm])
        gate_name = "MX_minus"
    else:
        raise ValueError("measure should be None, +1 or -1")

    return Gate.create(2, name=gate_name, gen_KM=lambda params: kmap, num_modes=1)


def Reset():
    g = basis(2, 0)
    e = basis(2, 1)

    gg = g @ g.dag()
    ge = g @ e.dag()

    kmap = Qarray.from_list([gg, ge])
    return Gate.create(2, name="Reset", gen_KM=lambda params: kmap, num_modes=1)


def IP_Reset(p_eg, p_ee):
    g = basis(2, 0)
    e = basis(2, 1)

    gg = g @ g.dag()
    ge = g @ e.dag()
    eg = e @ g.dag()
    ee = e @ e.dag()

    k_0 = jnp.sqrt(1 - p_eg) * gg + jnp.sqrt(p_eg) * eg
    k_1 = jnp.sqrt(p_ee) * ee + jnp.sqrt(1 - p_ee) * ge

    kmap = Qarray.from_list([k_0, k_1])

    return Gate.create(
        2,
        name="IP_Reset",
        params={"p_eg": p_eg, "p_ge": p_ee},
        gen_KM=lambda params: kmap,
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
    p = n_bar / (n_bar + 1)
    return [
        Qarray.create(
            jnp.sqrt(1 - p) * jnp.array([[1, 0], [0, jnp.sqrt(1 - err_prob)]])
        ),
        Qarray.create(jnp.sqrt(1 - p) * jnp.array([[0, jnp.sqrt(err_prob)], [0, 0]])),
        Qarray.create(jnp.sqrt(p) * jnp.array([[0, 0], [jnp.sqrt(err_prob), 0]])),
        Qarray.create(jnp.sqrt(p) * jnp.array([[jnp.sqrt(1 - err_prob), 0], [0, 1]])),
    ]


def Thermal_Ch_Qb(err_prob, n_bar):
    kmap = lambda params: Qarray.from_list(_Thermal_Kraus_Ops_Qb(err_prob,
                                                               n_bar))
    return Gate.create(
        2,
        name="Thermal_Ch_Qb",
        params={"err_prob": err_prob, "n_bar": n_bar},
        gen_KM=kmap,
        num_modes=1,
    )


def _Pure_Dephasing_Ops_Qb(err_prob):
    """ " Returns the Kraus Operators for a thermal channel with probability
    err_prob and average photon number n_bar in a Hilbert Space of size 2"""
    return [
        jnp.sqrt(1-err_prob)*identity(2),
        jnp.sqrt(err_prob)*sigmaz()
    ]


def Dephasing_Ch_Qb(err_prob):
    kmap = lambda params: Qarray.from_list(_Pure_Dephasing_Ops_Qb(err_prob))
    return Gate.create(
        2,
        name="Dephasing_Ch_Qb",
        params={"err_prob": err_prob},
        gen_KM=kmap,
        num_modes=1,
    )

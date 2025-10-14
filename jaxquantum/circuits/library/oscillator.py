"""Oscillator gates."""

from jaxquantum.core.operators import displace, basis, destroy, create, num
from jaxquantum.circuits.gates import Gate
from jax.scipy.special import factorial
import jax.numpy as jnp
from jaxquantum import Qarray
from jaxquantum.utils import hermgauss


def D(N, alpha, ts=None, c_ops=None):
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


def CD(N, beta, ts=None):
    g = basis(2, 0)
    e = basis(2, 1)

    gg = g @ g.dag()
    ee = e @ e.dag()

    gen_Ht = None
    if ts is not None:
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
        gen_U=lambda params: (gg ^ displace(N, params["beta"] / 2))
        + (ee ^ displace(N, -params["beta"] / 2)),
        gen_Ht=gen_Ht,
        ts=ts,
        num_modes=2,
    )

def CR(N, theta):
    g = basis(2, 0)
    e = basis(2, 1)

    gg = g @ g.dag()
    ee = e @ e.dag()


    return Gate.create(
        [2, N],
        name="CR",
        params={"theta": theta},
        gen_U=lambda params: (gg ^ (-1.j*theta/2*destroy(N)@create(N)).expm())
        + (ee ^ (1.j*theta/2*destroy(N)@create(N)).expm()),
        num_modes=2,
    )


def _Ph_Loss_Kraus_Op(N, err_prob, l):
    """ " Returns the Kraus Operators for l-photon loss with probability
    err_prob in a Hilbert Space of size N"""
    return (
        jnp.sqrt(jnp.power(err_prob, l) / factorial(l))
        * (num(N) * jnp.log(jnp.sqrt(1 - err_prob))).expm()
        * destroy(N).powm(l)
    )


def Amp_Damp(N, err_prob, max_l):
    kmap = lambda params: Qarray.from_list(
        [_Ph_Loss_Kraus_Op(N, err_prob, l) for l in range(max_l + 1)]
    )
    return Gate.create(
        N,
        name="Amp_Damp",
        params={"err_prob": err_prob, "max_l": max_l},
        gen_KM=kmap,
        num_modes=1,
    )


def _Ph_Gain_Kraus_Op(N, err_prob, l):
    """ " Returns the Kraus Operators for l-photon gain with probability
    err_prob in a Hilbert Space of size N"""
    return (
        jnp.sqrt(jnp.power(err_prob, l) / factorial(l))
        * create(N).powm(l)
        * (num(N) * jnp.log(jnp.sqrt(1 - err_prob))).expm()
    )


def Amp_Gain(N, err_prob, max_l):
    kmap = lambda params: Qarray.from_list(
        [_Ph_Gain_Kraus_Op(N, err_prob, l) for l in range(max_l + 1)]
    )
    return Gate.create(
        N,
        name="Amp_Gain",
        params={"err_prob": err_prob, "max_l": max_l},
        gen_KM=kmap,
        num_modes=1,
    )


def _Thermal_Kraus_Op(N, err_prob, n_bar, l, k):
    """ " Returns the Kraus Operators for a thermal channel with probability
    err_prob and average photon number n_bar in a Hilbert Space of size N"""
    return (
        jnp.sqrt(
            jnp.power(err_prob * (1 + n_bar), k)
            * jnp.power(err_prob * n_bar, l)
            / factorial(l)
            / factorial(k)
        )
        * (num(N) * jnp.log(jnp.sqrt(1 - err_prob))).expm()
        * destroy(N).powm(k)
        * create(N).powm(l)
    )


def Thermal_Ch(N, err_prob, n_bar, max_l):
    kmap = lambda params: Qarray.from_list(
        [
            _Thermal_Kraus_Op(N, err_prob, n_bar, l, k)
            for l in range(max_l + 1)
            for k in range(max_l + 1)
        ]
    )
    return Gate.create(
        N,
        name="Thermal_Ch",
        params={"err_prob": err_prob, "n_bar": n_bar, "max_l": max_l},
        gen_KM=kmap,
        num_modes=1,
    )


def _Dephasing_Kraus_Op(N, w, phi):
    """ " Returns the Kraus Operators for dephasing with weight w and phase phi
     in a Hilbert Space of size N"""
    return (
        jnp.sqrt(w)*(1.j*phi*num(N)).expm()
    )


def Dephasing_Ch(N, err_prob, max_l):

    xs, ws = hermgauss(max_l)
    phis = jnp.sqrt(2*err_prob)*xs
    ws = 1/jnp.sqrt(jnp.pi)*ws

    kmap = lambda params: Qarray.from_list(
        [_Dephasing_Kraus_Op(N, w, phi) for (w, phi) in zip(ws, phis)]
    )
    return Gate.create(
        N,
        name="Amp_Gain",
        params={"err_prob": err_prob, "max_l": max_l},
        gen_KM=kmap,
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

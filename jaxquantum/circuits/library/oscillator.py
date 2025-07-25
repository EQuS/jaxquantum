"""Oscillator gates."""

from jaxquantum.core.operators import displace, basis, destroy, create, num
from jaxquantum.circuits.gates import Gate
from jax.scipy.special import factorial
import jax.numpy as jnp
from jaxquantum import Qarray


def D(N, alpha, ts=None, c_ops=None):

    gen_Ht = None
    if ts is not None:
        delta_t = ts[-1] - ts[0]
        amp = 1j * alpha / delta_t
        a = destroy(N)
        gen_Ht = lambda params: (lambda t: -1.0j * (amp * a.dag() - jnp.conj(amp) * a))

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
        amp = 1j * beta / delta_t
        a = destroy(N)
        gen_Ht = lambda params: lambda t: (-1.0j *
            (gg ^ (amp / 2 * a.dag() - jnp.conj(amp / 2) * a)) +
            (ee ^ (-amp / 2 * a.dag() - jnp.conj(-amp / 2) * a)))

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


def _Kraus_Op(N, err_prob, l):
    """ " Returns the Kraus Operators for l-photon loss with probability
    err_prob in a Hilbert Space of size N"""
    return (
        jnp.sqrt(jnp.power(err_prob, l) / factorial(l))
        * (num(N) * jnp.log(jnp.sqrt(1 - err_prob))).expm()
        * destroy(N).powm(l)
    )


def Amp_Damp(N, err_prob, max_l):
    kmap = lambda params: Qarray.from_list(
        [_Kraus_Op(N, err_prob, l) for l in range(max_l + 1)]
    )
    return Gate.create(
        N,
        name="Amp_Damp",
        params={"err_prob": err_prob, "max_l": max_l},
        gen_KM=kmap,
        num_modes=1,
    )

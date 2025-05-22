""" Oscillator gates. """

from jaxquantum.core.operators import displace, basis, destroy, num
from jaxquantum.circuits.gates import Gate
from jax.scipy.special import factorial
import jax.numpy as jnp
from jaxquantum import Qarray

def D(N, alpha):
    return Gate.create(
        N, 
        name="D",
        params={"alpha": alpha},
        gen_U = lambda params: displace(N,params["alpha"]),
        num_modes=1
    )

def CD(N, beta):
    g = basis(2,0)
    e = basis(2,1)
    
    gg = g @ g.dag()
    ee = e @ e.dag()

    return Gate.create(
        [2,N], 
        name="CD",
        params={"beta": beta},
        gen_U = lambda params: (gg ^ displace(N, params["beta"]/2)) + (ee ^ displace(N, -params["beta"]/2)),
        num_modes=2
    )

def _K(N, err_prob, l):
    """" Returns the Kraus Operators for l-photon loss with probability
    err_prob in a Hilbert Space of size N """
    return jnp.sqrt(jnp.power(err_prob, l)/factorial(l)) * (num(N) * jnp.log(jnp.sqrt(1-err_prob))).expm() * destroy(N).powm(l)


def Amp_Damp(N, err_prob, max_l):
    kmap = lambda params: Qarray.from_list([_K(N, err_prob, l) for l in
                                            range(max_l+1)])
    return Gate.create(
        N,
        name="Amp_Damp",
        params={"err_prob": err_prob, "max_l": max_l},
        gen_KM = kmap,
        num_modes=1
    )
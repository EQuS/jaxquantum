""" Generic gates. """


from jaxquantum.core.operators import identity
from jaxquantum.circuits.gates import Gate
import jax.numpy as jnp
from jaxquantum import Qarray, tensor


def Id(Ns, ts=None, c_ops=None):

    Is = tensor(*[identity(N) for N in Ns])

    return Gate.create(
        Ns,
        name="Id",
        params={},
        gen_U=lambda params: Is,
        gen_Ht=lambda params: (lambda t: 0*Is),
        ts=ts,
        gen_c_ops=lambda params: Qarray.from_list([]) if c_ops is None else c_ops,
        num_modes=len(Ns),
    )
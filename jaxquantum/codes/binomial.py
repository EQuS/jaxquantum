"""
Cat Code Qubit
"""

from typing import Tuple


from jaxquantum.utils.utils import comb
from bosonic.codes import BosonicQubit
import jaxquantum as jqt

from jax import vmap
from jax import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)


class BinomialQubit(BosonicQubit):
    """
    Cat Qubit Class.
    """
    name = "binomial"

    def _params_validation(self):
        super()._params_validation()

        # notation https://arxiv.org/pdf/2010.08699.pdf
        if "L" not in self.params:
            self.params["L"] = 1
        if "G" not in self.params:
            self.params["G"] = 0
        if "D" not in self.params:
            self.params["D"] = 0

    def _get_basis_z(self) -> Tuple[jqt.Qarray, jqt.Qarray]:
        """
        Construct basis states |+-x>, |+-y>, |+-z>
        """
        N = self.params["N"]

        L = self.params["L"]
        G = self.params["G"]
        D = self.params["D"]

        S = L + G

        M = jnp.max(jnp.array([L, G, 2 * D]))
        
        def plus_z_gen(p):
            C = comb(M + 1, p)
            return jnp.sqrt(C) * jqt.basis(N, p * (S + 1)).data

        plus_z = jnp.sum(vmap(plus_z_gen)(jnp.arange(0, M + 2, 2)), axis=0)
        plus_z = jqt.unit(jqt.Qarray.create(plus_z))

        def minus_z_gen(p):
            C = comb(M + 1, p)
            return jnp.sqrt(C) * jqt.basis(N, p * (S + 1)).data

        minus_z = jnp.sum(vmap(minus_z_gen)(jnp.arange(1, M + 2, 2)), axis=0)
        minus_z = jqt.unit(jqt.Qarray.create(minus_z))

        return plus_z, minus_z

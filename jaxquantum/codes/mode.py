"""
Generic Bosonic Mode Class
"""

from typing import Tuple

from jaxquantum.codes.base import BosonicQubit
import jaxquantum as jqt

from jax import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)


class BosonicMode(BosonicQubit):
    """
    FockQubit
    """

    def _params_validation(self):
        super()._params_validation()

    def _get_basis_z(self) -> Tuple[jqt.Qarray, jqt.Qarray]:
        """
        Construct basis states |+-x>, |+-y>, |+-z>
        """
        N = int(self.params["N"])
        plus_z = jqt.basis(N, 0)
        minus_z = jqt.basis(N, 1)
        return plus_z, minus_z

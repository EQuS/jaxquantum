"""
Cat Code Qubit
"""

from typing import Tuple

from bosonic.codes import BosonicQubit
import jaxquantum as jqt

from jax import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)


class CatQubit(BosonicQubit):
    """
    Cat Qubit Class.
    """
    name = "cat"

    @property
    def _non_device_params(self):
        param_list = super()._non_device_params
        param_list.append("alpha")
        return param_list

    def _params_validation(self):
        super()._params_validation()
        if "alpha" not in self.params:
            self.params["alpha"] = 2

    def _get_basis_z(self) -> Tuple[jqt.Qarray, jqt.Qarray]:
        """
        Construct basis states |+-x>, |+-y>, |+-z>
        """
        N = self.params["N"]
        a = self.params["alpha"]
        plus_z = jqt.unit(jqt.coherent(N, a) + jqt.coherent(N, -1.0 * a))
        minus_z = jqt.unit(jqt.coherent(N, 1.0j * a) + jqt.coherent(N, -1.0j * a))
        return plus_z, minus_z

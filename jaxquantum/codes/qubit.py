"""
Qubit
"""

from typing import Tuple
import warnings

from bosonic.codes.base import BosonicQubit
import jaxquantum as jqt

from jax import config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import qutip as qt

config.update("jax_enable_x64", True)


class Qubit(BosonicQubit):
    """
    FockQubit
    """

    def _params_validation(self):
        super()._params_validation()
        self.params["N"] = 2

    def _get_basis_z(self) -> Tuple[jqt.Qarray, jqt.Qarray]:
        """
        Construct basis states |+-x>, |+-y>, |+-z>
        """
        N = int(self.params["N"])
        plus_z = jqt.basis(N, 0)
        minus_z = jqt.basis(N, 1)
        return plus_z, minus_z

    @property
    def x_U(self) -> jqt.Qarray:
        return jqt.sigmax()

    @property
    def y_U(self) -> jqt.Qarray:
        return jqt.sigmay()

    @property
    def z_U(self) -> jqt.Qarray:
        return jqt.sigmaz()

    def plot(self, state, ax=None, qp_type="", **kwargs) -> None:
        state = self.jqt2qt(state)
        with warnings.catch_warnings():
            # TODO: suppressing deprecation warnings, deal with this
            warnings.simplefilter("ignore")
            b = qt.Bloch()
            b.add_states(state)
            b.render()
            b.show()
            plt.tight_layout()
            plt.show()

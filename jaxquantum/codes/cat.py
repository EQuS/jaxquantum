"""
Cat Code Qubit
"""

from typing import Tuple

from jaxquantum.codes.base import BosonicQubit
import jaxquantum as jqt
import jax.numpy as jnp

from jax import config

config.update("jax_enable_x64", True)


class CatQubit(BosonicQubit):
    """
    Cat Qubit Class.
    """

    PARAMETERS = ["alpha", "delta"]

    name = "cat"

    @property
    def _non_device_params(self):
        param_list = super()._non_device_params
        param_list.extend(["alpha", "delta"])
        return param_list

    def _params_validation(self):
        super()._params_validation()
        if "alpha" not in self.params:
            self.params["alpha"] = 2
        if "delta" not in self.params:
            self.params["delta"] = 1.0
        if not 0 < self.params["delta"] <= 1:
            raise ValueError("delta must satisfy 0 < delta <= 1")

    @classmethod
    def displaced_squeezed_state(cls, N, alpha, delta):
        """Return D(alpha) S(-log(delta))|0>."""
        return (
            jqt.displace(N, alpha) @ jqt.squeeze(N, -jnp.log(delta)) @ jqt.basis(N, 0)
        )

    @classmethod
    def cat_state(cls, N, alpha, delta, parity="even"):
        """Return the even or odd squeezed cat from the paper."""
        if parity not in ("even", "odd"):
            raise ValueError("parity must be 'even' or 'odd'")
        plus = cls.displaced_squeezed_state(N, alpha, delta)
        minus = cls.displaced_squeezed_state(N, -alpha, delta)
        return jqt.unit(plus + (1 if parity == "even" else -1) * minus)

    def _get_basis_z(self) -> Tuple[jqt.Qarray, jqt.Qarray]:
        """Return the displaced squeezed states at +/- alpha."""
        N = self.params["N"]
        alpha = self.params["alpha"]
        delta = self.params["delta"]
        return (
            self.displaced_squeezed_state(N, alpha, delta),
            self.displaced_squeezed_state(N, -alpha, delta),
        )

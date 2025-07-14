"""Tunable Transmon."""

from flax import struct
from jax import config
import jax.numpy as jnp

from jaxquantum.devices.superconducting.transmon import Transmon

config.update("jax_enable_x64", True)


@struct.dataclass
class TunableTransmon(Transmon):
    """
    Tunable Transmon Device.
    """

    @property
    def Ej(self):
        Ejsum = self.params["Ej1"] + self.params["Ej2"]
        phi_ext = 2 * jnp.pi * self.params["phi_ext"]
        gamma = self.params["Ej2"] / self.params["Ej1"]
        d = (gamma - 1) / (gamma + 1)
        external_flux_factor = jnp.abs(
            jnp.sqrt(jnp.cos(phi_ext / 2) ** 2 + d**2 * jnp.sin(phi_ext / 2) ** 2)
        )
        return Ejsum * external_flux_factor

"""Transmon."""

from flax import struct
from jax import config

import jax.numpy as jnp
import jax.scipy as jsp

from jaxquantum.devices.superconducting.flux_base import FluxDevice
from jaxquantum.core.operators import identity, destroy, create
from jaxquantum.core import cosm, powm

config.update("jax_enable_x64", True)


@struct.dataclass
class TruncatedTransmon(FluxDevice):
    """
    Transmon Device.
    """

    def common_ops(self):
        """Written in the linear basis."""

        ops = {}

        N = self.N_pre_diag
        ops["id"] = identity(N)
        ops["a"] = destroy(N)
        ops["a_dag"] = create(N)
        ops["phi"] = self.phi_zpf() * (ops["a"] + ops["a_dag"])
        ops["n"] = 1j * self.n_zpf() * (ops["a_dag"] - ops["a"])
        return ops

    def phi_zpf(self):
        """Return Phase ZPF."""
        return (2 * self.params["Ec"] / self.params["Ej"]) ** (0.25)

    def n_zpf(self):
        """Return Charge ZPF."""
        return (self.params["Ej"] / (32 * self.params["Ec"])) ** (0.25)

    def get_linear_ω(self):
        """Get frequency of linear terms."""
        return jnp.sqrt(8 * self.params["Ec"] * self.params["Ej"])

    def get_H_linear(self):
        """Return linear terms in H."""
        w = self.get_linear_ω()
        return w * self.linear_ops["a_dag"] @ self.linear_ops["a"]

    def get_H_full(self):
        """Return full H in linear basis."""
        cos_phi_op = cosm(self.linear_ops["phi"])

        H_nl = -self.params["Ej"] * cos_phi_op - self.params[
            "Ej"
        ] / 2 * powm(self.linear_ops["phi"], 2)
        return self.get_H_linear() + H_nl

    def potential(self, phi):
        """Return potential energy for a given phi."""
        return -self.params["Ej"] * jnp.cos(2 * jnp.pi * phi)

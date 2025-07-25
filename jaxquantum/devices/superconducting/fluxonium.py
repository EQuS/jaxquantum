"""Fluxonium."""

from flax import struct
from jax import config
import jax.numpy as jnp

from jaxquantum.devices.superconducting.flux_base import FluxDevice
from jaxquantum.devices.base.base import HamiltonianTypes
from jaxquantum.core.operators import identity, destroy, create
from jaxquantum.core import cosm, sinm

config.update("jax_enable_x64", True)


@struct.dataclass
class Fluxonium(FluxDevice):
    """
    Fluxonium Device.
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

        ops["cos(φ/2)"] = cosm(ops["phi"] / 2)
        ops["sin(φ/2)"] = sinm(ops["phi"] / 2)

        return ops

    def n_zpf(self):
        n_zpf = (self.params["El"] / (32.0 * self.params["Ec"])) ** (0.25)
        return n_zpf

    def phi_zpf(self):
        """Return Phase ZPF."""
        return (2 * self.params["Ec"] / self.params["El"]) ** (0.25)

    def get_linear_ω(self):
        """Get frequency of linear terms."""
        return jnp.sqrt(8 * self.params["Ec"] * self.params["El"])

    def get_H_linear(self):
        """Return linear terms in H."""
        w = self.get_linear_ω()
        return w * (
            self.linear_ops["a_dag"] @ self.linear_ops["a"]
            + 0.5 * self.linear_ops["id"]
        )

    def get_H_full(self):
        """Return full H in linear basis."""

        phi_op = self.linear_ops["phi"]
        return self.get_H_linear() + self.get_H_nonlinear(phi_op)

    def get_H_nonlinear(self, phi_op):
        op_cos_phi = cosm(phi_op)
        op_sin_phi = sinm(phi_op)

        phi_ext = self.params["phi_ext"]
        Hcos = op_cos_phi * jnp.cos(2.0 * jnp.pi * phi_ext) + op_sin_phi * jnp.sin(
            2.0 * jnp.pi * phi_ext
        )
        H_nl = -self.params["Ej"] * Hcos
        return H_nl

    def potential(self, phi):
        """Return potential energy for a given phi."""
        phi_ext = self.params["phi_ext"]
        V_linear = 0.5 * self.params["El"] * (2 * jnp.pi * phi) ** 2

        if self.hamiltonian == HamiltonianTypes.linear:
            return V_linear

        V_nonlinear = -self.params["Ej"] * jnp.cos(2.0 * jnp.pi * (phi - phi_ext))
        if self.hamiltonian == HamiltonianTypes.full:
            return V_linear + V_nonlinear

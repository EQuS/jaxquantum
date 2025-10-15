"""ATS."""

from flax import struct
from jax import config

import jax.numpy as jnp

from jaxquantum.devices.superconducting.flux_base import FluxDevice
from jaxquantum.core.operators import identity, destroy, create
from jaxquantum.core.qarray import cosm, sinm




@struct.dataclass
class ATS(FluxDevice):
    """
    ATS Device.
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
        return (2 * self.params["Ec"] / self.params["El"]) ** (0.25)

    def n_zpf(self):
        """Return Charge ZPF."""
        return (self.params["El"] / (32 * self.params["Ec"])) ** (0.25)

    def get_linear_ω(self):
        """Get frequency of linear terms."""
        return jnp.sqrt(8 * self.params["El"] * self.params["Ec"])

    def get_H_linear(self):
        """Return linear terms in H."""
        w = self.get_linear_ω()
        return w * (
            self.linear_ops["a_dag"] @ self.linear_ops["a"]
            + 0.5 * self.linear_ops["id"]
        )

    @staticmethod
    def get_H_nonlinear_static(phi_op, Ej, dEj, Ej2, phi_sum, phi_delta):
        cos_phi_op = cosm(phi_op)
        sin_phi_op = sinm(phi_op)

        cos_2phi_op = cos_phi_op @ cos_phi_op - sin_phi_op @ sin_phi_op
        sin_2phi_op = 2 * cos_phi_op @ sin_phi_op

        H_nl_Ej = (
            -2
            * Ej
            * (
                cos_phi_op * jnp.cos(2 * jnp.pi * phi_delta)
                - sin_phi_op * jnp.sin(2 * jnp.pi * phi_delta)
            )
            * jnp.cos(2 * jnp.pi * phi_sum)
        )
        H_nl_dEj = (
            2
            * dEj
            * (
                sin_phi_op * jnp.cos(2 * jnp.pi * phi_delta)
                + cos_phi_op * jnp.sin(2 * jnp.pi * phi_delta)
            )
            * jnp.sin(2 * jnp.pi * phi_sum)
        )
        H_nl_Ej2 = (
            2
            * Ej2
            * (
                cos_2phi_op * jnp.cos(2 * 2 * jnp.pi * phi_delta)
                - sin_2phi_op * jnp.sin(2 * 2 * jnp.pi * phi_delta)
            )
            * jnp.cos(2 * 2 * jnp.pi * phi_sum)
        )

        H_nl = H_nl_Ej + H_nl_dEj + H_nl_Ej2

        # id_op = jqt.identity_like(phi_op)
        # phi_delta_ext_op = self.params["phi_delta_ext"] * id_op
        # H_nl_old = - 2 * Ej * jqt.cosm(phi_op + 2 * jnp.pi * phi_delta_ext_op) * jnp.cos(2 * jnp.pi * self.params["phi_sum_ext"])
        # H_nl_old += 2 * dEj * jqt.sinm(phi_op + 2 * jnp.pi * phi_delta_ext_op) * jnp.sin(2 * jnp.pi * self.params["phi_sum_ext"])
        # H_nl_old += 2 * Ej2 * jqt.cosm(2*phi_op + 2 * 2 * jnp.pi * phi_delta_ext_op) * jnp.cos(2 * 2 * jnp.pi * self.params["phi_sum_ext"])

        return H_nl

    def get_H_nonlinear(self, phi_op):
        """Return nonlinear terms in H."""

        Ej = self.params["Ej"]
        dEj = self.params["dEj"]
        Ej2 = self.params["Ej2"]

        phi_sum = self.params["phi_sum_ext"]
        phi_delta = self.params["phi_delta_ext"]

        return ATS.get_H_nonlinear_static(phi_op, Ej, dEj, Ej2, phi_sum, phi_delta)

    def get_H_full(self):
        """Return full H in linear basis."""
        phi_b = self.linear_ops["phi"]
        H_nl = self.get_H_nonlinear(phi_b)
        H = self.get_H_linear() + H_nl
        return H

    def potential(self, phi):
        """Return potential energy for a given phi."""

        phi_delta_ext = self.params["phi_delta_ext"]
        phi_sum_ext = self.params["phi_sum_ext"]

        V = 0.5 * self.params["El"] * (2 * jnp.pi * phi) ** 2
        V += (
            -2
            * self.params["Ej"]
            * jnp.cos(2 * jnp.pi * (phi + phi_delta_ext))
            * jnp.cos(2 * jnp.pi * phi_sum_ext)
        )
        V += (
            2
            * self.params["dEj"]
            * jnp.sin(2 * jnp.pi * (phi + phi_delta_ext))
            * jnp.sin(2 * jnp.pi * phi_sum_ext)
        )
        V += (
            2
            * self.params["Ej2"]
            * jnp.cos(2 * 2 * jnp.pi * (phi + phi_delta_ext))
            * jnp.cos(2 * 2 * jnp.pi * phi_sum_ext)
        )

        return V

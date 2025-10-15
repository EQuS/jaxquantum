"""Kerr Nonlinear Oscillator"""

from flax import struct
from jax import config

import jax.numpy as jnp

from jaxquantum.devices.base.base import Device, BasisTypes, HamiltonianTypes
from jaxquantum.core.operators import identity, destroy, create




@struct.dataclass
class KNO(Device):
    """
    Kerr Nonlinear Oscillator Device.
    """

    @classmethod
    def param_validation(cls, N, N_pre_diag, params, hamiltonian, basis):
        """This can be overridden by subclasses."""
        assert basis == BasisTypes.fock, (
            "Kerr Nonlinear Oscillator must be defined in the Fock basis."
        )
        assert hamiltonian == HamiltonianTypes.full, (
            "Kerr Nonlinear Oscillator uses a full Hamiltonian."
        )
        assert "ω" in params and "α" in params, (
            "Kerr Nonlinear Oscillator requires frequency 'ω' and anharmonicity 'α' as parameters."
        )

    def common_ops(self):
        ops = {}

        N = self.N
        ops["id"] = identity(N)
        ops["a"] = destroy(N)
        ops["a_dag"] = create(N)
        ops["phi"] = (ops["a"] + ops["a_dag"]) / jnp.sqrt(2)
        ops["n"] = 1j * (ops["a_dag"] - ops["a"]) / jnp.sqrt(2)
        return ops

    def get_linear_ω(self):
        """Get frequency of linear terms."""
        return self.params["ω"]

    def get_anharm(self):
        """Get anharmonicity."""
        return self.params["α"]

    def get_H_linear(self):
        """Return linear terms in H."""
        w = self.get_linear_ω()
        return w * self.linear_ops["a_dag"] @ self.linear_ops["a"]

    def get_H_full(self):
        """Return full H in linear basis."""
        α = self.get_anharm()

        return self.get_H_linear() + (α / 2) * (
            self.linear_ops["a_dag"]
            @ self.linear_ops["a_dag"]
            @ self.linear_ops["a"]
            @ self.linear_ops["a"]
        )

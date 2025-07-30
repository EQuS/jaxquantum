"""Transmon."""

from flax import struct
from jax import config

import jax.numpy as jnp

from jaxquantum.devices.base.base import BasisTypes, HamiltonianTypes
from jaxquantum.devices.superconducting.flux_base import FluxDevice
from jaxquantum.core.operators import identity, destroy, create
from jaxquantum.core.conversions import jnp2jqt

config.update("jax_enable_x64", True)


@struct.dataclass
class SNAIL(FluxDevice):
    """
    SNAIL Device.
    """

    DEFAULT_BASIS = BasisTypes.charge
    DEFAULT_HAMILTONIAN = HamiltonianTypes.full

    @classmethod
    def param_validation(cls, N, N_pre_diag, params, hamiltonian, basis):
        """This can be overridden by subclasses."""

        assert params["m"] % 1 == 0, "m must be an integer."
        assert params["m"] >= 2, "m must be greater than or equal to 2."

        if hamiltonian == HamiltonianTypes.linear:
            assert basis == BasisTypes.fock, "Linear Hamiltonian only works with Fock basis."
        elif hamiltonian == HamiltonianTypes.truncated:
            assert basis == BasisTypes.fock, "Truncated Hamiltonian only works with Fock basis."
        elif hamiltonian == HamiltonianTypes.full:
            charge_basis_types = [
                BasisTypes.charge
            ]
            assert basis in charge_basis_types, "Full Hamiltonian only works with Cooper pair charge or single-electron charge bases."

            assert (N_pre_diag - 1) % 2 * (params["m"]) == 0, "(N_pre_diag - 1)/2 must be divisible by m."

        # Set the gate offset charge to zero if not provided
        if "ng" not in params:
            params["ng"] = 0.0

    def common_ops(self):
        """ Written in the specified basis. """
        
        ops = {}

        N = self.N_pre_diag

        if self.basis == BasisTypes.fock:
            ops["id"] = identity(N)
            ops["a"] = destroy(N)
            ops["a_dag"] = create(N)
            ops["phi"] = self.phi_zpf() * (ops["a"] + ops["a_dag"])
            ops["n"] = 1j * self.n_zpf() * (ops["a_dag"] - ops["a"])

        elif self.basis == BasisTypes.charge:
            """
            Here H = 4 * Ec (n - ng)² - Ej cos(φ) in the Cooper pair charge basis. 
            """
            m = self.params["m"]
            ops["id"] = identity(N)
            ops["cos(φ/m)"] = 0.5 * (jnp2jqt(jnp.eye(N, k=1) + jnp.eye(N, k=-1)))
            ops["sin(φ/m)"] = 0.5j * (jnp2jqt(jnp.eye(N, k=1) - jnp.eye(N, k=-1)))
            ops["cos(φ)"] = 0.5 * (jnp2jqt(jnp.eye(N, k=m) + jnp.eye(N, k=-m)))
            ops["sin(φ)"] = 0.5j * (jnp2jqt(jnp.eye(N, k=m) - jnp.eye(N, k=-m)))

            n_max = (N - 1) // 2
            n_array = jnp.arange(-n_max, n_max + 1) / self.params["m"]
            ops["n"] = jnp2jqt(jnp.diag(n_array))
            
            n_minus_ng_array = n_array - self.params["ng"] * jnp.ones(N)
            ops["H_charge"] = jnp2jqt(jnp.diag(4 * self.params["Ec"] * n_minus_ng_array**2))

        return ops

    @property
    def Ej(self):
        return self.params["Ej"]

    def phi_zpf(self):
        """Return Phase ZPF."""
        return (2 * self.params["Ec"] / self.Ej) ** (0.25)

    def n_zpf(self):
        """Return Charge ZPF."""
        return (self.Ej / (32 * self.params["Ec"])) ** (0.25)

    def get_linear_ω(self):
        """Get frequency of linear terms."""
        return jnp.sqrt(8 * self.params["Ec"] * self.Ej)

    def get_H_linear(self):
        """Return linear terms in H."""
        w = self.get_linear_ω()
        return w * self.original_ops["a_dag"] @ self.original_ops["a"]

    def get_H_full(self):
        """Return full H in specified basis."""

        α = self.params["alpha"]
        m = self.params["m"]
        phi_ext = self.params["phi_ext"]
        Ej = self.Ej

        H_charge = self.original_ops["H_charge"]
        H_inductive = - α * Ej * self.original_ops["cos(φ)"] - m * Ej * (
            jnp.cos(2 * jnp.pi * phi_ext/m) * self.original_ops["cos(φ/m)"] + jnp.sin(2 * jnp.pi * phi_ext/m) * self.original_ops["sin(φ/m)"]
        )
        return H_charge + H_inductive
    
    def get_H_truncated(self):
        """Return truncated H in specified basis."""
        raise NotImplementedError("Truncated Hamiltonian not implemented for SNAIL.")
        # phi_op = self.original_ops["phi"]  
        # fourth_order_term =  -(1 / 24) * self.Ej * phi_op @ phi_op @ phi_op @ phi_op 
        # sixth_order_term = (1 / 720) * self.Ej * phi_op @ phi_op @ phi_op @ phi_op @ phi_op @ phi_op
        # return self.get_H_linear() + fourth_order_term + sixth_order_term
    
    def _get_H_in_original_basis(self):
        """ This returns the Hamiltonian in the original specified basis. This can be overridden by subclasses."""

        if self.hamiltonian == HamiltonianTypes.linear:
            return self.get_H_linear()
        elif self.hamiltonian == HamiltonianTypes.full:
            return self.get_H_full()
        elif self.hamiltonian == HamiltonianTypes.truncated:
            return self.get_H_truncated()

    def potential(self, phi):
        """Return potential energy for a given phi."""
        if self.hamiltonian == HamiltonianTypes.linear:
            return 0.5 * self.Ej * (2 * jnp.pi * phi) ** 2
        elif self.hamiltonian == HamiltonianTypes.full:

            α = self.params["alpha"]
            m = self.params["m"]
            phi_ext = self.params["phi_ext"]

            return - α * self.Ej * jnp.cos(2 * jnp.pi * phi) - (
                m * self.Ej * jnp.cos(2 * jnp.pi * (phi_ext - phi) / m)
            )

        elif self.hamiltonian == HamiltonianTypes.truncated:
            raise NotImplementedError("Truncated potential not implemented for SNAIL.")
            # phi_scaled = 2 * jnp.pi * phi
            # second_order = 0.5 * self.Ej * phi_scaled ** 2
            # fourth_order =  -(1 / 24) * self.Ej * phi_scaled ** 4
            # sixth_order = (1 / 720) * self.Ej * phi_scaled ** 6
            # return second_order + fourth_order + sixth_order
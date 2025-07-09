""" Transmon."""

from flax import struct
from jax import config

import jax.numpy as jnp
import jax.scipy as jsp

from jaxquantum.devices.base.base import BasisTypes, HamiltonianTypes
from jaxquantum.devices.superconducting.flux_base import FluxDevice
from jaxquantum.core.operators import identity, destroy, create
from jaxquantum.core.conversions import jnp2jqt

config.update("jax_enable_x64", True)


@struct.dataclass
class Transmon(FluxDevice):
    """
    Transmon Device.
    """
    DEFAULT_BASIS = BasisTypes.charge
    DEFAULT_HAMILTONIAN = HamiltonianTypes.full

    @classmethod
    def param_validation(cls, N, N_pre_diag, params, hamiltonian, basis):
        """ This can be overridden by subclasses."""
        if hamiltonian == HamiltonianTypes.linear:
            assert basis == BasisTypes.fock, "Linear Hamiltonian only works with Fock basis."
        elif hamiltonian == HamiltonianTypes.truncated:
            assert basis == BasisTypes.fock, "Truncated Hamiltonian only works with Fock basis."
        elif hamiltonian == HamiltonianTypes.full:
            assert basis in [BasisTypes.charge, BasisTypes.single_charge], "Full Hamiltonian only works with Cooper pair charge or single-electron charge bases."
        
        # Set the gate offset charge to zero if not provided
        if "ng" not in params:
            params["ng"] = 0.0
        
        assert (N_pre_diag - 1) % 2 == 0, "N_pre_diag must be odd."

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
            ops["id"] = identity(N)
            ops["cos(φ)"] = 0.5*(jnp2jqt(jnp.eye(N,k=1) + jnp.eye(N,k=-1)))
            ops["sin(φ)"] = 0.5j*(jnp2jqt(jnp.eye(N,k=1) - jnp.eye(N,k=-1)))
            n_max = (N - 1) // 2
            ops["n"] = jnp2jqt(jnp.diag(jnp.arange(-n_max, n_max + 1)))

            n_minus_ng_array = jnp.arange(-n_max, n_max + 1) - self.params["ng"] * jnp.ones(N)
            ops["H_charge"] = jnp2jqt(jnp.diag(4 * self.params["Ec"] * n_minus_ng_array**2))

        elif self.basis == BasisTypes.single_charge:
            """
            Here H = Ec (n - 2ng)² - Ej cos(φ) in the single-electron charge basis. Using Eq. (5.36) of Kyle Serniak's
            thesis, we have H = Ec ∑ₙ(n - 2*ng) |n⟩⟨n| - Ej/2 * ∑ₙ|n⟩⟨n+2| + h.c where n counts the number of electrons, 
            not Cooper pairs. Note, we use 2ng instead of ng to match the gate offset charge convention of the transmon 
            (as done in Kyle's thesis).
            """
            n_max = (N - 1) // 2

            ops["id"] = identity(N)
            ops["cos(φ)"] = 0.5*(jnp2jqt(jnp.eye(N,k=2) + jnp.eye(N,k=-2)))
            ops["sin(φ)"] = 0.5j*(jnp2jqt(jnp.eye(N,k=2) - jnp.eye(N,k=-2)))
            ops["cos(φ/2)"] = 0.5*(jnp2jqt(jnp.eye(N,k=1) + jnp.eye(N,k=-1)))
            ops["sin(φ/2)"] = 0.5j*(jnp2jqt(jnp.eye(N,k=1) - jnp.eye(N,k=-1)))
            ops["n"] = jnp2jqt(jnp.diag(jnp.arange(-n_max, n_max + 1)))

            n_minus_ng_array = jnp.arange(-n_max, n_max + 1) - 2 * self.params["ng"] * jnp.ones(N)
            ops["H_charge"] = jnp2jqt(jnp.diag(self.params["Ec"] * n_minus_ng_array**2))

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
        return self.original_ops["H_charge"] - self.Ej * self.original_ops["cos(φ)"]
    
    def get_H_truncated(self):
        """Return truncated H in specified basis."""
        phi_op = self.original_ops["phi"]  
        fourth_order_term =  - (1/24) * self.Ej * phi_op @ phi_op @ phi_op @ phi_op 
        sixth_order_term = (1/720) * self.Ej * phi_op @ phi_op @ phi_op @ phi_op @ phi_op @ phi_op
        return self.get_H_linear() + fourth_order_term + sixth_order_term
    
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
            return - self.Ej * jnp.cos(2 * jnp.pi * phi)
        elif self.hamiltonian == HamiltonianTypes.truncated:
            phi_scaled = 2 * jnp.pi * phi
            second_order = 0.5 * self.Ej * phi_scaled ** 2
            fourth_order =  - (1/24) * self.Ej * phi_scaled ** 4
            sixth_order = (1/720) * self.Ej * phi_scaled ** 6
            return second_order + fourth_order + sixth_order

    def calculate_wavefunctions(self, phi_vals):
        """Calculate wavefunctions at phi_exts."""

        if self.basis == BasisTypes.fock:
            return super().calculate_wavefunctions(phi_vals)
        elif self.basis == BasisTypes.single_charge:
            raise NotImplementedError("Wavefunctions for single charge basis not yet implemented.")
        elif self.basis == BasisTypes.charge:
            phi_vals = jnp.array(phi_vals)

            n_labels = jnp.diag(self.original_ops["n"].data)

            wavefunctions = []
            for nj in range(self.N_pre_diag):
                wavefunction = []
                for phi in phi_vals:
                    wavefunction.append(
                        (1j ** nj / jnp.sqrt(2*jnp.pi)) * jnp.sum(
                            self.eig_systems["vecs"][:,nj] * jnp.exp(1j * phi * n_labels)
                        )
                    )
                wavefunctions.append(jnp.array(wavefunction))
            return jnp.array(wavefunctions)
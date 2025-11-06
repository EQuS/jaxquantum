"""IdealQubit."""

from flax import struct
from jax import config


from jaxquantum.devices.base.base import Device, BasisTypes, HamiltonianTypes
from jaxquantum.core.operators import identity, sigmaz, sigmax, sigmay, sigmam, sigmap

config.update("jax_enable_x64", True)


@struct.dataclass
class IdealQubit(Device):
    """
    Ideal qubit Device.
    """

    @classmethod
    def param_validation(cls, N, N_pre_diag, params, hamiltonian, basis):
        """This can be overridden by subclasses."""
        assert basis == BasisTypes.fock, (
            "IdealQubit is a two-level system defined in the Fock basis."
        )
        assert hamiltonian == HamiltonianTypes.full, (
            "IdealQubit requires a full Hamiltonian."
        )
        assert N == N_pre_diag == 2, "IdealQubit is a two-level system."
        assert "ω" in params, "IdealQubit requires a frequency parameter 'ω'."

        params["Δ"] = params.get("Δ", 0.0)
        
    def common_ops(self):
        """Written in the linear basis."""
        ops = {}

        assert self.N_pre_diag == 2
        assert self.N == 2

        N = self.N_pre_diag
        ops["id"] = identity(N)
        ops["sigmaz"] = sigmaz()
        ops["sigmax"] = sigmax()
        ops["sigmay"] = sigmay()
        ops["sigmam"] = sigmam()
        ops["sigmap"] = sigmap()

        return ops

    def get_linear_ω(self):
        """Get frequency of linear terms."""
        return self.params["ω"]

    def get_H_linear(self):
        """Return linear terms in H."""
        w = self.get_linear_ω()
        return (w / 2) * self.linear_ops["sigmaz"]

    def get_H_full(self):
        """Return full H in linear basis."""
         
        return self.get_H_linear() + self.params["Δ"] / 2 * self.linear_ops["sigmax"] 

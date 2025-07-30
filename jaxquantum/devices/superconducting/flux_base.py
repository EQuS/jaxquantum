"""Flux base device."""

from abc import abstractmethod

from flax import struct
from jax import config
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxquantum.devices.common.utils import harm_osc_wavefunction
from jaxquantum.devices.base.base import Device, BasisTypes

config.update("jax_enable_x64", True)


@struct.dataclass
class FluxDevice(Device):
    @abstractmethod
    def phi_zpf(self):
        """Return Phase ZPF."""

    def _calculate_wavefunctions_fock(self, phi_vals):
        """Calculate wavefunctions at phi_exts."""
        phi_osc = self.phi_zpf() * jnp.sqrt(2)  # length of oscillator
        phi_vals = jnp.array(phi_vals)

        # calculate basis functions
        basis_functions = []
        for n in range(self.N_pre_diag):
            basis_functions.append(
                harm_osc_wavefunction(n, phi_vals, jnp.real(phi_osc))
            )
        basis_functions = jnp.array(basis_functions)

        # transform to better diagonal basis
        basis_functions_in_H_eigenbasis = self.get_vec_data_in_H_eigenbasis(
            basis_functions
        )

        # the below is equivalent to evecs_in_H_eigenbasis @ basis_functions_in_H_eigenbasis
        # since evecs in H_eigenbasis is diagonal, i.e. the identity matrix
        wavefunctions = basis_functions_in_H_eigenbasis
        return wavefunctions

    def _calculate_wavefunctions_charge(self, phi_vals):
        phi_vals = jnp.array(phi_vals)

        # calculate basis functions
        basis_functions = []
        n_labels = jnp.diag(self.original_ops["n"].data)
        for n in n_labels:
            basis_functions.append(
                1 / (jnp.sqrt(2 * jnp.pi)) * jnp.exp(1j * n * (2 * jnp.pi * -1 * phi_vals)) # Added a -1 to work with the SNAIL
            )
        basis_functions = jnp.array(basis_functions)

        # transform to better diagonal basis
        basis_functions_in_H_eigenbasis = self.get_vec_data_in_H_eigenbasis(
            basis_functions
        )

        # the below is equivalent to evecs_in_H_eigenbasis @ basis_functions_in_H_eigenbasis
        # since evecs in H_eigenbasis is diagonal, i.e. the identity matrix
        phase_correction_factors = (1j ** (jnp.arange(0, self.N_pre_diag))).reshape(
            self.N_pre_diag, 1
        )  # TODO: review why these are needed...
        wavefunctions = basis_functions_in_H_eigenbasis * phase_correction_factors
        return wavefunctions

    @abstractmethod
    def potential(self, phi):
        """Return potential energy as a function of phi."""

    def plot_wavefunctions(self, phi_vals, max_n=None, which=None, ax=None, mode="abs", ylim=None):
        if self.basis == BasisTypes.fock:
            _calculate_wavefunctions = self._calculate_wavefunctions_fock
        elif self.basis == BasisTypes.charge:
            _calculate_wavefunctions = self._calculate_wavefunctions_charge
        else:
            raise NotImplementedError(
                f"The {self.basis} is not yet supported for plotting wavefunctions."
            )

        """Plot wavefunctions at phi_exts."""
        wavefunctions = _calculate_wavefunctions(phi_vals)
        energy_levels = self.eig_systems["vals"][: self.N]

        potential = self.potential(phi_vals)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5), dpi=1000)
        else:
            fig = ax.get_figure()

        min_val = None
        max_val = None

        assert max_n is None or which is None, "Can't specify both max_n and which"

        max_n = self.N if max_n is None else max_n
        levels = range(max_n) if which is None else which

        for n in levels:
            if mode == "abs":
                wf_vals = jnp.abs(wavefunctions[n, :]) ** 2
            elif mode == "real":
                wf_vals = wavefunctions[n, :].real
            elif mode == "imag":
                wf_vals = wavefunctions[n, :].imag

            wf_vals += energy_levels[n]
            curr_min_val = min(wf_vals)
            curr_max_val = max(wf_vals)

            if min_val is None or curr_min_val < min_val:
                min_val = curr_min_val

            if max_val is None or curr_max_val > max_val:
                max_val = curr_max_val

            ax.plot(
                phi_vals, wf_vals, label=f"$|${n}$\\rangle$", linestyle="-", linewidth=1
            )
            ax.fill_between(phi_vals, energy_levels[n], wf_vals, alpha=0.5)

        ax.plot(
            phi_vals,
            potential,
            label="potential",
            color="black",
            linestyle="-",
            linewidth=1,
        )

        ylim = ylim if ylim is not None else [min_val - 1, max_val + 1]
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$\Phi/\Phi_0$")
        ax.set_ylabel(r"Energy [GHz]")

        if mode == "abs":
            title_str = r"$|\psi_n(\Phi)|^2$"
        elif mode == "real":
            title_str = r"Re($\psi_n(\Phi)$)"
        elif mode == "imag":
            title_str = r"Im($\psi_n(\Phi)$)"

        ax.set_title(f"{title_str}")

        plt.legend(fontsize=6)
        fig.tight_layout()

        return ax

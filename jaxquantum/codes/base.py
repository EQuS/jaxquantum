"""
Base Bosonic Qubit Class
"""

from typing import Dict, Optional, Tuple
from abc import abstractmethod, ABCMeta

from jaxquantum.utils.utils import device_put_params
import jaxquantum as jqt

from jax import config
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)


class BosonicQubit(metaclass=ABCMeta):
    """
    Base class for Bosonic Qubits.
    """

    name = "bqubit"

    @property
    def _non_device_params(self):
        """
        Can be overriden in child classes.
        """
        return ["N"]

    def __init__(self, params: Optional[Dict[str, float]] = None, name: str = None):
        if name is not None:
            self.name = name

        self.params = params if params else {}
        self._params_validation()

        self.params = device_put_params(self.params, self._non_device_params)

        self.common_gates: Dict[str, jqt.Qarray] = {}
        self._gen_common_gates()

        self.wigner_pts = jnp.linspace(-4.5, 4.5, 61)

        self.basis = self._get_basis_states()

        for basis_state in ["+x", "-x", "+y", "-y", "+z", "-z"]:
            assert basis_state in self.basis, (
                f"Please set the {basis_state} basis state."
            )

    def _params_validation(self):
        """
        Override this method to add additional validation to params.

        E.g.
        if "N" not in self.params:
            self.params["N"] = 50
        """
        if "N" not in self.params:
            self.params["N"] = 50

    def _gen_common_gates(self):
        """
        Override this method to add additional common gates.

        E.g.
        if "N" not in self.params:
            self.params["N"] = 50
        """
        N = self.params["N"]
        self.common_gates["a_dag"] = jqt.create(N)
        self.common_gates["a"] = jqt.destroy(N)

    @abstractmethod
    def _get_basis_z(self) -> Tuple[jqt.Qarray, jqt.Qarray]:
        """
        Returns:
            plus_z (jqt.Qarray), minus_z (jqt.Qarray): z basis states
        """

    def _get_basis_states(self) -> Dict[str, jqt.Qarray]:
        """
        Construct basis states |+-x>, |+-y>, |+-z>
        """
        plus_z, minus_z = self._get_basis_z()
        return self._gen_basis_states_from_z(plus_z, minus_z)

    def _gen_basis_states_from_z(
        self, plus_z: jqt.Qarray, minus_z: jqt.Qarray
    ) -> Dict[str, jqt.Qarray]:
        """
        Construct basis states |+-x>, |+-y>, |+-z> from |+-z>
        """
        basis: Dict[str, jqt.Qarray] = {}

        # import to make sure that each basis state is a column vec
        # otherwise, transposing a 1D vector will do nothing

        basis["+z"] = plus_z
        basis["-z"] = minus_z

        basis["+x"] = jqt.unit(basis["+z"] + basis["-z"])
        basis["-x"] = jqt.unit(basis["+z"] - basis["-z"])
        basis["+y"] = jqt.unit(basis["+z"] + 1j * basis["-z"])
        basis["-y"] = jqt.unit(basis["+z"] - 1j * basis["-z"])
        return basis

    def jqt2qt(self, state):
        return jqt.jqt2qt(state)

    # gates
    # ======================================================
    # @abstractmethod
    # def stabilize(self) -> None:
    #     """
    #     Stabilizing/measuring syndromes.
    #     """

    @property
    def x_U(self) -> jqt.Qarray:
        """
        Logical X unitary gate.
        """
        return self._gen_pauli_U("x")

    @property
    def x_H(self) -> Optional[jqt.Qarray]:
        """
        Logical X hamiltonian.
        """
        return None

    @property
    def y_U(self) -> jqt.Qarray:
        """
        Logical Y unitary gate.
        """
        return self._gen_pauli_U("y")

    @property
    def y_H(self) -> Optional[jqt.Qarray]:
        """
        Logical Y hamiltonian.
        """
        return None

    @property
    def z_U(self) -> jqt.Qarray:
        """
        Logical Z unitary gate.
        """
        return self._gen_pauli_U("z")

    @property
    def z_H(self) -> Optional[jqt.Qarray]:
        """
        Logical Z hamiltonian.
        """
        return None

    @property
    def h_H(self) -> Optional[jqt.Qarray]:
        """
        Logical Hadamard hamiltonian.
        """
        return None

    @property
    def h_U(self) -> jqt.Qarray:
        """
        Logical Hadamard unitary gate.
        """
        return (
            self.basis["+x"] @ self.basis["+z"].dag()
            + self.basis["-x"] @ self.basis["-z"].dag()
        )

    def _gen_pauli_U(self, basis_state: str) -> jqt.Qarray:
        """
        Generates unitary for Pauli X, Y, Z.

        Args:
            basis_state (str): "x", "y", "z"

        Returns:
            U (jqt.Qarray): Pauli unitary
        """
        H = getattr(self, basis_state + "_H")
        if H is not None:
            return jqt.expm(1.0j * H)

        gate = (
            self.basis["+" + basis_state] @ self.basis["+" + basis_state].dag()
            - self.basis["-" + basis_state] @ self.basis["-" + basis_state].dag()
        )

        return gate

    @property
    def projector(self):
        return (
            self.basis["+z"] @ self.basis["+z"].dag()
            + self.basis["-z"] @ self.basis["-z"].dag()
        )

    @property
    def maximally_mixed_state(self):
        return (1 / 2.0) * self.projector()

    # Plotting
    # ======================================================
    def _prepare_state_plot(self, state):
        """
        Can be overriden.

        E.g. in the case of cavity x transmon system
        return qt.ptrace(state, 0)
        """
        return state

    def plot(self, state, ax=None, qp_type=jqt.WIGNER, **kwargs) -> None:
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(4, 3), dpi=200)
        fig = ax.get_figure()

        if qp_type == jqt.WIGNER:
            vmin = -1
            vmax = 1
        elif qp_type == jqt.QFUNC:
            vmin = 0
            vmax = 1

        w_plt = self._plot_single(state, ax=ax, qp_type=qp_type, **kwargs)

        ax.set_title(qp_type.capitalize() + " Quasi-Probability Dist.")
        ticks = np.linspace(vmin, vmax, 5)
        fig.colorbar(w_plt, ax=ax, ticks=ticks)
        ax.set_xlabel(r"Re$(\alpha)$")
        ax.set_ylabel(r"Im$(\alpha)$")
        fig.tight_layout()

        plt.show()

    def _plot_single(self, state, ax=None, contour=True, qp_type=jqt.WIGNER):
        """
        Assumes state has same dims as initial_state.
        """
        state = self.jqt2qt(state)

        if ax is None:
            _, ax = plt.subplots(1, figsize=(4, 3), dpi=200)

        return jqt.plot_qp(
            state, self.wigner_pts, axs=ax, contour=contour, qp_type=qp_type
        )

    def plot_code_states(self, qp_type: str = jqt.WIGNER, **kwargs):
        """
        Plot |±x⟩, |±y⟩, |±z⟩ code states.

        Args:
            qp_type (str):
                WIGNER or QFUNC

        Return:
            axs: Axes
        """
        fig, axs = plt.subplots(2, 3, figsize=(12, 6), dpi=200)
        if qp_type == jqt.WIGNER:
            cbar_title = r"$\frac{\pi}{2} W(\alpha)$"
            vmin = -1
            vmax = 1
        elif qp_type == jqt.QFUNC:
            cbar_title = r"$\pi Q(\alpha)$"
            vmin = 0
            vmax = 1

        for i, label in enumerate(["+z", "+x", "+y", "-z", "-x", "-y"]):
            state = self._prepare_state_plot(self.basis[label])
            pos = (i // 3, i % 3)
            ax = axs[pos]
            _, w_plt = self._plot_single(state, ax=ax, qp_type=qp_type, **kwargs)
            ax.set_title(f"|{label}" + r"$\rangle$")
            ax.set_xlabel(r"Re[$\alpha$]")
            ax.set_ylabel(r"Im[$\alpha$]")

        fig.suptitle(self.name)
        fig.tight_layout()
        fig.subplots_adjust(right=0.8, hspace=0.2, wspace=0.2)
        fig.align_xlabels(axs)
        fig.align_ylabels(axs)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

        ticks = np.linspace(vmin, vmax, 5)
        fig.colorbar(w_plt, cax=cbar_ax, ticks=ticks)

        cbar_ax.set_title(cbar_title, pad=20)
        fig.tight_layout()
        plt.show()

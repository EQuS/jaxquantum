"""
Cat Code Qubit
"""

from typing import Tuple

from jaxquantum.codes.base import BosonicQubit
import jaxquantum as jqt

import jax.numpy as jnp


class GKPQubit(BosonicQubit):
    """
    GKP Qubit Class.
    """

    name = "gkp"

    def _params_validation(self):
        super()._params_validation()

        if "delta" not in self.params:
            self.params["delta"] = 0.25
        self.params["l"] = 2.0 * jnp.sqrt(jnp.pi)
        s_delta = jnp.sinh(self.params["delta"] ** 2)
        self.params["epsilon"] = s_delta * self.params["l"]

    def _gen_common_gates(self) -> None:
        """
        Overriding this method to add additional common gates.
        """
        super()._gen_common_gates()

        # phase space
        self.common_gates["x"] = (
            self.common_gates["a_dag"] + self.common_gates["a"]
        ) / jnp.sqrt(2.0)
        self.common_gates["p"] = (
            1.0j * (self.common_gates["a_dag"] - self.common_gates["a"]) / jnp.sqrt(2.0)
        )

        # finite energy
        self.common_gates["E"] = jqt.expm(
            -(self.params["delta"] ** 2)
            * self.common_gates["a_dag"]
            @ self.common_gates["a"]
        )
        self.common_gates["E_inv"] = jqt.expm(
            self.params["delta"] ** 2
            * self.common_gates["a_dag"]
            @ self.common_gates["a"]
        )

        # axis
        x_axis, z_axis = self._get_axis()
        y_axis = x_axis + z_axis

        # gates
        X_0 = jqt.expm(1.0j * self.params["l"] / 2.0 * z_axis)
        Z_0 = jqt.expm(1.0j * self.params["l"] / 2.0 * x_axis)
        Y_0 = 1.0j * X_0 @ Z_0
        self.common_gates["X_0"] = X_0
        self.common_gates["Z_0"] = Z_0
        self.common_gates["Y_0"] = Y_0
        self.common_gates["X"] = self._make_op_finite_energy(X_0)
        self.common_gates["Z"] = self._make_op_finite_energy(Z_0)
        self.common_gates["Y"] = self._make_op_finite_energy(Y_0)

        # symmetric stabilizers and gates
        self.common_gates["Z_s_0"] = self._symmetrized_expm(
            1.0j * self.params["l"] / 2.0 * x_axis
        )
        self.common_gates["S_x_0"] = self._symmetrized_expm(
            1.0j * self.params["l"] * z_axis
        )
        self.common_gates["S_z_0"] = self._symmetrized_expm(
            1.0j * self.params["l"] * x_axis
        )
        self.common_gates["S_y_0"] = self._symmetrized_expm(
            1.0j * self.params["l"] * y_axis
        )

    def _get_basis_z(self) -> Tuple[jqt.Qarray, jqt.Qarray]:
        """
        Construct basis states |+-x>, |+-y>, |+-z>.
        step 1: use ideal GKP stabilizers to find ideal GKP |+z> state
        step 2: make ideal eigenvector finite energy
            We want the groundstate of H = E H_0 E⁻¹.
            So, we can begin by find the groundstate of H_0 -> |λ₀⟩
            Then, we know that E|λ₀⟩ = |λ⟩ is the groundstate of H.
            pf. H|λ⟩ = (E H_0 E⁻¹)(E|λ₀⟩) = E H_0 |λ₀⟩ = λ₀ (E|λ₀⟩) = λ₀|λ⟩

        TODO (if necessary):
            Alternatively, we could construct a hamiltonian using
            finite energy stabilizers S_x, S_y, S_z, Z_s. However,
            this would make H = - S_x - S_y - S_z - Z_s non-hermitian.
            Currently, JAX does not support derivatives of jnp.linalg.eig,
            while it does support derivatives of jnp.linalg.eigh.
            Discussion: https://github.com/google/jax/issues/2748
        """

        # step 1: use ideal GKP stabilizers to find ideal GKP |+z> state
        H_0 = (
            -self.common_gates["S_x_0"]
            - self.common_gates["S_y_0"]
            - self.common_gates["S_z_0"]
            - self.common_gates["Z_s_0"]  # bosonic |+z> state
        )

        _, vecs = jnp.linalg.eigh(H_0.data)
        gstate_ideal = jqt.Qarray.create(vecs[:, 0])

        # step 2: make ideal eigenvector finite energy
        gstate = self.common_gates["E"] @ gstate_ideal

        plus_z = jqt.unit(gstate)
        minus_z = jqt.unit(self.common_gates["X"] @ plus_z)
        return plus_z, minus_z

    # utils
    # ======================================================
    def _get_axis(self):
        x_axis = self.common_gates["x"]
        z_axis = -self.common_gates["p"]
        return x_axis, z_axis

    def _make_op_finite_energy(self, op):
        return self.common_gates["E"] @ op @ self.common_gates["E_inv"]

    def _symmetrized_expm(self, op):
        return (jqt.expm(op) + jqt.expm(-1.0 * op)) / 2.0

    # gates
    # ======================================================
    @property
    def x_U(self) -> jqt.Qarray:
        return self.common_gates["X"]

    @property
    def y_U(self) -> jqt.Qarray:
        return self.common_gates["Y"]

    @property
    def z_U(self) -> jqt.Qarray:
        return self.common_gates["Z"]


class RectangularGKPQubit(GKPQubit):
    def _params_validation(self):
        super()._params_validation()
        if "a" not in self.params:
            self.params["a"] = 0.8

    def _get_axis(self):
        a = self.params["a"]
        x_axis = a * self.common_gates["x"]
        z_axis = -1 / a * self.common_gates["p"]
        return x_axis, z_axis


class SquareGKPQubit(GKPQubit):
    def _params_validation(self):
        super()._params_validation()
        self.params["a"] = 1.0


class HexagonalGKPQubit(GKPQubit):
    def _get_axis(self):
        a = jnp.sqrt(2 / jnp.sqrt(3))
        x_axis = a * (
            jnp.sin(jnp.pi / 3.0) * self.common_gates["x"]
            + jnp.cos(jnp.pi / 3.0) * self.common_gates["p"]
        )
        z_axis = a * (-self.common_gates["p"])
        return x_axis, z_axis


## Citations

# Stabilization of Finite-Energy Gottesman-Kitaev-Preskill States
# Baptiste Royer, Shraddha Singh, and S. M. Girvin
# Phys. Rev. Lett. 125, 260509 – Published 31 December 2020

# Quantum error correction of a qubit encoded in grid states of an oscillator.
# Campagne-Ibarcq, P., Eickbusch, A., Touzard, S. et al.
# Nature 584, 368–372 (2020).

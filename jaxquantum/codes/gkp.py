"""
Cat Code Qubit
"""

from typing import Tuple

from jaxquantum.codes.base import BosonicQubit
import jaxquantum as jqt

from jax import jit, lax, vmap

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

    @staticmethod
    def _q_quadrature(q_points, n):
        q_points = q_points.T

        F_0_init = jnp.ones_like(q_points)
        F_1_init = jnp.sqrt(2) * q_points

        def scan_body(n, carry):
            F_0, F_1 = carry
            F_n = (jnp.sqrt(2 / n) * lax.mul(q_points, F_1) - jnp.sqrt(
                (n - 1) / n) * F_0)

            new_carry = (F_1, F_n)

            return new_carry

        initial_carry = (F_0_init, F_1_init)
        final_carry = lax.fori_loop(2, jnp.max(jnp.array([n + 1, 2])),
                                    scan_body, initial_carry)

        q_quad = lax.select(n == 0, F_0_init,
                            lax.select(n == 1, F_1_init,
                                       final_carry[1]))

        q_quad = jnp.pi ** (-0.25) * lax.mul(
            jnp.exp(-lax.pow(q_points, 2) / 2), q_quad)

        return q_quad

    @staticmethod
    def _compute_gkp_basis_z(delta, dim, mu):
        """
        Args:
            mu: state index (0 or 1)

        Returns:
            GKP basis state

        Adapted from code by Lev-Arcady Sellem <lev-arcady.sellem@inria.fr>
        """
        truncat_series = 100

        

        q_points = jnp.sqrt(jnp.pi) * (2 * jnp.arange(truncat_series) + mu)

        def compute_pop(n):
            quadvals = GKPQubit._q_quadrature(q_points, n)
            return jnp.exp(-(delta ** 2) * n) * (
                    2 * jnp.sum(quadvals) - (1 - mu) * quadvals[0])

        psi_even = vmap(compute_pop)(jnp.arange(0, dim, 2))

        psi = jnp.zeros(2 * psi_even.size, dtype=psi_even.dtype)

        psi = psi.at[::2].set(psi_even)

        psi = jqt.Qarray.create(jnp.array(psi))

        return psi.unit()

    


    def _get_basis_z(self) -> Tuple[jqt.Qarray, jqt.Qarray]:
        """
        Construct basis states |+-z>.
        """

        delta = self.params["delta"]
        dim = self.params["N"]
        
        jitted_compute_gkp_basis_z = jit(self._compute_gkp_basis_z, 
                                         static_argnames=("dim",))
        
        plus_z = jitted_compute_gkp_basis_z(delta, dim, 0)
        minus_z = jitted_compute_gkp_basis_z(delta, dim, 1)
        
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

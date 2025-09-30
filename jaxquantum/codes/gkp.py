"""
Cat Code Qubit
"""

from typing import Tuple

from jaxquantum.codes.base import BosonicQubit
import jaxquantum as jqt
from functools import partial
from jax import jit, lax
from jax.nn import one_hot
from jax.scipy.special import gammaln
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

    def _log_hermite(self, n, xs):
        float_dtype = xs.dtype

        def n_is_zero_branch(_xs):
            return jnp.ones_like(_xs, dtype=jnp.int32), jnp.zeros_like(_xs,
                                                                       dtype=float_dtype)

        def n_is_not_zero_branch(_xs):
            zeros = self.hermroots(n, n + 1).astype(float_dtype)
            xs_minus_zeros = _xs[:, None] - zeros[None, :]
            num_negative_factors = (xs_minus_zeros < 0).sum(axis=1)
            signs = (1 - 2 * (num_negative_factors % 2))
            abs_xs_minus_zeros = jnp.abs(xs_minus_zeros)
            log_of_2 = jnp.log(jnp.array(2.0, dtype=float_dtype))
            log_abs = n * log_of_2 + jnp.log(abs_xs_minus_zeros).sum(axis=1)

            return signs.astype(jnp.int32), log_abs.astype(float_dtype)

        return lax.cond(n == 0, n_is_zero_branch, n_is_not_zero_branch,
                            operand=xs)

    def _log_ho_basis_prefactors(self, n, xs):
        return (-n / 2.0 * jnp.log(2.0)
                - 0.5 * gammaln(n + 1.0)
                - 0.25 * jnp.log(jnp.pi)
                - xs ** 2 / 2.0)

    def _log_ho_basis_function(self, n, xs):
        signs_herm, log_abs_herm = self._log_hermite(n, xs)
        log_prefactors = self._log_ho_basis_prefactors(n, xs)
        return signs_herm, log_abs_herm + log_prefactors


    def _calculate_single_amplitude(self, n, xs, delta):
        logenv = -delta ** 2 * n
        sgns, logamps = self._log_ho_basis_function(n, xs)
        amps = sgns * jnp.exp(logenv + logamps)
        return amps.sum()

    def _finite_gkp(self, mu, delta, dim):
        """
        Constructs a finite-energy GKP state in the Fock basis.
        """
        l = 2.0 * jnp.sqrt(jnp.pi)
        xs = l * (jnp.arange(-5000, 5001, dtype=jnp.float64) + mu / 2.0)

        amplitudes = []
        for n in range(dim):
            amp = self._calculate_single_amplitude(n, xs, delta)
            amplitudes.append(amp)

        amplitudes = jnp.array(amplitudes)

        norm = jnp.linalg.norm(amplitudes)
        normalized_amplitudes = amplitudes / (norm + 1e-10)

        return jqt.Qarray.create(normalized_amplitudes)

    def hermcompanion(self, c, dim):
        """Return the scaled companion matrix of c.

        The basis polynomials are scaled so that the companion matrix is
        symmetric when `c` is an Hermite basis polynomial. This provides
        better eigenvalue estimates than the unscaled case and for basis
        polynomials the eigenvalues are guaranteed to be real if
        `jax.numpy.linalg.eigvalsh` is used to obtain them.

        Parameters
        ----------
        c : array_like
            1-D array of Hermite series coefficients ordered from low to high
            degree.

        Returns
        -------
        mat : ndarray
            Scaled companion matrix of dimensions (deg, deg).

        """
        n = dim - 1
        mat = jnp.zeros((n, n), dtype=c.dtype)
        scl = jnp.hstack((1.0, 1.0 / jnp.sqrt(2.0 * jnp.arange(n - 1, 0, -1))))
        scl = jnp.cumprod(scl)[::-1]
        shp = mat.shape
        mat = mat.flatten()
        mat = mat.at[1:: n + 1].set(jnp.sqrt(0.5 * jnp.arange(1, n)))
        mat = mat.at[n:: n + 1].set(jnp.sqrt(0.5 * jnp.arange(1, n)))
        mat = mat.reshape(shp)
        mat = mat.at[:, -1].add(-scl * c[:-1] / (2.0 * c[-1]))
        return mat


    def hermroots(self, n, dim):
        r"""Compute the roots of a Hermite series.

        Return the roots (a.k.a. "zeros") of the polynomial

        .. math:: p(x) = \sum_i c[i] * H_i(x).

        Parameters
        ----------
        c : 1-D array_like
            1-D array of coefficients.

        Returns
        -------
        out : ndarray
            Array of the roots of the series. If all the roots are real,
            then `out` is also real, otherwise it is complex.

        See Also
        --------
        orthax.polynomial.polyroots
        orthax.legendre.legroots
        orthax.laguerre.lagroots
        orthax.chebyshev.chebroots
        orthax.hermite_e.hermeroots

        Notes
        -----
        The root estimates are obtained as the eigenvalues of the companion
        matrix, Roots far from the origin of the complex plane may have large
        errors due to the numerical instability of the series for such
        values. Roots with multiplicity greater than 1 will also show larger
        errors as the value of the series near such points is relatively
        insensitive to errors in the roots. Isolated roots near the origin can
        be improved by a few iterations of Newton's method.

        The Hermite series basis polynomials aren't powers of `x` so the
        results of this function may seem unintuitive.

        Examples
        --------
        from orthax.hermite import hermroots, hermfromroots
        coef = hermfromroots([-1, 0, 1])
        coef
        array([0.   ,  0.25 ,  0.   ,  0.125])
        hermroots(coef)
        array([-1.00000000e+00, -1.38777878e-17,  1.00000000e+00])

        """
        if n + 1 <= 1:
            return jnp.array([])
        if n + 1 == 2:
            return jnp.array([0])

        c = one_hot(n, dim)

        # rotated companion matrix reduces error
        m = self.hermcompanion(c, dim)[::-1, ::-1]
        r = jnp.linalg.eigvalsh(m)
        r = jnp.sort(r)
        return r

    def _get_basis_z(self) -> Tuple[jqt.Qarray, jqt.Qarray]:
        """
        Construct basis states |+-x>, |+-y>, |+-z>.
        """

        plus_z = self._finite_gkp(0, self.params["delta"], self.params[
            "N"])
        minus_z = self._finite_gkp(1, self.params["delta"], self.params[
            "N"])
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

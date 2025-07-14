"""Utility functions"""

from scipy.special import pbdv
from scipy import constants

import jax.scipy as jsp
import jax.numpy as jnp


def factorial_approx(n):
    return jsp.special.gamma(n + 1)


# physics utils

# ----------------


def harm_osc_wavefunction(n, x, l_osc):
    r"""
    Taken from scqubits... not jit-able

    For given quantum number n=0,1,2,... return the value of the harmonic
    oscillator wave function :math:`\psi_n(x) = N H_n(x/l_{osc}) \exp(-x^2/2l_\text{
    osc})`, N being the proper normalization factor.

    Directly uses `scipy.special.pbdv` (implementation of the parabolic cylinder
    function) to mitigate numerical stability issues with the more commonly used
    expression in terms of a Gaussian and a Hermite polynomial factor.

    Parameters
    ----------
    n:
        index of wave function, n=0 is ground state
    x:
        coordinate(s) where wave function is evaluated
    l_osc:
        oscillator length, defined via <0|x^2|0> = l_osc^2/2

    Returns
    -------
        value of harmonic oscillator wave function
    """
    x = 2 * jnp.pi * x
    result = pbdv(n, jnp.sqrt(2.0) * x / l_osc)[0]
    result = result / jnp.sqrt(l_osc * jnp.sqrt(jnp.pi) * factorial_approx(n))
    return result


def calculate_lambda_over_four_resonator_zpf(freq, impedance):
    expected_Z0 = impedance  # Ohms
    expected_E_L_over_E_C = (1 / (4 * expected_Z0)) ** 2 * (
        constants.h**2 / (8 * constants.e**4)
    )
    desired_E_C = jnp.sqrt(freq**2 / expected_E_L_over_E_C / 8)
    desired_E_L = freq**2 / desired_E_C / 8
    storage_q_zpf = (1 / 32 * desired_E_L / desired_E_C) ** (1 / 4)
    return storage_q_zpf, desired_E_C, desired_E_L

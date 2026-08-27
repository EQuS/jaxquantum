"""Utility functions"""

import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax
from scipy import constants


def factorial_approx(n):
    return jsp.special.gamma(n + 1)


# physics utils

# ----------------


def harm_osc_wavefunction(n, x, l_osc):
    r"""
    For given quantum number n=0,1,2,... return the value of the harmonic
    oscillator wave function :math:`\psi_n(x) = N H_n(x/l_{osc}) \exp(-x^2/2l_\text{
    osc})`, N being the proper normalization factor.

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
    return harm_osc_wavefunctions(n + 1, x, l_osc)[n]


def harm_osc_wavefunctions(num_levels, x, l_osc):
    """Evaluate the first ``num_levels`` normalized oscillator wavefunctions."""
    if num_levels < 1:
        raise ValueError("num_levels must be positive")

    coordinate = 2 * jnp.pi * jnp.asarray(x) / l_osc
    psi0 = jnp.exp(-(coordinate**2) / 2) / jnp.sqrt(l_osc * jnp.sqrt(jnp.pi))
    if num_levels == 1:
        return psi0[None]

    psi1 = jnp.sqrt(2.0) * coordinate * psi0

    def next_level(carry, level):
        previous, current = carry
        following = (
            jnp.sqrt(2.0 / (level + 1)) * coordinate * current
            - jnp.sqrt(level / (level + 1)) * previous
        )
        return (current, following), following

    _, remaining = lax.scan(
        next_level,
        (psi0, psi1),
        jnp.arange(1, num_levels - 1),
    )
    return jnp.concatenate((psi0[None], psi1[None], remaining), axis=0)


def calculate_lambda_over_four_resonator_zpf(freq, impedance):
    expected_Z0 = impedance  # Ohms
    expected_E_L_over_E_C = (1 / (4 * expected_Z0)) ** 2 * (
        constants.h**2 / (8 * constants.e**4)
    )
    desired_E_C = jnp.sqrt(freq**2 / expected_E_L_over_E_C / 8)
    desired_E_L = freq**2 / desired_E_C / 8
    storage_q_zpf = (1 / 32 * desired_E_L / desired_E_C) ** (1 / 4)
    return storage_q_zpf, desired_E_C, desired_E_L

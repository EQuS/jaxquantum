""" Units handling."""


import scipy.constants as constants
import jax.numpy as np

# Common Units
# ================================================================================================

FLUX_QUANTUM = constants.h / (2 * constants.e)


def GHz_to_joule(ghz):
    return ghz * 1e9 * constants.h


def joule_to_GHz(joule):
    return joule / (1e9 * constants.h)

def n_thermal(frequency: float, temperature: float) -> float:
    """Calculate the average thermal photon number for a given frequency and temperature.

    Args:
        frequency (float): Frequency in GHz.
        temperature (float): Temperature in Kelvin.

    Returns:
        float: Average thermal photon number.
    """
    k_B = constants.k  # Boltzmann constant in J/K
    h = constants.h  # Planck constant in J·s

    exponent = h * (frequency * 1e9) / (k_B * temperature)
    n_avg = 1 / (np.exp(exponent) - 1)
    return n_avg


# Superconducting Qubit Unit Conversions
# ================================================================================================

FLUX_QUANTUM = constants.h / (2 * constants.e)


def inductive_energy_to_inductance(El):
    """Convert inductive energy E_L to inductance.

    Args:
        El (float): inductive energy in GHz.

    Returns:
        float: Inductance in nH.
    """

    inv_L = GHz_to_joule(El) * (2 * np.pi) ** 2 / (FLUX_QUANTUM**2)
    return 1e9 / inv_L

def inductance_to_inductive_energy(L):
    """Convert inductance to inductive energy E_L.

    Args:
        L (float): Inductance in nH.

    Returns:
        float: Inductive energy in GHz.
    """

    inv_L = 1e9 / L
    El_joules = inv_L * (FLUX_QUANTUM**2) / (2 * np.pi) ** 2
    return joule_to_GHz(El_joules)


def inv_pF_to_Ec(inv_pfarad):
    """
    1/picoFarad -> GHz
    """
    inv_nFarad = inv_pfarad * 1e3
    Gjoule = (constants.e) ** 2 / (2) * inv_nFarad
    return joule_to_GHz(Gjoule * 1e9)


def Ec_to_inv_pF(Ec):
    """
    GHz -> 1/picoFarad
    """
    joule = GHz_to_joule(Ec)
    Gjoule = joule / 1e9
    inv_nFarad = Gjoule / ((constants.e) ** 2 / (2))
    return inv_nFarad * 1e-3


def calculate_resonator_zpf(freq, impedance):
    expected_Z0 = impedance  # Ohms
    expected_E_L_over_E_C = (1 / (4 * expected_Z0)) ** 2 * (
        constants.h**2 / (8 * constants.e**4)
    )
    desired_E_C = np.sqrt(freq**2 / expected_E_L_over_E_C / 8)
    desired_E_L = freq**2 / desired_E_C / 8
    storage_q_zpf = (1 / 32 * desired_E_L / desired_E_C) ** (1 / 4)
    

    # print((desired_E_L / desired_E_C), expected_E_L_over_E_C)

    return storage_q_zpf, desired_E_C, desired_E_L

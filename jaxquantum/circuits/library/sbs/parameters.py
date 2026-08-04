"""Named parameter sets for reproducible sBs simulations."""

from math import pi

__all__ = ("GKP_JULY30", "PARAMETER_SETS")


GKP_JULY30 = {
    "device": {
        "storage_t1": 90.96363465452683e-6,
        "storage_tphi": 1.080e-3,
        "storage_nbar": 0.0,
        "qubit_t1": 438.2766324140162e-6,
        "qubit_t1_cd": 56.38e-6,
        "qubit_tphi": 1 / (1 / 51.56110574223823e-6 - 1 / (2 * 438.2766324140162e-6)),
        "qubit_excited_population": 0.0,
        "qubit_cd_excited_population": 0.5,
        "cd_durations": (416e-9, 2.504e-6, 536e-9),
        "rotation_durations": (144e-9,) * 4,
        "reset_duration": 324e-9,
        "reset_error": 0.01,
    },
    "control": {
        "state_delta": 0.438,
        "delta": 0.42,
        "small_ratio": 1.3,
        "small_displacement_scales": (
            0.868641814887524,
            1.1443416709546,
        ),
        "big_displacement": 2.5062270435214,
        "epsilon_model": "quadratic",
        "final_storage_rotation": 2 * pi * (-1.66112496703863 / 360),
        "alternate_cd_direction": True,
        "max_alpha": 7.7,
    },
    "timing": {
        "cd_delay_ns": (40, 1084, 100),
        "storage_displacement_pulse_ns": 40,
        "cd_pi_padding_ns": 16,
        "qubit_pi_pulse_ns": 144,
        "reset_components_ns": (176, 48, 100),
        "stabilizer_us": 4.356,
        "four_way_round_us": 17.424,
    },
    "provenance": {
        "coherence": (
            "July 30 device snapshot; storage Tphi=1.08 ms was fixed because "
            "the fresh fit was unresolved"
        ),
        "control": "run_gkp_lifetime_measurement BEST_GKP_LIFETIME_CONTROL",
        "timing": "gkp.py formulas with the July 30/31 D9 pulse snapshot",
        "qubit_t1_cd": (
            "56.38 +/- 1.43 us population-contrast lifetime measured at the "
            "big-CD operating point; applied to all three CDs"
        ),
        "qubit_cd_excited_population": (
            "repeated-ECD steady state was approximately one half"
        ),
        "reset_error": "1% sensitivity assumption; reset duration is measured",
    },
}


PARAMETER_SETS = {"gkp_july30": GKP_JULY30}

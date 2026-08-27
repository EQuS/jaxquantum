import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import pbdv

from jaxquantum.devices import SNAIL, Resonator
from jaxquantum.devices.common.utils import (
    harm_osc_wavefunction,
    harm_osc_wavefunctions,
)


def oscillator_reference(num_levels, coordinates, oscillator_length):
    coordinates = 2 * np.pi * np.asarray(coordinates)
    return np.stack(
        [
            pbdv(
                level,
                np.sqrt(2) * coordinates / oscillator_length,
            )[0]
            / np.sqrt(
                oscillator_length * np.sqrt(np.pi) * math.factorial(level)
            )
            for level in range(num_levels)
        ]
    )


def test_harmonic_oscillator_wavefunctions_match_scipy():
    coordinates = jnp.linspace(-1.5, 1.5, 101)
    actual = harm_osc_wavefunctions(52, coordinates, 0.7)
    expected = oscillator_reference(52, coordinates, 0.7)

    assert jnp.allclose(actual, expected, atol=1e-12, rtol=1e-11)
    assert jnp.allclose(harm_osc_wavefunction(17, coordinates, 0.7), expected[17])


def test_harmonic_oscillator_wavefunction_level_edges():
    coordinates = jnp.array([-0.2, 0.0, 0.2])

    assert harm_osc_wavefunctions(1, coordinates, 0.7).shape == (1, 3)
    assert harm_osc_wavefunctions(2, coordinates, 0.7).shape == (2, 3)
    assert jnp.allclose(
        harm_osc_wavefunctions(1, coordinates, 0.7)[0],
        harm_osc_wavefunction(0, coordinates, 0.7),
    )
    with pytest.raises(ValueError, match="num_levels must be positive"):
        harm_osc_wavefunctions(0, coordinates, 0.7)


def test_harmonic_oscillator_wavefunctions_are_jittable_and_differentiable():
    coordinates = jnp.linspace(-0.8, 0.8, 31)
    evaluate = jax.jit(harm_osc_wavefunctions, static_argnums=0)
    actual = evaluate(12, coordinates, 0.6)
    gradient = jax.grad(
        lambda length: jnp.sum(harm_osc_wavefunctions(12, coordinates, length) ** 2)
    )(0.6)

    assert actual.shape == (12, 31)
    assert jnp.isfinite(actual).all()
    assert jnp.isfinite(gradient)


def test_fock_wavefunctions_match_scipy_and_support_jax_transforms():
    coordinates = jnp.linspace(-0.8, 0.8, 31)

    def calculate(ec):
        device = Resonator.create(
            6,
            {"Ec": ec, "El": 1.0},
            N_pre_diag=12,
        )
        return device._calculate_wavefunctions_fock(coordinates)

    device = Resonator.create(6, {"Ec": 0.2, "El": 1.0}, N_pre_diag=12)
    basis = oscillator_reference(
        device.N_pre_diag,
        coordinates,
        float(jnp.real(device.phi_zpf() * jnp.sqrt(2))),
    )
    vectors = device.eig_systems["vecs"][..., :, : device.N]
    expected = jnp.swapaxes(vectors.conj(), -1, -2) @ basis
    actual = jax.jit(calculate)(0.2)
    batched = jax.jit(jax.vmap(calculate))(jnp.array([0.18, 0.2, 0.22]))

    assert actual.shape == (device.N, coordinates.size)
    assert batched.shape == (3, device.N, coordinates.size)
    assert jnp.allclose(actual, expected, atol=1e-12, rtol=1e-11)
    assert jnp.isfinite(jax.grad(lambda ec: jnp.sum(jnp.abs(calculate(ec)) ** 2))(0.2))


def test_charge_wavefunctions_match_loop_and_support_jax_transforms():
    coordinates = jnp.linspace(-0.4, 0.4, 17)

    def calculate(ng):
        device = SNAIL.create(
            4,
            {
                "Ec": 0.2,
                "Ej": 3.0,
                "alpha": 0.25,
                "m": 2,
                "phi_ext": 0.1,
                "ng": ng,
            },
            N_pre_diag=5,
        )
        return device._calculate_wavefunctions_charge(coordinates)

    device = SNAIL.create(
        4,
        {
            "Ec": 0.2,
            "Ej": 3.0,
            "alpha": 0.25,
            "m": 2,
            "phi_ext": 0.1,
            "ng": 0.05,
        },
        N_pre_diag=5,
    )
    labels = jnp.diag(device.original_ops["n"].data)
    basis = jnp.stack(
        [
            jnp.exp(-2j * jnp.pi * label * coordinates) / jnp.sqrt(2 * jnp.pi)
            for label in labels
        ]
    )
    vectors = device.eig_systems["vecs"][..., :, : device.N]
    expected = jnp.swapaxes(vectors.conj(), -1, -2) @ basis
    expected *= jnp.power(1j, jnp.arange(device.N))[:, None]
    actual = jax.jit(calculate)(0.05)
    batched = jax.jit(jax.vmap(calculate))(jnp.array([0.0, 0.05, 0.1]))

    assert actual.shape == (device.N, coordinates.size)
    assert batched.shape == (3, device.N, coordinates.size)
    assert jnp.allclose(actual, expected, atol=1e-12)
    assert jnp.isfinite(
        jax.grad(lambda ng: jnp.sum(jnp.abs(calculate(ng)) ** 2))(0.05)
    )

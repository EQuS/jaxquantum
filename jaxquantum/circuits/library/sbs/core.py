"""Functional sBs circuit primitives."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp

from jaxquantum.circuits.channels import apply_kraus_map, apply_shifted_channel
from jaxquantum.circuits.library.oscillator import (
    CD,
    Dephasing_Reset,
    _amp_damp_coefficients,
    _thermal_coefficients,
)
from jaxquantum.circuits.library.qubit import Rx
from jaxquantum.core.operators import basis, displace, sigmam, sigmap

__all__ = (
    "SBSNoise",
    "SBSNoiseOps",
    "SBSCDGeometry",
    "SBSCDOps",
    "SBSHalfRound",
    "SBSProtocol",
    "noisy_cd_kraus",
    "build_sbs_cd_geometry",
    "build_sbs_half_round",
    "apply_sbs_half_round",
    "oscillator_state",
    "simulate_sbs",
)


@dataclass(frozen=True)
class SBSNoise:
    """Coherence parameters in the same time unit as the sequence."""

    oscillator_t1: float | None = None
    oscillator_tphi: float | None = None
    oscillator_nbar: float = 0.0
    qubit_t1: float | None = None
    qubit_t1_cd: float | Sequence[float] | None = None
    qubit_tphi: float | None = None
    qubit_excited_population: float = 0.0
    reset_error: float = 0.0
    reset_chi: float = 0.0
    qubit_cd_excited_population: float | Sequence[float] | None = None


class SBSNoiseOps(NamedTuple):
    oscillator_coefficients: jax.Array
    oscillator_shifts: jax.Array
    oscillator_dephasing: jax.Array
    qubit_probability: jax.Array
    qubit_excited_population: jax.Array
    qubit_dephasing: jax.Array


class SBSCDGeometry(NamedTuple):
    displacements: jax.Array
    jump_displacements: jax.Array


class SBSCDOps(NamedTuple):
    displacements: jax.Array
    jump_displacements: jax.Array
    relaxation_probability: jax.Array
    excitation_probability: jax.Array


class SBSHalfRound(NamedTuple):
    rotations: jax.Array
    cd: SBSCDOps
    echoes: jax.Array
    rotation_noise: SBSNoiseOps
    cd_noise: SBSNoiseOps
    reset_kraus: jax.Array
    reset_noise: SBSNoiseOps
    microsteps: int


class SBSProtocol(NamedTuple):
    """sBs rounds with an optional alternating sequence."""

    rounds: tuple[SBSHalfRound, ...]
    alternate_rounds: tuple[SBSHalfRound, ...] | None = None


def _lifetime(value):
    return jnp.inf if value is None else jnp.asarray(value)


def _probability(duration, lifetime):
    return -jnp.expm1(-jnp.asarray(duration) / _lifetime(lifetime))


def _noise_ops(
    dimension,
    oscillator_durations,
    qubit_durations,
    noise,
    max_loss,
):
    oscillator_durations = jnp.atleast_1d(jnp.asarray(oscillator_durations))
    qubit_durations = jnp.atleast_1d(jnp.asarray(qubit_durations))
    if oscillator_durations.shape != qubit_durations.shape:
        raise ValueError("oscillator and qubit durations must have equal shapes")

    loss_probability = _probability(
        oscillator_durations,
        noise.oscillator_t1,
    )
    if noise.oscillator_nbar == 0:
        coefficients = _amp_damp_coefficients(
            dimension,
            loss_probability,
            max_loss,
        )
        shifts = jnp.arange(coefficients.shape[-2])
    else:
        coefficients, gains, losses = _thermal_coefficients(
            dimension,
            loss_probability,
            noise.oscillator_nbar,
            max_loss,
        )
        shifts = losses - gains
    indices = jnp.arange(dimension)
    delta = indices[:, None] - indices[None, :]
    oscillator_dephasing = jnp.exp(
        -oscillator_durations[:, None, None]
        / _lifetime(noise.oscillator_tphi)
        * delta**2
    )
    return SBSNoiseOps(
        oscillator_coefficients=coefficients,
        oscillator_shifts=shifts,
        oscillator_dephasing=oscillator_dephasing,
        qubit_probability=_probability(qubit_durations, noise.qubit_t1),
        qubit_excited_population=jnp.broadcast_to(
            jnp.asarray(noise.qubit_excited_population),
            qubit_durations.shape,
        ),
        qubit_dephasing=jnp.exp(-qubit_durations / _lifetime(noise.qubit_tphi)),
    )


@partial(jax.jit, static_argnames=("dimension", "jump_samples"))
def noisy_cd_kraus(
    dimension,
    beta,
    duration,
    t1,
    excited_population=0.0,
    jump_samples=4,
):
    """Return a thermally damped CD with midpoint-sampled jump times."""
    if jump_samples < 1:
        raise ValueError("jump_samples must be positive")
    probability = _probability(duration, t1)
    excited_population = jnp.asarray(excited_population)
    p_down = probability * (1 - excited_population)
    p_up = probability * excited_population
    ground = basis(2, 0).data
    excited = basis(2, 1).data
    Pg = jnp.outer(ground, ground.conj())
    Pe = jnp.outer(excited, excited.conj())
    identity = jnp.eye(dimension)

    no_jump = (
        jnp.kron(jnp.sqrt(1 - p_up) * Pg + jnp.sqrt(1 - p_down) * Pe, identity)
        @ CD(dimension, beta).U.data
    )
    times = (jnp.arange(jump_samples) + 0.5) / jump_samples
    displacements = jax.vmap(lambda time: CD(dimension, beta * (2 * time - 1)).U.data)(
        times
    )
    # JAXQuantum names |g><e| ``sigmap`` and |e><g| ``sigmam``.
    lower = jnp.kron(sigmap().data, identity)
    raise_ = jnp.kron(sigmam().data, identity)
    relaxation = jnp.sqrt(p_down / jump_samples) * jnp.einsum(
        "ij,kjl->kil", lower, displacements
    )
    excitation = jnp.sqrt(p_up / jump_samples) * jnp.einsum(
        "ij,kjl->kil", raise_, displacements
    )
    return jnp.concatenate((no_jump[None], relaxation, excitation))


def build_sbs_cd_geometry(
    dimension,
    displacements,
    *,
    microsteps=1,
    jump_samples=4,
):
    """Precompute the displacement matrices shared by noise variants."""
    if microsteps < 1 or jump_samples < 1:
        raise ValueError("microsteps and jump_samples must be positive")
    times = (jnp.arange(jump_samples) + 0.5) / jump_samples

    def build(beta):
        beta = beta / (2 * microsteps)
        displacement = displace(dimension, beta / 2).data
        jumps = jax.vmap(
            lambda time: displace(
                dimension,
                beta * (2 * time - 1) / 2,
            ).data
        )(times)
        return displacement, jumps

    displacement, jumps = zip(*(build(beta) for beta in displacements))
    return SBSCDGeometry(
        displacements=jnp.stack(displacement),
        jump_displacements=jnp.stack(jumps),
    )


def _build_cd_ops(geometry, durations, t1, excited_population):
    probabilities = _probability(durations, t1)
    excited_population = jnp.asarray(excited_population)
    return SBSCDOps(
        displacements=geometry.displacements,
        jump_displacements=geometry.jump_displacements,
        relaxation_probability=probabilities * (1 - excited_population),
        excitation_probability=probabilities * excited_population,
    )


def build_sbs_half_round(
    dimension: int,
    displacements: Sequence[complex],
    rotations: Sequence[jax.Array],
    cd_durations: Sequence[float],
    rotation_durations: Sequence[float],
    reset_duration: float,
    noise: SBSNoise,
    *,
    echoes: Sequence[jax.Array] | None = None,
    microsteps: int = 1,
    jump_samples: int = 4,
    max_loss: int = 8,
    max_reset: int = 10,
    storage_placement: str = "segment",
    extra_storage_duration: float = 0.0,
    reset_qubit_duration: float | None = None,
    cd_geometry: SBSCDGeometry | None = None,
):
    """Build one three-ECD sBs half-round."""
    if len(displacements) != 3 or len(cd_durations) != 3:
        raise ValueError("sBs requires three displacements and CD durations")
    if len(rotations) != 4 or len(rotation_durations) != 4:
        raise ValueError("sBs requires four rotations and rotation durations")
    if microsteps < 1:
        raise ValueError("microsteps must be positive")
    if storage_placement not in {"segment", "lumped"}:
        raise ValueError("storage_placement must be 'segment' or 'lumped'")

    cd_durations = jnp.asarray(cd_durations)
    rotation_durations = jnp.asarray(rotation_durations)
    if (
        bool(jnp.any(cd_durations < 0))
        or bool(jnp.any(rotation_durations < 0))
        or reset_duration < 0
        or extra_storage_duration < 0
    ):
        raise ValueError("durations must be nonnegative")
    reset_qubit_duration = (
        reset_duration if reset_qubit_duration is None else reset_qubit_duration
    )
    if reset_qubit_duration < 0:
        raise ValueError("durations must be nonnegative")

    echoes = (
        [Rx(jnp.pi).U.data] * 3
        if echoes is None
        else [getattr(echo, "data", echo) for echo in echoes]
    )
    if len(echoes) != 3:
        raise ValueError("echoes must contain three qubit rotations")

    cd_t1 = noise.qubit_t1 if noise.qubit_t1_cd is None else noise.qubit_t1_cd
    substep_durations = cd_durations / (2 * microsteps)
    if cd_geometry is None:
        cd_geometry = build_sbs_cd_geometry(
            dimension,
            displacements,
            microsteps=microsteps,
            jump_samples=jump_samples,
        )
    cd_population = (
        noise.qubit_excited_population
        if noise.qubit_cd_excited_population is None
        else noise.qubit_cd_excited_population
    )
    cd = _build_cd_ops(
        cd_geometry,
        substep_durations,
        cd_t1,
        cd_population,
    )

    if storage_placement == "segment":
        rotation_storage = rotation_durations
        cd_storage = substep_durations
        reset_storage = jnp.asarray([reset_duration + extra_storage_duration])
    else:
        rotation_storage = jnp.zeros_like(rotation_durations)
        cd_storage = jnp.zeros_like(substep_durations)
        reset_storage = jnp.asarray(
            [
                rotation_durations.sum()
                + cd_durations.sum()
                + reset_duration
                + extra_storage_duration
            ]
        )

    cd_noise = SBSNoise(
        oscillator_t1=noise.oscillator_t1,
        oscillator_tphi=noise.oscillator_tphi,
        oscillator_nbar=noise.oscillator_nbar,
        qubit_tphi=noise.qubit_tphi,
        qubit_excited_population=noise.qubit_excited_population,
    )
    reset_kraus = Dephasing_Reset(
        dimension,
        noise.reset_error,
        reset_duration,
        noise.reset_chi,
        max_reset,
    ).KM.data
    return SBSHalfRound(
        rotations=jnp.stack(
            [getattr(rotation, "data", rotation) for rotation in rotations]
        ),
        cd=cd,
        echoes=jnp.stack(echoes),
        rotation_noise=_noise_ops(
            dimension,
            rotation_storage,
            rotation_durations,
            noise,
            max_loss,
        ),
        cd_noise=_noise_ops(
            dimension,
            cd_storage,
            substep_durations,
            cd_noise,
            max_loss,
        ),
        reset_kraus=reset_kraus,
        reset_noise=_noise_ops(
            dimension,
            reset_storage,
            jnp.asarray([reset_qubit_duration]),
            noise,
            max_loss,
        ),
        microsteps=microsteps,
    )


def _apply_joint_kraus(joint, kraus):
    dimension = joint.shape[-1]
    flat = joint.reshape(joint.shape[:-4] + (2 * dimension,) * 2)
    flat = apply_kraus_map(kraus, flat)
    return flat.reshape(flat.shape[:-2] + (2, dimension, 2, dimension))


def _apply_noisy_cd(joint, cd: SBSCDOps, index, inverse=False):
    displacement = cd.displacements[index]
    jumps = cd.jump_displacements[index]
    if inverse:
        displacement = jnp.swapaxes(displacement.conj(), -1, -2)
        jumps = jnp.swapaxes(jumps.conj(), -1, -2)
    conditional = jnp.stack((displacement, jnp.swapaxes(displacement.conj(), -1, -2)))
    no_jump = jnp.einsum(
        "ami,...aibj,bnj->...ambn",
        conditional,
        joint,
        conditional.conj(),
    )
    relaxation = cd.relaxation_probability[index]
    excitation = cd.excitation_probability[index]
    scales = jnp.sqrt(jnp.stack((1 - excitation, 1 - relaxation)))
    no_jump = no_jump * scales[:, None, None, None] * scales[None, None, :, None]

    ground = joint[..., 0, :, 0, :]
    excited = joint[..., 1, :, 1, :]

    def transform(unitary, state):
        return unitary @ state @ jnp.swapaxes(unitary.conj(), -1, -2)

    down = jax.vmap(
        lambda unitary: transform(
            jnp.swapaxes(unitary.conj(), -1, -2),
            excited,
        )
    )(jumps).mean(axis=0)
    up = jax.vmap(lambda unitary: transform(unitary, ground))(jumps).mean(axis=0)
    no_jump = no_jump.at[..., 0, :, 0, :].add(relaxation * down)
    return no_jump.at[..., 1, :, 1, :].add(excitation * up)


def _apply_qubit_unitary(joint, unitary):
    return jnp.einsum(
        "ai,...imjn,cj->...amcn",
        unitary,
        joint,
        unitary.conj(),
    )


def _apply_noise(joint, noise, index):
    moved = jnp.moveaxis(joint, (-3, -1), (-2, -1))
    moved = apply_shifted_channel(
        moved,
        {
            "_coefficients": noise.oscillator_coefficients[index],
            "_shifts": noise.oscillator_shifts,
        },
    )
    joint = jnp.moveaxis(moved, (-2, -1), (-3, -1))
    joint = joint * noise.oscillator_dephasing[index][None, :, None, :]

    probability = noise.qubit_probability[index]
    excited_population = noise.qubit_excited_population[index]
    ground = joint[..., 0, :, 0, :]
    excited = joint[..., 1, :, 1, :]
    coherence = jnp.sqrt(1 - probability) * noise.qubit_dephasing[index]
    output_ground = (1 - excited_population * probability) * ground + (
        1 - excited_population
    ) * probability * excited
    output_excited = (
        excited_population * probability * ground
        + (1 - (1 - excited_population) * probability) * excited
    )
    ge = coherence * joint[..., 0, :, 1, :]
    eg = coherence * joint[..., 1, :, 0, :]
    return jnp.stack(
        (
            jnp.stack((output_ground, ge), axis=-2),
            jnp.stack((eg, output_excited), axis=-2),
        ),
        axis=-4,
    )


def apply_sbs_half_round(joint, ops: SBSHalfRound):
    """Apply one prepared sBs half-round to a joint density matrix."""
    for index in range(3):
        joint = _apply_qubit_unitary(joint, ops.rotations[index])
        joint = _apply_noise(joint, ops.rotation_noise, index)

        def positive(_, state):
            state = _apply_noisy_cd(state, ops.cd, index)
            return _apply_noise(state, ops.cd_noise, index)

        def negative(_, state):
            state = _apply_noisy_cd(state, ops.cd, index, inverse=True)
            return _apply_noise(state, ops.cd_noise, index)

        joint = jax.lax.fori_loop(0, ops.microsteps, positive, joint)
        joint = _apply_qubit_unitary(joint, ops.echoes[index])
        joint = jax.lax.fori_loop(0, ops.microsteps, negative, joint)

    joint = _apply_qubit_unitary(joint, ops.rotations[3])
    joint = _apply_noise(joint, ops.rotation_noise, 3)
    joint = _apply_joint_kraus(joint, ops.reset_kraus)
    return _apply_noise(joint, ops.reset_noise, 0)


def oscillator_state(joint):
    """Trace the ancilla from a joint density matrix."""
    return jnp.trace(joint, axis1=-4, axis2=-2)


@partial(jax.jit, static_argnames=("cycles",))
def simulate_sbs(initial_states, observables, half_rounds, cycles):
    """Evolve batched oscillator states and retain only observables."""
    if cycles < 0:
        raise ValueError("cycles must be non-negative")
    if isinstance(half_rounds, SBSProtocol):
        rounds = half_rounds.rounds
        alternate_rounds = half_rounds.alternate_rounds
    else:
        rounds, alternate_rounds = half_rounds, None
    ground = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    joint = jnp.einsum("ij,...mn->...imjn", ground, initial_states)

    def expectation(state):
        reduced = oscillator_state(state)
        return jnp.einsum("...ij,...ji->...", observables, reduced).real

    initial = expectation(joint)

    def apply_round(state, selected):
        for half_round in selected:
            state = apply_sbs_half_round(state, half_round)
        return state

    def step(state, _):
        state = apply_round(state, rounds)
        return state, expectation(state)

    if alternate_rounds is None:
        joint, values = jax.lax.scan(step, joint, None, length=cycles)
    else:

        def alternate_step(state, cycle):
            state = jax.lax.cond(
                cycle % 2 == 0,
                lambda value: apply_round(value, rounds),
                lambda value: apply_round(value, alternate_rounds),
                state,
            )
            return state, expectation(state)

        joint, values = jax.lax.scan(
            alternate_step,
            joint,
            jnp.arange(cycles),
        )
    return (
        oscillator_state(joint),
        jnp.concatenate(
            (initial[None], values),
        ),
        joint,
    )

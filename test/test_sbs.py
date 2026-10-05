from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import jaxquantum as jqt
import jaxquantum.circuits as jqtc
import jaxquantum.circuits.sbs as legacy_sbs
import jaxquantum.circuits.library.sbs as model
import jaxquantum.circuits.library.sbs.core as sbs


def test_legacy_sbs_import_reexports_public_api():
    assert legacy_sbs.SBSNoise is model.SBSNoise
    assert legacy_sbs.simulate_sbs is model.simulate_sbs


def _joint_state(dimension):
    oscillator = jqt.displace(dimension, 0.2 + 0.1j) @ jqt.basis(dimension, 0)
    rho = (oscillator @ oscillator.dag()).data[None]
    ground = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    return jnp.einsum("ij,bmn->bimjn", ground, rho), rho


def test_noisy_cd_is_complete_and_has_native_ideal_limit():
    dimension = 7
    beta = 0.3 - 0.2j
    kraus = sbs.noisy_cd_kraus(
        dimension,
        beta,
        0.2,
        10.0,
        excited_population=0.07,
        jump_samples=5,
    )
    completeness = jnp.einsum("kji,kjl->il", kraus.conj(), kraus)
    assert jnp.allclose(completeness, jnp.eye(2 * dimension), atol=1e-12)

    joint, _ = _joint_state(dimension)
    positive = sbs.noisy_cd_kraus(
        dimension,
        beta / 2,
        0.0,
        1.0,
        jump_samples=3,
    )
    negative = sbs.noisy_cd_kraus(
        dimension,
        -beta / 2,
        0.0,
        1.0,
        jump_samples=3,
    )
    split = sbs._apply_joint_kraus(joint, positive)
    split = sbs._apply_qubit_unitary(split, jqtc.Rx(jnp.pi).U.data)
    split = sbs._apply_joint_kraus(split, negative)
    native = jqtc.apply_channel(
        jqtc.ECD(dimension, beta),
        joint.reshape(1, 2 * dimension, 2 * dimension),
    ).reshape(joint.shape)
    assert jnp.allclose(split, native, atol=1e-12)
    with pytest.raises(ValueError, match="jump_samples"):
        sbs.noisy_cd_kraus(dimension, beta, 0.2, 10.0, jump_samples=0)


@pytest.mark.parametrize("inverse", [False, True])
def test_direct_noisy_cd_matches_kraus_map(inverse):
    dimension = 6
    beta = 0.3 - 0.2j
    duration = 0.2
    t1 = 10.0
    population = 0.07
    samples = 5
    joint, _ = _joint_state(dimension)
    geometry = sbs.build_sbs_cd_geometry(
        dimension,
        (2 * beta,),
        jump_samples=samples,
    )
    ops = sbs._build_cd_ops(
        geometry,
        jnp.asarray((duration,)),
        t1,
        population,
    )
    actual = sbs._apply_noisy_cd(joint, ops, 0, inverse=inverse)
    expected = sbs._apply_joint_kraus(
        joint,
        sbs.noisy_cd_kraus(
            dimension,
            -beta if inverse else beta,
            duration,
            t1,
            population,
            samples,
        ),
    )
    assert jnp.allclose(actual, expected, atol=1e-12)


def test_direct_reset_matches_kraus_map():
    dimension = 5
    oscillator = jqt.displace(dimension, 0.2j) @ jqt.basis(dimension, 0)
    density = (oscillator @ oscillator.dag()).data
    ancilla = 0.5 * jnp.ones((2, 2))
    joint = jnp.einsum("ij,mn->imjn", ancilla, density)
    probability, duration, chi, order = 0.13, 0.4, 0.7, 5
    half_round = sbs.build_sbs_half_round(
        dimension,
        (0.0,) * 3,
        (jnp.eye(2),) * 4,
        (0.0,) * 3,
        (0.0,) * 4,
        duration,
        sbs.SBSNoise(reset_error=probability, reset_chi=chi),
        max_reset=order,
    )

    actual = sbs._apply_reset(joint, half_round.reset)
    expected = sbs._apply_joint_kraus(
        joint,
        jqtc.Dephasing_Reset(
            dimension,
            probability,
            duration,
            chi,
            order,
        ).KM.data,
    )
    assert jnp.allclose(actual, expected, atol=1e-12)


def test_cd_population_can_differ_from_idle_population():
    ops = sbs.build_sbs_half_round(
        4,
        (0.0, 0.0, 0.0),
        (jnp.eye(2),) * 4,
        (0.2, 0.2, 0.2),
        (0.0,) * 4,
        0.0,
        sbs.SBSNoise(
            qubit_t1_cd=2.0,
            qubit_excited_population=0.4,
            qubit_cd_excited_population=0.1,
        ),
        jump_samples=2,
        max_loss=2,
    )
    total = ops.cd.relaxation_probability + ops.cd.excitation_probability
    assert jnp.allclose(ops.cd.excitation_probability / total, 0.1)


def test_zero_temperature_noise_matches_thermal_channel():
    dimension = 7
    probability = 0.08
    duration = 0.4
    t1 = -duration / np.log1p(-probability)
    ops = sbs._noise_ops(
        dimension,
        (duration,),
        (0.0,),
        sbs.SBSNoise(oscillator_t1=t1),
        max_loss=4,
    )
    _, rho = _joint_state(dimension)
    actual = jqtc.apply_shifted_channel(
        rho,
        {
            "_coefficients": ops.oscillator_coefficients[0],
            "_shifts": ops.oscillator_shifts,
        },
    )
    expected = jqtc.apply_channel(
        jqtc.Thermal_Ch(dimension, probability, 0.0, 4),
        rho,
    )
    assert jnp.allclose(actual, expected, atol=1e-12)


def test_half_round_matches_manual_ideal_sequence():
    dimension = 8
    displacements = (0.15, -0.4j, 0.21)
    rotations = (
        jqtc.Ry(jnp.pi / 2).U.data,
        jqtc.Rx(-jnp.pi / 2).U.data,
        jqtc.Rx(jnp.pi / 2).U.data,
        jqtc.Ry(-jnp.pi / 2).U.data,
    )
    ops = sbs.build_sbs_half_round(
        dimension,
        displacements,
        rotations,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        0.0,
        sbs.SBSNoise(),
        jump_samples=2,
        max_loss=2,
    )
    joint, _ = _joint_state(dimension)
    actual = sbs.oscillator_state(sbs.apply_sbs_half_round(joint, ops))

    manual = joint
    for rotation, beta in zip(rotations[:3], displacements):
        manual = sbs._apply_qubit_unitary(manual, rotation)
        manual = jqtc.apply_channel(
            jqtc.ECD(dimension, beta),
            manual.reshape(1, 2 * dimension, 2 * dimension),
        ).reshape(manual.shape)
    manual = sbs._apply_qubit_unitary(manual, rotations[3])
    assert jnp.allclose(actual, sbs.oscillator_state(manual), atol=1e-12)


@pytest.mark.parametrize("microsteps", [1, 4])
def test_segment_dephasing_uses_coherence_time(microsteps):
    dimension = 5
    tphi = 20.0
    cd_durations = (1.0, 2.0, 3.0)
    rotation_durations = (0.2, 0.3, 0.4, 0.5)
    reset_duration = 0.6
    ops = sbs.build_sbs_half_round(
        dimension,
        (0.0, 0.0, 0.0),
        (jnp.eye(2),) * 4,
        cd_durations,
        rotation_durations,
        reset_duration,
        sbs.SBSNoise(oscillator_tphi=tphi),
        microsteps=microsteps,
        jump_samples=2,
        max_loss=2,
    )
    rho = jnp.zeros((1, dimension, dimension), dtype=complex)
    rho = rho.at[0, :2, :2].set(0.5)
    ground = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    joint = jnp.einsum("ij,bmn->bimjn", ground, rho)
    result = sbs.oscillator_state(sbs.apply_sbs_half_round(joint, ops))
    duration = sum(cd_durations) + sum(rotation_durations) + reset_duration
    assert jnp.allclose(
        result[0, 0, 1] / rho[0, 0, 1],
        jnp.exp(-duration / tphi),
        atol=1e-12,
    )


def test_shared_cat_protocol_matches_native_ideal_round():
    dimension = 10
    nbar = 2.0
    protocol = model.cat_protocol(dimension, nbar, enabled=())
    initial, observables = model.cat_problem(dimension, nbar)
    final, values, _ = jqtc.simulate_sbs(
        initial,
        observables,
        protocol,
        2,
    )
    assert final.shape == initial.shape
    assert values.shape == (3, 2)
    assert jnp.allclose(
        jnp.trace(final, axis1=-2, axis2=-1),
        1,
        atol=1e-12,
    )


def test_cat_protocol_alternates_displacement_direction():
    initial, observables = model.cat_problem(8, 1.0)
    protocol = model.cat_protocol(
        8,
        1.0,
        enabled=(),
        jump_samples=2,
        max_loss=2,
        alternate_cd_direction=True,
    )
    final, _, _ = jqtc.simulate_sbs(initial, observables, protocol, 2)

    ground = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    joint = jnp.einsum("ij,...mn->...imjn", ground, initial)
    joint = sbs.apply_sbs_half_round(joint, protocol.rounds[0])
    joint = sbs.apply_sbs_half_round(joint, protocol.alternate_rounds[0])
    assert jnp.allclose(final, sbs.oscillator_state(joint), atol=1e-12)


def test_measured_cat_device_round_time():
    device = model.CAT_MEASURED_DEVICE
    t2_echo = 1 / (1 / device.qubit_tphi + 1 / (2 * device.qubit_t1))
    assert np.isclose(model.round_time(device), 6.348e-6)
    assert np.isclose(t2_echo, 51.56110574223823e-6)


def test_july30_gkp_parameters_match_experiment():
    parameters = model.GKP_JULY30
    device = model.GKP_JULY30_DEVICE
    control = parameters["control"]
    z_displacements, _ = model.gkp_displacements(
        control["delta"],
        control["small_ratio"],
        control["small_displacement_scales"],
        control["big_displacement"],
        control["epsilon_model"],
        1.0,
    )
    assert np.allclose(
        np.abs(np.asarray(z_displacements)),
        (0.192043340194444, 2.5062270435214, 0.328895237325122),
    )
    assert np.isclose(model.round_time(device), 4.356e-6)
    assert np.isclose(4 * model.round_time(device), 17.424e-6)
    assert np.isclose(device.qubit_t1_cd, 56.38e-6)


def test_gkp_protocol_supports_experimental_four_way_control():
    control = model.GKP_JULY30["control"]
    protocol = model.gkp_protocol(
        7,
        device=model.GKP_JULY30_DEVICE,
        enabled=("qubit_t1_cd",),
        jump_samples=2,
        max_loss=2,
        max_reset=2,
        **{
            key: control[key]
            for key in (
                "delta",
                "small_ratio",
                "small_displacement_scales",
                "big_displacement",
                "epsilon_model",
                "final_storage_rotation",
                "alternate_cd_direction",
            )
        },
    )
    assert isinstance(protocol, jqtc.SBSProtocol)
    assert len(protocol.rounds) == len(protocol.alternate_rounds) == 2
    reference = model.gkp_protocol(
        7,
        device=model.GKP_JULY30_DEVICE,
        enabled=("qubit_t1_cd",),
        jump_samples=2,
        max_loss=2,
        max_reset=2,
        delta=control["delta"],
        small_ratio=control["small_ratio"],
        small_displacement_scales=control["small_displacement_scales"],
        big_displacement=control["big_displacement"],
        epsilon_model=control["epsilon_model"],
    )[0]
    phase = jnp.exp(-1j * control["final_storage_rotation"] * jnp.arange(7))
    phase_factor = phase[:, None] * phase.conj()[None, :]
    assert jnp.allclose(
        protocol.rounds[0].reset.phase_factor,
        phase_factor * reference.reset.phase_factor,
    )
    for forward, reverse in zip(protocol.rounds, protocol.alternate_rounds):
        assert jnp.allclose(
            reverse.cd.displacements,
            jnp.swapaxes(forward.cd.displacements.conj(), -1, -2),
        )
        total_cd_probability = (
            forward.cd.relaxation_probability + forward.cd.excitation_probability
        )
        assert jnp.allclose(
            forward.cd.relaxation_probability,
            forward.cd.excitation_probability,
        )
        assert jnp.all(total_cd_probability > 0)


def test_gkp_protocol_accepts_per_cd_thermalization():
    device = replace(
        model.GKP_JULY30_DEVICE,
        qubit_t1_cd=(100e-6, 50e-6, 25e-6),
        qubit_cd_excited_population=(0.0, 0.25, 0.5),
    )
    half_round = model.gkp_protocol(
        6,
        device=device,
        enabled=("qubit_t1_cd",),
        jump_samples=2,
        max_loss=2,
        max_reset=2,
    )[0]
    total = half_round.cd.relaxation_probability + half_round.cd.excitation_probability
    assert jnp.allclose(
        total,
        -jnp.expm1(
            -jnp.asarray(device.cd_durations) / (2 * jnp.asarray(device.qubit_t1_cd))
        ),
    )
    assert jnp.allclose(
        half_round.cd.excitation_probability / total,
        jnp.asarray(device.qubit_cd_excited_population),
    )


def test_gkp_problem_supports_both_logical_axes():
    for kind in ("x", "z"):
        states, observables = model.gkp_problem(8, 0.438, kind)
        assert states.shape == observables.shape == (2, 8, 8)
    with pytest.raises(ValueError, match="kind"):
        model.gkp_problem(8, kind="y")


def test_prepared_protocol_reuses_cd_geometry():
    build = model.prepare_gkp_protocol(8, jump_samples=2, max_loss=2)
    baseline = build(())
    noisy = build(model.ERROR_CHANNELS)
    for baseline_half, noisy_half in zip(baseline, noisy):
        assert baseline_half.cd.displacements is noisy_half.cd.displacements
        assert baseline_half.cd.jump_displacements is noisy_half.cd.jump_displacements


def test_error_budget_subtracts_baseline_and_reports_context():
    initial, observables = model.cat_problem(7, 1.0)
    channels = ("storage_t1", "storage_tphi")
    budget = model.compute_error_budget(
        lambda enabled: model.cat_protocol(
            7,
            1.0,
            enabled=enabled,
            jump_samples=2,
            max_loss=2,
        ),
        initial,
        observables,
        cycles=5,
        cycle_time=model.round_time(model.CAT_DEVICE),
        channels=channels,
        fit_start=1,
    )
    for channel in channels:
        assert np.isclose(
            budget.isolated_increments[channel],
            budget.isolated[channel].rate - budget.baseline.rate,
        )
        assert np.isclose(
            budget.context_increments[channel],
            budget.all_on.rate - budget.without[channel].rate,
        )
    assert set(budget.ranking) == set(channels)


def test_batched_error_budget_matches_sequential():
    initial, observables = model.gkp_problem(7)
    channels = ("storage_t1", "qubit_t1")
    build = model.prepare_gkp_protocol(
        7,
        jump_samples=2,
        max_loss=2,
        max_reset=2,
    )
    arguments = (
        build,
        initial,
        observables,
        5,
        model.round_time(model.GKP_DEVICE, 2),
    )
    batched = model.compute_error_budget(
        *arguments,
        channels=channels,
        fit_start=1,
    )
    sequential = model.compute_error_budget(
        *arguments,
        channels=channels,
        fit_start=1,
        batched=False,
    )
    assert np.isclose(batched.all_on.rate, sequential.all_on.rate)
    assert np.allclose(
        list(batched.context_increments.values()),
        list(sequential.context_increments.values()),
    )


def test_single_decay_variant_matches_direct_simulation():
    initial, observables = model.cat_problem(7, 1.0)
    protocol = model.cat_protocol(
        7,
        1.0,
        enabled=(),
        jump_samples=2,
        max_loss=2,
    )
    arguments = (
        initial,
        observables,
        protocol,
        5,
        model.round_time(model.CAT_DEVICE),
    )
    direct = model.simulate_decay(*arguments, fit_start=1)
    mapped = model.simulate_decay_variants(
        initial,
        observables,
        [protocol],
        5,
        model.round_time(model.CAT_DEVICE),
        fit_start=1,
    )[0]
    assert np.allclose(mapped.contrast, direct.contrast)
    assert np.isclose(mapped.lifetime, direct.lifetime)


def test_batched_alternating_protocol_matches_sequential():
    initial, observables = model.cat_problem(7, 1.0)
    protocols = [
        model.cat_protocol(
            7,
            1.0,
            delta=delta,
            enabled=(),
            jump_samples=2,
            max_loss=2,
            alternate_cd_direction=True,
        )
        for delta in (0.5, 0.6)
    ]
    batched = model.simulate_decay_variants(
        initial,
        observables,
        protocols,
        5,
        model.round_time(model.CAT_DEVICE),
        fit_start=1,
    )
    sequential = [
        model.simulate_decay(
            initial,
            observables,
            protocol,
            5,
            model.round_time(model.CAT_DEVICE),
            fit_start=1,
        )
        for protocol in protocols
    ]
    assert np.allclose(
        [result.lifetime for result in batched],
        [result.lifetime for result in sequential],
    )


def test_decay_fit_floor_excludes_low_signal_tail():
    times = np.arange(6, dtype=float)
    contrast = np.exp(-times / 2)
    contrast[-2:] = (0.08, 0.09)
    lifetime, rate, r2 = model.fit_decay(
        times,
        contrast,
        start=0,
        floor=0.1,
    )
    assert np.isclose(lifetime, 2.0)
    assert np.isclose(rate, 0.5)
    assert np.isclose(r2, 1.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"displacements": (0.0, 0.0)}, "three displacements"),
        ({"rotations": (jnp.eye(2),) * 3}, "four rotations"),
        ({"microsteps": 0}, "microsteps"),
        ({"jump_samples": 0}, "jump_samples"),
        ({"max_loss": -1}, "max_loss"),
        ({"max_reset": 1}, "max_reset"),
        ({"storage_placement": "invalid"}, "storage_placement"),
    ],
)
def test_half_round_validates_structure(kwargs, message):
    arguments = {
        "dimension": 4,
        "displacements": (0.0, 0.0, 0.0),
        "rotations": (jnp.eye(2),) * 4,
        "cd_durations": (0.0, 0.0, 0.0),
        "rotation_durations": (0.0, 0.0, 0.0, 0.0),
        "reset_duration": 0.0,
        "noise": sbs.SBSNoise(),
        "jump_samples": 2,
        "max_loss": 2,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        sbs.build_sbs_half_round(**arguments)


def test_simulate_sbs_rejects_negative_cycles():
    with pytest.raises(ValueError, match="non-negative"):
        sbs.simulate_sbs(
            jnp.eye(2)[None],
            jnp.eye(2)[None],
            (),
            -1,
        )

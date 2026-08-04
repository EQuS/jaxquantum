"""Shared cat and GKP sBs device simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

import jaxquantum as jqt
import jaxquantum.circuits as jqtc
import jaxquantum.codes as jqcodes
from experiments.circuit.sbs_parameters import GKP_JULY30


ERROR_CHANNELS = (
    "storage_t1",
    "storage_tphi",
    "qubit_t1",
    "qubit_tphi",
    "qubit_t1_cd",
    "reset",
)


@dataclass(frozen=True)
class DeviceParameters:
    storage_t1: float
    storage_tphi: float
    storage_nbar: float
    qubit_t1: float
    qubit_t1_cd: float | tuple[float, float, float]
    qubit_tphi: float
    qubit_excited_population: float
    cd_durations: tuple[float, float, float]
    rotation_durations: tuple[float, float, float, float]
    reset_duration: float
    reset_error: float = 0.0
    reset_chi: float = 0.0
    extra_storage_duration: float = 0.0
    qubit_cd_excited_population: float | tuple[float, float, float] | None = None


CAT_DEVICE = DeviceParameters(
    storage_t1=100e-6,
    storage_tphi=0.87e-3,
    storage_nbar=0.0,
    qubit_t1=400e-6,
    qubit_t1_cd=400e-6,
    qubit_tphi=1 / (1 / 90e-6 - 1 / (2 * 400e-6)),
    qubit_excited_population=0.0,
    cd_durations=(1.088e-6, 2.688e-6, 1.088e-6),
    rotation_durations=(144e-9,) * 4,
    reset_duration=324e-9,
    extra_storage_duration=28e-9,
)

# July 30 measured values used in the cat-sBs lifetime audit.
CAT_MEASURED_DEVICE = DeviceParameters(
    storage_t1=90.96363465452683e-6,
    storage_tphi=1.080e-3,
    storage_nbar=0.0,
    qubit_t1=438.2766324140162e-6,
    qubit_t1_cd=438.2766324140162e-6,
    qubit_tphi=1 / (1 / 51.56110574223823e-6 - 1 / (2 * 438.2766324140162e-6)),
    qubit_excited_population=0.0,
    cd_durations=(1.088e-6, 3.088e-6, 1.088e-6),
    rotation_durations=(144e-9,) * 4,
    reset_duration=480e-9,
    extra_storage_duration=28e-9,
)


GKP_DEVICE = DeviceParameters(
    storage_t1=606e-6,
    storage_tphi=24e-3,
    storage_nbar=0.0,
    qubit_t1=280e-6,
    qubit_t1_cd=280e-6,
    qubit_tphi=1 / (1 / 238e-6 - 1 / (2 * 280e-6)),
    qubit_excited_population=0.04,
    cd_durations=(470e-9, 676e-9, 230e-9),
    rotation_durations=(0.0, 32e-9, 32e-9, 0.0),
    reset_duration=2.380e-6,
    reset_error=0.01,
)

GKP_JULY30_DEVICE = DeviceParameters(**GKP_JULY30["device"])


GKP_LEGACY_DEVICE = DeviceParameters(
    storage_t1=90e-6,
    storage_tphi=1e-3,
    storage_nbar=0.00702828004369955,
    qubit_t1=200e-6,
    qubit_t1_cd=30e-6,
    qubit_tphi=30e-6,
    qubit_excited_population=0.430562654241043,
    cd_durations=(400e-9, 1.2e-6, 400e-9),
    rotation_durations=(0.0, 144e-9, 144e-9, 0.0),
    reset_duration=236e-9,
    reset_error=0.1,
    reset_chi=2 * np.pi * 30e3,
)


@dataclass
class DecayResult:
    lifetime: float
    rate: float
    r2: float
    contrast: np.ndarray
    trace_error: float
    hermiticity_error: float
    minimum_eigenvalue: float


@dataclass
class ErrorBudget:
    baseline: DecayResult
    all_on: DecayResult
    isolated: dict[str, DecayResult]
    without: dict[str, DecayResult]
    isolated_increments: dict[str, float]
    context_increments: dict[str, float]
    interaction_rate: float

    @property
    def ranking(self):
        return sorted(
            self.context_increments,
            key=self.context_increments.get,
            reverse=True,
        )

    def summary(self):
        return {
            "baseline_rate": self.baseline.rate,
            "all_on_rate": self.all_on.rate,
            "isolated_increments": self.isolated_increments,
            "context_increments": self.context_increments,
            "interaction_rate": self.interaction_rate,
            "ranking": self.ranking,
        }


def round_time(device: DeviceParameters, half_rounds=1):
    return half_rounds * (
        sum(device.cd_durations)
        + sum(device.rotation_durations)
        + device.reset_duration
        + device.extra_storage_duration
    )


def _enabled(value, name, enabled):
    return value if name in enabled else jnp.inf


def _noise(device, enabled):
    return jqtc.SBSNoise(
        oscillator_t1=_enabled(
            device.storage_t1,
            "storage_t1",
            enabled,
        ),
        oscillator_tphi=_enabled(
            device.storage_tphi,
            "storage_tphi",
            enabled,
        ),
        oscillator_nbar=device.storage_nbar,
        qubit_t1=_enabled(device.qubit_t1, "qubit_t1", enabled),
        qubit_t1_cd=_enabled(
            device.qubit_t1_cd,
            "qubit_t1_cd",
            enabled,
        ),
        qubit_tphi=_enabled(
            device.qubit_tphi,
            "qubit_tphi",
            enabled,
        ),
        qubit_excited_population=device.qubit_excited_population,
        qubit_cd_excited_population=device.qubit_cd_excited_population,
        reset_error=device.reset_error if "reset" in enabled else 0.0,
        reset_chi=device.reset_chi,
    )


def cat_protocol(
    dimension,
    nbar,
    *,
    delta=0.6,
    ratio=3.125,
    device=CAT_DEVICE,
    enabled=ERROR_CHANNELS,
    microsteps=1,
    jump_samples=4,
    max_loss=8,
    cd_geometry=None,
    alternate_cd_direction=False,
):
    """Build the nominal cat sBs measurement round."""
    alpha = jnp.sqrt(nbar)
    small = jnp.pi * delta**2 / (4 * alpha)
    displacements = (small, -1j * jnp.pi / (2 * alpha), ratio * small)
    rotations = (
        jqtc.Ry(jnp.pi / 2).U.data,
        jqtc.Rx(-jnp.pi / 2).U.data,
        jqtc.Rx(-jnp.pi / 2).U.data,
        jqtc.Ry(jnp.pi / 2).U.data,
    )

    def build(values, geometry):
        return jqtc.build_sbs_half_round(
            dimension,
            values,
            rotations,
            device.cd_durations,
            device.rotation_durations,
            device.reset_duration,
            _noise(device, set(enabled)),
            microsteps=microsteps,
            jump_samples=jump_samples,
            max_loss=max_loss,
            storage_placement="lumped",
            extra_storage_duration=device.extra_storage_duration,
            reset_qubit_duration=0.0,
            cd_geometry=geometry,
        )

    forward = build(displacements, cd_geometry)
    if not alternate_cd_direction:
        return (forward,)
    reverse_geometry = jqtc.SBSCDGeometry(
        jnp.swapaxes(forward.cd.displacements.conj(), -1, -2),
        jnp.swapaxes(forward.cd.jump_displacements.conj(), -1, -2),
    )
    reverse = build(tuple(-value for value in displacements), reverse_geometry)
    return jqtc.SBSProtocol((forward,), (reverse,))


def gkp_protocol(
    dimension,
    *,
    delta=0.428,
    small_ratio=1.083,
    small_displacement_scales=(1.0, 1.0),
    big_displacement=None,
    epsilon_model="sinh",
    final_storage_rotation=0.0,
    alternate_cd_direction=False,
    length_scale=1.0,
    device=GKP_DEVICE,
    enabled=ERROR_CHANNELS,
    microsteps=1,
    jump_samples=4,
    max_loss=8,
    max_reset=12,
    cd_geometries=None,
):
    """Build the two complementary GKP sBs half-rounds."""
    z_displacements, x_displacements = _gkp_displacements(
        delta,
        small_ratio,
        small_displacement_scales,
        big_displacement,
        epsilon_model,
        length_scale,
    )
    rotations = (
        jqtc.Ry(jnp.pi / 2).U.data,
        jqtc.Rx(-jnp.pi / 2).U.data,
        jqtc.Rx(jnp.pi / 2).U.data,
        jqtc.Ry(-jnp.pi / 2).U.data,
    )
    noise = _noise(device, set(enabled))
    if cd_geometries is None:
        cd_geometries = (None, None)

    def build(displacements, geometry):
        half_round = jqtc.build_sbs_half_round(
            dimension,
            displacements,
            rotations,
            device.cd_durations,
            device.rotation_durations,
            device.reset_duration,
            noise,
            microsteps=microsteps,
            jump_samples=jump_samples,
            max_loss=max_loss,
            max_reset=max_reset,
            storage_placement="segment",
            extra_storage_duration=device.extra_storage_duration,
            cd_geometry=geometry,
        )
        if final_storage_rotation:
            phase = jnp.diag(
                jnp.exp(-1j * final_storage_rotation * jnp.arange(dimension))
            )
            rotation = jnp.kron(jnp.eye(2), phase)
            half_round = half_round._replace(
                reset_kraus=jnp.einsum(
                    "ij,kjl->kil",
                    rotation,
                    half_round.reset_kraus,
                )
            )
        return half_round

    forward = (
        build(z_displacements, cd_geometries[0]),
        build(x_displacements, cd_geometries[1]),
    )
    if not alternate_cd_direction:
        return forward
    reverse_geometries = tuple(
        jqtc.SBSCDGeometry(
            jnp.swapaxes(half_round.cd.displacements.conj(), -1, -2),
            jnp.swapaxes(half_round.cd.jump_displacements.conj(), -1, -2),
        )
        for half_round in forward
    )
    reverse = tuple(
        build(
            tuple(-value for value in displacements),
            geometry,
        )
        for displacements, geometry in zip(
            (z_displacements, x_displacements),
            reverse_geometries,
        )
    )
    return jqtc.SBSProtocol(forward, reverse)


def _gkp_displacements(
    delta,
    small_ratio,
    small_displacement_scales,
    big_displacement,
    epsilon_model,
    length_scale,
):
    length = jnp.sqrt(2 * jnp.pi) * length_scale
    if epsilon_model == "sinh":
        epsilon = jnp.sinh(delta**2) * length
    elif epsilon_model == "quadratic":
        epsilon = delta**2 * length
    else:
        raise ValueError("epsilon_model must be 'sinh' or 'quadratic'")
    scales = jnp.asarray(small_displacement_scales)
    small = (epsilon / 2 * scales[0], small_ratio * epsilon / 2 * scales[1])
    big = length if big_displacement is None else big_displacement
    return (
        (small[0], -1j * big, small[1]),
        (1j * small[0], big, 1j * small[1]),
    )


def prepare_gkp_protocol(
    dimension,
    *,
    delta=0.428,
    small_ratio=1.083,
    small_displacement_scales=(1.0, 1.0),
    big_displacement=None,
    epsilon_model="sinh",
    final_storage_rotation=0.0,
    alternate_cd_direction=False,
    length_scale=1.0,
    device=GKP_DEVICE,
    microsteps=1,
    jump_samples=4,
    max_loss=8,
    max_reset=12,
):
    """Return an error-channel builder with shared pulse geometry."""
    displacements = _gkp_displacements(
        delta,
        small_ratio,
        small_displacement_scales,
        big_displacement,
        epsilon_model,
        length_scale,
    )
    geometries = tuple(
        jqtc.build_sbs_cd_geometry(
            dimension,
            values,
            microsteps=microsteps,
            jump_samples=jump_samples,
        )
        for values in displacements
    )

    def build(enabled):
        return gkp_protocol(
            dimension,
            delta=delta,
            small_ratio=small_ratio,
            small_displacement_scales=small_displacement_scales,
            big_displacement=big_displacement,
            epsilon_model=epsilon_model,
            final_storage_rotation=final_storage_rotation,
            alternate_cd_direction=alternate_cd_direction,
            length_scale=length_scale,
            device=device,
            enabled=enabled,
            microsteps=microsteps,
            jump_samples=jump_samples,
            max_loss=max_loss,
            max_reset=max_reset,
            cd_geometries=geometries,
        )

    return build


def prepare_cat_protocol(
    dimension,
    nbar,
    *,
    delta=0.6,
    ratio=3.125,
    device=CAT_DEVICE,
    microsteps=1,
    jump_samples=4,
    max_loss=8,
    alternate_cd_direction=False,
):
    """Return an error-channel builder with shared pulse geometry."""
    alpha = jnp.sqrt(nbar)
    small = jnp.pi * delta**2 / (4 * alpha)
    displacements = (small, -1j * jnp.pi / (2 * alpha), ratio * small)
    geometry = jqtc.build_sbs_cd_geometry(
        dimension,
        displacements,
        microsteps=microsteps,
        jump_samples=jump_samples,
    )

    def build(enabled):
        return cat_protocol(
            dimension,
            nbar,
            delta=delta,
            ratio=ratio,
            device=device,
            enabled=enabled,
            microsteps=microsteps,
            jump_samples=jump_samples,
            max_loss=max_loss,
            cd_geometry=geometry,
            alternate_cd_direction=alternate_cd_direction,
        )

    return build


def cat_problem(dimension, nbar, kind="bit"):
    """Return the two cat states and observable defining a contrast."""
    alpha = jnp.sqrt(nbar)
    plus = jqt.displace(dimension, alpha) @ jqt.basis(dimension, 0)
    minus = jqt.displace(dimension, -alpha) @ jqt.basis(dimension, 0)
    if kind == "bit":
        q = (jqt.destroy(dimension) + jqt.create(dimension)).data / jnp.sqrt(2)
        values, vectors = jnp.linalg.eigh(q)
        observable = (vectors * jnp.sign(values)) @ vectors.conj().T
        states = (plus, minus)
    elif kind == "phase":
        states = (jqt.unit(plus + minus), jqt.unit(plus - minus))
        observable = jnp.diag((-1.0) ** jnp.arange(dimension))
    else:
        raise ValueError("kind must be 'bit' or 'phase'")
    density = jnp.stack([(state @ state.dag()).data for state in states])
    return density, jnp.broadcast_to(observable, density.shape)


def gkp_problem(dimension, delta=0.428, kind="x"):
    """Return the two GKP states and observable defining a logical contrast."""
    kind = kind.lower()
    if kind not in ("x", "z"):
        raise ValueError("kind must be 'x' or 'z'")
    code = jqcodes.GKPQubit({"delta": delta, "N": dimension})
    states = (code.basis[f"-{kind}"], code.basis[f"+{kind}"])
    density = jnp.stack([(state @ state.dag()).data for state in states])
    observable = code.common_gates[f"{kind.upper()}_0"].data
    return density, jnp.broadcast_to(observable, density.shape)


def fit_decay(times, contrast, start=4, floor=1e-10):
    values = np.abs(np.asarray(contrast))
    times = np.asarray(times)
    valid = np.isfinite(values) & (values > floor) & (np.arange(values.size) >= start)
    if valid.sum() < 3:
        raise ValueError("at least three finite decay samples are required")
    x = times[valid]
    y = np.log(values[valid])
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual = np.sum((y - fitted) ** 2)
    total = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 if total == 0 else 1 - residual / total
    lifetime = np.inf if slope >= 0 else -1 / slope
    return lifetime, max(0.0, -slope), r2


def simulate_decay(
    initial_states,
    observables,
    half_rounds,
    cycles,
    cycle_time,
    *,
    fit_start=4,
    fit_floor=1e-10,
):
    final, values, _ = jqtc.simulate_sbs(
        initial_states,
        observables,
        half_rounds,
        cycles,
    )
    final, values = jax.device_get((final, values))
    return _analyze_decay(
        final,
        values,
        cycles,
        cycle_time,
        fit_start,
        fit_floor,
    )


def _analyze_decay(final, values, cycles, cycle_time, fit_start, fit_floor):
    contrast = np.asarray(values[:, 0] - values[:, 1])
    times = np.arange(cycles + 1) * cycle_time
    lifetime, rate, r2 = fit_decay(
        times,
        contrast,
        start=fit_start,
        floor=fit_floor,
    )
    traces = np.trace(final, axis1=-2, axis2=-1)
    hermiticity = np.max(np.abs(final - final.conj().swapaxes(-1, -2)))
    minimum_eigenvalue = np.min(np.linalg.eigvalsh(final))
    return DecayResult(
        lifetime=lifetime,
        rate=rate,
        r2=r2,
        contrast=contrast,
        trace_error=float(np.max(np.abs(traces - 1))),
        hermiticity_error=float(hermiticity),
        minimum_eigenvalue=float(minimum_eigenvalue),
    )


def _stack_protocols(protocols):
    flattened = [jax.tree.flatten(protocol) for protocol in protocols]
    structure = flattened[0][1]
    if any(item[1] != structure for item in flattened[1:]):
        raise ValueError("protocol variants must have equal structures")
    leaves = []
    axes = []
    for items in zip(*(item[0] for item in flattened)):
        if all(item is items[0] for item in items[1:]):
            leaves.append(items[0])
            axes.append(None)
        else:
            leaves.append(jnp.stack(items))
            axes.append(0)
    return (
        jax.tree.unflatten(structure, leaves),
        jax.tree.unflatten(structure, axes),
    )


def simulate_decay_variants(
    initial_states,
    observables,
    protocols,
    cycles,
    cycle_time,
    *,
    fit_start=4,
    fit_floor=1e-10,
):
    """Simulate equal-structure protocol variants in one mapped call."""
    if len(protocols) == 1:
        return [
            simulate_decay(
                initial_states,
                observables,
                protocols[0],
                cycles,
                cycle_time,
                fit_start=fit_start,
                fit_floor=fit_floor,
            )
        ]
    protocols, axes = _stack_protocols(protocols)
    final, values, _ = jax.vmap(
        lambda rounds: jqtc.simulate_sbs(
            initial_states,
            observables,
            rounds,
            cycles,
        ),
        in_axes=(axes,),
    )(protocols)
    final, values = jax.device_get((final, values))
    return [
        _analyze_decay(
            result,
            samples,
            cycles,
            cycle_time,
            fit_start,
            fit_floor,
        )
        for result, samples in zip(final, values)
    ]


def compute_error_budget(
    protocol: Callable[[set[str]], tuple],
    initial_states,
    observables,
    cycles,
    cycle_time,
    *,
    channels=ERROR_CHANNELS,
    fit_start=4,
    fit_floor=1e-10,
    batched=True,
):
    """Return baseline-subtracted and all-on-context channel budgets."""
    channels = tuple(channels)

    def run(enabled):
        return simulate_decay(
            initial_states,
            observables,
            protocol(set(enabled)),
            cycles,
            cycle_time,
            fit_start=fit_start,
            fit_floor=fit_floor,
        )

    variants = [
        (),
        channels,
        *((channel,) for channel in channels),
        *(set(channels) - {channel} for channel in channels),
    ]
    if batched:
        results = simulate_decay_variants(
            initial_states,
            observables,
            [protocol(set(enabled)) for enabled in variants],
            cycles,
            cycle_time,
            fit_start=fit_start,
            fit_floor=fit_floor,
        )
    else:
        results = [run(enabled) for enabled in variants]
    baseline, all_on = results[:2]
    split = 2 + len(channels)
    isolated = dict(zip(channels, results[2:split]))
    without = dict(zip(channels, results[split:]))
    isolated_increments = {
        channel: isolated[channel].rate - baseline.rate for channel in channels
    }
    context_increments = {
        channel: all_on.rate - without[channel].rate for channel in channels
    }
    interaction = all_on.rate - baseline.rate - sum(isolated_increments.values())
    return ErrorBudget(
        baseline=baseline,
        all_on=all_on,
        isolated=isolated,
        without=without,
        isolated_increments=isolated_increments,
        context_increments=context_increments,
        interaction_rate=interaction,
    )


def gkp_error_budget(
    *,
    dimension=60,
    delta=0.428,
    state_delta=None,
    kind="x",
    small_ratio=1.083,
    small_displacement_scales=(1.0, 1.0),
    big_displacement=None,
    epsilon_model="sinh",
    final_storage_rotation=0.0,
    alternate_cd_direction=False,
    cycles=512,
    microsteps=1,
    device=GKP_DEVICE,
    fit_start=4,
    fit_floor=1e-10,
    max_loss=8,
    max_reset=12,
):
    initial, observables = gkp_problem(
        dimension,
        delta if state_delta is None else state_delta,
        kind,
    )
    protocol = prepare_gkp_protocol(
        dimension,
        delta=delta,
        small_ratio=small_ratio,
        small_displacement_scales=small_displacement_scales,
        big_displacement=big_displacement,
        epsilon_model=epsilon_model,
        final_storage_rotation=final_storage_rotation,
        alternate_cd_direction=alternate_cd_direction,
        device=device,
        microsteps=microsteps,
        max_loss=max_loss,
        max_reset=max_reset,
    )
    return compute_error_budget(
        protocol,
        initial,
        observables,
        cycles,
        round_time(device, 2),
        fit_start=fit_start,
        fit_floor=fit_floor,
    )


def cat_error_budget(
    nbar,
    *,
    dimension=48,
    kind="bit",
    cycles=480,
    microsteps=1,
    device=CAT_DEVICE,
    fit_start=4,
    fit_floor=1e-10,
    max_loss=8,
):
    initial, observables = cat_problem(dimension, nbar, kind)
    protocol = prepare_cat_protocol(
        dimension,
        nbar,
        device=device,
        microsteps=microsteps,
        max_loss=max_loss,
    )
    return compute_error_budget(
        protocol,
        initial,
        observables,
        cycles,
        round_time(device),
        fit_start=fit_start,
        fit_floor=fit_floor,
    )

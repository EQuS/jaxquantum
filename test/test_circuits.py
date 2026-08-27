import os
import sys

# Add the jaxquantum directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diffrax
import jax
import jax.numpy as jnp
import pytest
from jax.scipy.special import gammaln

import jaxquantum as jqt
import jaxquantum.circuits as jqtc


def test_unitary_simulation():
    N = 10
    beta = 2
    reg = jqtc.Register([2, N])
    cirq = jqtc.Circuit.create(reg, layers=[])
    cirq.append(jqtc.X(), 0)
    cirq.append(jqtc.CD(N, beta), [0, 1])
    initial_state = jqt.basis(2, 0) ^ jqt.basis(N, 0)

    # Unitary simulation
    res = jqtc.simulate(cirq, initial_state)

    a = jqt.identity(2) ^ jqt.destroy(N)
    q = (a + a.dag()) / 2

    assert (
        jnp.abs(beta / 2 + jqt.overlap(res[-1][-1], q)) < 1e-6
    ), "Overlap with q should be close to beta/2"

    # Kraus map simulation
    res = jqtc.simulate(cirq, initial_state, mode="kraus")

    a = jqt.identity(2) ^ jqt.destroy(N)
    q = (a + a.dag()) / 2

    assert (
        jnp.abs(beta / 2 + jqt.overlap(res[-1][-1], q)) < 1e-6
    ), "Overlap with q should be close to beta/2"


def test_hamiltonian_simulation():
    N = 20
    alpha = 1
    reg = jqtc.Register([2, N])
    cirq = jqtc.Circuit.create(reg, layers=[])
    cirq.append(jqtc.X(), 0)
    cirq.append(
        jqtc.D(N, alpha, ts=jnp.linspace(0, 100, 101)),
        1,
        default_simulate_mode="hamiltonian",
    )
    initial_state = jqt.basis(2, 0) ^ jqt.basis(N, 0)
    res = jqtc.simulate(cirq, initial_state)

    a = jqt.identity(2) ^ jqt.destroy(N)
    q = (a + a.dag()) / 2

    assert jnp.abs(alpha - jqt.overlap(res[-1][-1], q)) < 1e-7


def test_local_unitary_matches_promoted_layer():
    reg = jqtc.Register([2, 5, 3, 2])
    layer = jqtc.Layer.create(
        [
            jqtc.Operation.create(jqtc.CD(3, 0.24 + 0.1j), [0, 2], reg),
            jqtc.Operation.create(jqtc.Ry(0.31), 3, reg),
        ]
    )
    circuit = jqtc.Circuit.create(reg, [layer])
    state = jqt.basis(60, 7).reshape_qdims(*reg.dims)

    expected = layer.gen_U() @ state
    actual = jqtc.simulate(circuit, state)[-1][-1]

    assert actual.dims == expected.dims
    assert jnp.allclose(actual.data, expected.data, atol=1e-12)


def test_local_unitary_density_matrix_matches_promoted_layer():
    reg = jqtc.Register([2, 3, 2])
    layer = jqtc.Layer.create(
        [
            jqtc.Operation.create(jqtc.Rx(0.27), 0, reg),
            jqtc.Operation.create(jqtc.D(3, -0.16j), 1, reg),
            jqtc.Operation.create(jqtc.H(), 2, reg),
        ]
    )
    circuit = jqtc.Circuit.create(reg, [layer])
    state = (
        (
            jqt.basis(12, 0).reshape_qdims(*reg.dims)
            + 1j * jqt.basis(12, 11).reshape_qdims(*reg.dims)
        )
        .unit()
        .to_dm()
    )

    unitary = layer.gen_U()
    expected = unitary @ state @ unitary.dag()
    actual = jqtc.simulate(circuit, state)[-1][-1]

    assert jnp.allclose(actual.data, expected.data, atol=1e-12)
    assert jnp.allclose(actual.tr(), 1.0, atol=1e-12)


def test_batched_local_unitary_matches_vmap():
    angles = jnp.linspace(-0.4, 0.5, 7)
    reg = jqtc.Register([2, 2])
    batched = jqtc.Circuit.create(reg)
    batched.append(jqtc.Rx(angles), 1)
    state = jqt.basis(4, 0).reshape_qdims(*reg.dims)

    actual = jqtc.simulate(batched, state)[-1].data[-1]

    def scalar_run(angle):
        circuit = jqtc.Circuit.create(reg)
        circuit.append(jqtc.Rx(angle), 1)
        return jqtc.simulate(circuit, state)[-1][-1].data

    expected = jax.vmap(scalar_run)(angles)
    assert jnp.allclose(actual, expected, atol=1e-12)


def test_local_unitary_is_jittable_and_differentiable():
    reg = jqtc.Register([2] * 5)
    state = jqt.basis(2**5, 0).reshape_qdims(*reg.dims)

    def excited_probability(angle):
        circuit = jqtc.Circuit.create(reg)
        circuit.append(jqtc.Rx(angle), 3)
        final_state = jqtc.simulate(circuit, state)[-1][-1]
        return jnp.abs(final_state.data[2]) ** 2

    value, derivative = jax.jit(jax.value_and_grad(excited_probability))(0.37)
    expected = jnp.sin(0.37 / 2) ** 2
    expected_derivative = 0.5 * jnp.sin(0.37)
    assert jnp.allclose(value, expected, atol=1e-12)
    assert jnp.allclose(derivative, expected_derivative, atol=1e-12)


def test_local_unitary_accepts_sparse_initial_state():
    register = jqtc.Register.create([2])
    circuit = jqtc.Circuit.create(register)
    circuit.append(jqtc.X(), 0)
    state = jqt.basis(
        2,
        0,
        implementation=jqt.QarrayImplType.SPARSE_BCOO,
    )

    actual = jqtc.simulate_final(circuit, state)
    assert actual.is_dense
    assert jnp.allclose(actual.data, jqt.basis(2, 1).data)


def test_empty_circuit_preserves_sparse_result_batch():
    register = jqtc.Register.create([3])
    state = jqt.basis(
        3,
        1,
        implementation=jqt.QarrayImplType.SPARSE_BCOO,
    )

    result = jqtc.simulate(jqtc.Circuit.create(register), state)[0]
    assert result.is_sparse_bcoo
    assert result.shape == (1, 3)
    assert jnp.allclose(result.to_dense().data[0], state.to_dense().data)


def _promoted_kraus_result(layer, state):
    kraus = layer.gen_KM()
    return (kraus @ state.to_dm() @ kraus.dag()).collapse()


def test_local_oscillator_channels_match_promoted_maps():
    reg = jqtc.Register([2, 6, 2])
    real, imag = jax.random.normal(jax.random.key(7), (2, 24))
    state = (
        jqt.Qarray.create(
            real + 1j * imag,
            qtype="ket",
        )
        .reshape_qdims(*reg.dims)
        .unit()
    )

    for gate in (
        jqtc.Amp_Damp(6, 0.07, 5),
        jqtc.Amp_Gain(6, 0.07, 5),
        jqtc.Thermal_Ch(6, 0.07, 0.03, 3),
        jqtc.Dephasing_Ch(6, 0.07, 5),
    ):
        layer = jqtc.Layer.create(
            [jqtc.Operation.create(gate, 1, reg)],
            default_simulate_mode=jqtc.SimulateMode.KRAUS,
        )
        actual = jqtc.simulate(jqtc.Circuit.create(reg, [layer]), state)[-1][-1]
        expected = _promoted_kraus_result(layer, state)
        assert jnp.allclose(actual.data, expected.data, atol=1e-12)


@pytest.mark.parametrize("probability", [0.13, jnp.asarray([0.13, 0.27])])
def test_generic_local_kraus_supports_noncontiguous_targets(probability):
    reg = jqtc.Register([2, 3, 2])
    identity = jqt.identity(2) ^ jqt.identity(2)
    flip = jqt.sigmax() ^ jqt.sigmax()

    def make_layer(value):
        gate = jqtc.Gate.create(
            [2, 2],
            gen_KM=lambda _: jqt.Qarray.from_list(
                [
                    jnp.sqrt(1 - value) * identity,
                    jnp.sqrt(value) * flip,
                ]
            ),
            num_modes=2,
        )
        return jqtc.Layer.create(
            [jqtc.Operation.create(gate, [2, 0], reg)],
            default_simulate_mode=jqtc.SimulateMode.KRAUS,
        )

    layer = make_layer(probability)
    state = (
        jqt.basis(12, 0).reshape_qdims(*reg.dims)
        + 0.3 * jqt.basis(12, 11).reshape_qdims(*reg.dims)
    ).unit()

    actual = jqtc.simulate(jqtc.Circuit.create(reg, [layer]), state)[-1][-1]
    if jnp.ndim(probability):
        expected = jnp.stack(
            [_promoted_kraus_result(make_layer(value), state).data for value in probability]
        )
    else:
        expected = _promoted_kraus_result(layer, state).data
    assert jnp.allclose(actual.data, expected, atol=1e-12)


def test_direct_channels_are_jittable_and_differentiable():
    reg = jqtc.Register([2, 5])
    state = (
        jqt.basis(10, 0).reshape_qdims(*reg.dims)
        + jqt.basis(10, 4).reshape_qdims(*reg.dims)
    ).unit()

    def population(probability):
        circuit = jqtc.Circuit.create(reg)
        circuit.append(
            jqtc.Amp_Damp(5, probability, 4),
            1,
            default_simulate_mode=jqtc.SimulateMode.KRAUS,
        )
        final = jqtc.simulate(circuit, state)[-1][-1]
        return jnp.real(final.data[0, 0])

    value, derivative = jax.jit(jax.value_and_grad(population))(0.09)
    assert jnp.isfinite(value)
    assert jnp.isfinite(derivative)


def test_final_only_simulation_preserves_result():
    reg = jqtc.Register([2, 3])
    circuit = jqtc.Circuit.create(reg)
    circuit.append(jqtc.Rx(0.2), 0)
    circuit.append(jqtc.D(3, 0.1j), 1)
    state = jqt.basis(6, 0).reshape_qdims(*reg.dims)

    expected = jqtc.simulate(circuit, state)[-1][-1]
    actual = jqtc.simulate_final(circuit, state)
    compact = jqtc.simulate(circuit, state, save_states=False)

    assert len(compact) == 1
    assert jnp.allclose(actual.data, expected.data, atol=1e-12)
    assert jnp.allclose(compact[-1][-1].data, expected.data, atol=1e-12)

    with pytest.raises(ValueError, match="Unsupported simulation mode"):
        jqtc.simulate_final(circuit, state, mode="invalid")


def test_repeated_simulation_matches_unrolled_kraus_circuit():
    reg = jqtc.Register([2, 5])
    one_round = jqtc.Circuit.create(reg)
    one_round.append(jqtc.Rx(0.17), 0)
    one_round.append(
        jqtc.Amp_Damp(5, 0.04, 4),
        1,
        default_simulate_mode=jqtc.SimulateMode.KRAUS,
    )
    unrolled = jqtc.Circuit.create(reg)
    for _ in range(5):
        unrolled.append(jqtc.Rx(0.17), 0)
        unrolled.append(
            jqtc.Amp_Damp(5, 0.04, 4),
            1,
            default_simulate_mode=jqtc.SimulateMode.KRAUS,
        )
    state = jqt.basis(10, 4).reshape_qdims(*reg.dims)

    actual = jax.jit(lambda value: jqtc.simulate_repeated(one_round, value, 5))(state)
    expected = jqtc.simulate_final(unrolled, state)

    assert jnp.allclose(actual.data, expected.data, atol=1e-12)
    assert jqtc.simulate_repeated(one_round, state, 0) is state
    with pytest.raises(ValueError, match="non-negative"):
        jqtc.simulate_repeated(one_round, state, -1)


def test_repeated_expectations_match_analytic_qubit_rotation():
    angle = 0.19
    reg = jqtc.Register([2])
    circuit = jqtc.Circuit.create(reg)
    circuit.append(jqtc.Rx(angle), 0)
    state = jqt.basis(2, 0)

    final, values = jax.jit(
        lambda initial: jqtc.simulate_repeated_expectations(
            circuit,
            initial,
            7,
            [jqt.sigmaz()],
        )
    )(state)

    assert values.shape == (8, 1)
    assert jnp.allclose(values[:, 0], jnp.cos(angle * jnp.arange(8)), atol=1e-12)
    assert jnp.allclose(
        jqt.overlap(final, jqt.sigmaz()),
        jnp.cos(7 * angle),
        atol=1e-12,
    )


def test_empty_circuit_expectations_without_initial_state():
    circuit = jqtc.Circuit.create(jqtc.Register([2]))
    state = jqt.basis(2, 0)

    final, values = jqtc.simulate_expectations(
        circuit,
        state,
        [jqt.sigmaz()],
        include_initial=False,
    )

    assert final is state
    assert values.shape == (0, 1)


def test_public_direct_channel_helpers_match_kraus_fallbacks():
    reg = jqtc.Register([2, 3])
    state = (
        jqt.basis(6, 0).reshape_qdims(*reg.dims)
        + (0.2 + 0.3j) * jqt.basis(6, 5).reshape_qdims(*reg.dims)
    ).unit()
    coefficients = jnp.array(
        [
            [1.0, 0.8 + 0.1j, 0.5],
            [0.3j, 0.2, 0.0],
        ]
    )
    shifts = jnp.array([0, 1])
    indices = jnp.arange(3)
    kraus = []
    for coefficient, shift in zip(coefficients, shifts):
        source = jnp.clip(indices + shift, 0, 2)
        valid = indices + shift < 3
        data = jnp.zeros((3, 3), dtype=coefficients.dtype)
        data = data.at[indices, source].set(jnp.where(valid, coefficient, 0))
        kraus.append(jqt.Qarray.create(data))

    channel = jqtc.ShiftedChannel(
        3,
        coefficients,
        shifts,
        kraus=kraus,
    )
    layer = jqtc.Layer.create(
        [jqtc.Operation.create(channel, 1, reg)],
        default_simulate_mode=jqtc.SimulateMode.KRAUS,
    )
    actual = jqtc.simulate_final(jqtc.Circuit.create(reg, [layer]), state)
    expected = _promoted_kraus_result(layer, state)

    assert jnp.allclose(actual.data, expected.data, atol=1e-12)
    with pytest.raises(ValueError, match="coefficients"):
        jqtc.ShiftedChannel(3, jnp.ones((2, 2)), shifts)
    with pytest.raises(ValueError, match="non-empty"):
        jqtc.ShiftedChannel(3, jnp.ones((0, 3)), ())
    with pytest.raises(ValueError, match="one-dimensional"):
        jqtc.ShiftedChannel(3, jnp.ones((2, 3)), jnp.ones((2, 1)))


def test_apply_channel_uses_direct_and_kraus_paths():
    rho = jqt.basis(4, 3).to_dm().data
    direct = jqtc.Amp_Damp(4, 0.2, 3)
    assert jnp.allclose(
        jqtc.apply_channel(direct, rho),
        jqtc.apply_kraus_map(direct.KM, rho),
    )

    kraus_only = jqtc.Gate.create(
        2,
        gen_KM=lambda params: jqt.Qarray.from_list([jqt.identity(2)]),
        lazy_kraus=True,
    )
    qubit_rho = jqt.basis(2, 1).to_dm().data
    assert jnp.allclose(jqtc.apply_channel(kraus_only, qubit_rho), qubit_rho)

    custom = jqtc.Channel(
        2,
        lambda value, params: value,
        kraus=lambda params: [jqt.identity(2)],
    )
    assert custom.channel_apply is not None
    assert isinstance(custom.KM, jqt.Qarray)
    assert jnp.allclose(jqtc.apply_kraus_map(custom.KM, qubit_rho), qubit_rho)


def test_apply_channel_targets_nontrailing_density_axes():
    rho = jnp.arange(72, dtype=float).reshape(2, 2, 3, 2, 3)
    channel = jqtc.Dephasing_Ch_Qb(0.2)
    kraus = channel.KM.data
    expected = jnp.einsum("kai,bimjn,kcj->bamcn", kraus, rho, kraus.conj())

    assert jnp.allclose(
        jqtc.apply_channel(channel, rho, axes=(1, 3)),
        expected,
    )

    fallback = jqtc.Gate.create(
        2,
        gen_KM=lambda params: channel.KM,
        lazy_kraus=True,
    )
    assert jnp.allclose(
        jqtc.apply_channel(fallback, rho, axes=(1, 3)),
        expected,
    )

    batched = jqtc.Dephasing_Ch_Qb(jnp.asarray([0.1, 0.2]))
    qubit_rho = jqt.basis(2, 0).to_dm().data
    expected = jnp.stack(
        [jqtc.apply_channel(jqtc.Dephasing_Ch_Qb(p), qubit_rho) for p in (0.1, 0.2)]
    )
    assert jnp.allclose(
        jqtc.apply_channel(batched, qubit_rho, axes=(0, 1)),
        expected,
    )
    thermal = jqtc.Thermal_Ch_Qb(jnp.asarray([0.1, 0.2]), 0.05)
    expected = jnp.stack(
        [jqtc.apply_channel(jqtc.Thermal_Ch_Qb(p, 0.05), qubit_rho) for p in (0.1, 0.2)]
    )
    assert jnp.allclose(
        jqtc.apply_channel(thermal, qubit_rho, axes=(0, 1)),
        expected,
    )


def test_apply_channel_supports_batched_parameters():
    rho = jqt.basis(4, 3).to_dm().data
    probabilities = jnp.asarray([0.1, 0.2])
    channels = (
        jqtc.Amp_Damp(4, probabilities, 3),
        jqtc.Amp_Gain(4, probabilities, 3),
        jqtc.Thermal_Ch(4, probabilities, jnp.asarray([0.01, 0.02]), 2),
        jqtc.Dephasing_Ch(4, probabilities, 4),
    )
    for channel in channels:
        direct = jax.jit(jqtc.apply_channel)(channel, rho)
        fallback = jqtc.apply_kraus_map(channel.KM, rho)
        assert direct.shape == fallback.shape == (2, 4, 4)
        assert jnp.allclose(direct, fallback, atol=1e-12)


def test_thermal_qubit_kraus_supports_batched_parameters():
    channel = jqtc.Thermal_Ch_Qb(jnp.asarray([0.1, 0.2]), 0.05)
    rho = jqt.basis(2, 1).to_dm().data

    fallback = jqtc.apply_kraus_map(channel.KM, rho)
    direct = jqtc.apply_channel(channel, rho)
    assert channel.KM.data.shape == (4, 2, 2, 2)
    assert jnp.allclose(fallback, direct, atol=1e-12)


def test_direct_dephasing_reset_matches_promoted_map():
    reg = jqtc.Register([3, 2, 5, 2])
    real, imag = jax.random.normal(jax.random.key(11), (2, 60))
    state = (
        jqt.Qarray.create(
            real + 1j * imag,
            qtype="ket",
        )
        .reshape_qdims(*reg.dims)
        .unit()
    )
    gate = jqtc.Dephasing_Reset(5, 0.13, 0.4, 0.7, 5)
    layer = jqtc.Layer.create(
        [jqtc.Operation.create(gate, [1, 2], reg)],
        default_simulate_mode=jqtc.SimulateMode.KRAUS,
    )

    actual = jqtc.simulate_final(jqtc.Circuit.create(reg, [layer]), state)
    expected = _promoted_kraus_result(layer, state)
    assert jnp.allclose(actual.data, expected.data, atol=1e-12)


def test_dephasing_reset_supports_batched_probabilities_and_endpoints():
    rho = (jqt.basis(6, 0) + 0.2j * jqt.basis(6, 5)).to_dm().unit().data
    gate = jqtc.Dephasing_Reset(
        3,
        jnp.asarray([0.0, 0.2, 1.0]),
        0.4,
        0.7,
        4,
    )

    direct = jax.jit(jqtc.apply_channel)(gate, rho)
    fallback = jqtc.apply_kraus_map(gate.KM, rho)

    assert direct.shape == fallback.shape == (3, 6, 6)
    assert jnp.all(jnp.isfinite(direct))
    assert jnp.allclose(direct, fallback, atol=1e-12)


def test_conditional_displacements_reuse_exact_inverse():
    dimension = 5
    beta = jnp.asarray(0.17 + 0.09j)
    g, e = jqt.basis(2, 0), jqt.basis(2, 1)
    displacement = jqt.displace(dimension, beta / 2)
    inverse = jqt.displace(dimension, -beta / 2)
    expected_cd = (g @ g.dag()) ^ displacement
    expected_cd += (e @ e.dag()) ^ inverse
    expected_ecd = (e @ g.dag()) ^ displacement
    expected_ecd += (g @ e.dag()) ^ inverse

    assert jnp.allclose(jqtc.CD(dimension, beta).U.data, expected_cd.data)
    assert jnp.allclose(jqtc.ECD(dimension, beta).U.data, expected_ecd.data)


def test_direct_channel_kraus_maps_are_lazy_and_preserved_by_copy():
    gate = jqtc.Amp_Damp(6, 0.12, 3)
    copied = gate.copy()

    assert gate._KM is None
    assert copied._KM is None
    assert jnp.allclose(gate.KM.data, copied.KM.data)


def test_thermal_kraus_constructor_matches_matrix_power_reference():
    dimension, max_l = 5, 2
    probability, n_bar = 0.08, 0.15
    a = jqt.destroy(dimension).data
    adag = jnp.conj(a.T)
    middle = jnp.diag(jnp.power(jnp.sqrt(1 - probability), jnp.arange(dimension)))
    expected = []
    for gain in range(max_l + 1):
        for loss in range(max_l + 1):
            prefactor = jnp.sqrt(
                (probability * (1 + n_bar)) ** loss
                * (probability * n_bar) ** gain
                / (jnp.exp(gammaln(loss + 1)) * jnp.exp(gammaln(gain + 1)))
            )
            expected.append(
                prefactor
                * middle
                @ jnp.linalg.matrix_power(a, loss)
                @ jnp.linalg.matrix_power(adag, gain)
            )

    actual = jqtc.Thermal_Ch(
        dimension,
        probability,
        n_bar,
        max_l,
    ).KM.data
    assert jnp.allclose(actual, jnp.stack(expected), atol=1e-12)


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_dephasing_reset_probability_endpoints(probability):
    dimension = 4
    reg = jqtc.Register([2, dimension])
    state = (
        (
            jqt.basis(2 * dimension, 0)
            + 0.3j * jqt.basis(2 * dimension, 2 * dimension - 1)
        )
        .reshape_qdims(*reg.dims)
        .unit()
        .to_dm()
    )
    gate = jqtc.Dephasing_Reset(dimension, probability, 0.4, 0.7, 4)
    layer = jqtc.Layer.create(
        [jqtc.Operation.create(gate, [0, 1], reg)],
        default_simulate_mode=jqtc.SimulateMode.KRAUS,
    )

    direct = jqtc.simulate_final(jqtc.Circuit.create(reg, [layer]), state)
    promoted = _promoted_kraus_result(layer, state)
    assert jnp.all(jnp.isfinite(direct.data))
    assert jnp.allclose(direct.data, promoted.data, atol=1e-12)
    assert jnp.allclose(direct.trace(), 1.0, atol=1e-12)


def test_channel_order_validation_and_large_truncation():
    with pytest.raises(ValueError, match="non-negative"):
        jqtc.Amp_Damp(4, 0.1, -1)
    with pytest.raises(ValueError, match="positive"):
        jqtc.Dephasing_Ch(4, 0.1, 0)
    with pytest.raises(ValueError, match="at least two"):
        jqtc.Dephasing_Reset(4, 0.1, 0.2, 0.3, 1)

    gate = jqtc.Amp_Gain(4, 0.1, 10)
    assert gate.KM.data.shape == (11, 4, 4)
    assert jnp.allclose(gate.KM.data[4:], 0)


@pytest.mark.parametrize(
    "gate",
    [
        jqtc.Reset(),
        jqtc.IP_Reset(0.07, 0.86),
        jqtc.Thermal_Ch_Qb(0.13, 0.04),
        jqtc.Dephasing_Ch_Qb(0.09),
        jqtc.MZ(),
        jqtc.MZ(1),
        jqtc.MZ(-1),
        jqtc.MX(),
        jqtc.MX(1),
        jqtc.MX(-1),
    ],
)
def test_direct_qubit_channels_match_promoted_maps(gate):
    reg = jqtc.Register([3, 2, 5])
    real, imag = jax.random.normal(jax.random.key(13), (2, 30))
    state = jqt.Qarray.create(real + 1j * imag).reshape_qdims(*reg.dims).unit()
    layer = jqtc.Layer.create(
        [jqtc.Operation.create(gate, 1, reg)],
        default_simulate_mode=jqtc.SimulateMode.KRAUS,
    )

    actual = jqtc.simulate_final(jqtc.Circuit.create(reg, [layer]), state)
    expected = _promoted_kraus_result(layer, state)
    assert jnp.allclose(actual.data, expected.data, atol=1e-12)


def test_direct_qubit_channels_are_jittable_and_differentiable():
    reg = jqtc.Register([2, 3])
    state = (jqt.basis(6, 0) + 0.3j * jqt.basis(6, 5)).reshape_qdims(*reg.dims).unit()

    def population(probability):
        circuit = jqtc.Circuit.create(reg)
        circuit.append(
            jqtc.Thermal_Ch_Qb(probability, 0.05),
            0,
            default_simulate_mode=jqtc.SimulateMode.KRAUS,
        )
        final = jqtc.simulate_final(circuit, state)
        return jnp.real(final.data[..., 0, 0])

    assert jnp.isfinite(jax.jit(jax.grad(population))(0.1))

    def evolve(probability):
        circuit = jqtc.Circuit.create(reg)
        circuit.append(
            jqtc.Thermal_Ch_Qb(probability, 0.05),
            0,
            default_simulate_mode=jqtc.SimulateMode.KRAUS,
        )
        return jqtc.simulate_final(circuit, state).data

    probabilities = jnp.array([0.05, 0.1, 0.2])
    batched = evolve(probabilities)
    vmapped = jax.vmap(evolve)(probabilities)
    assert jnp.allclose(batched, vmapped, atol=1e-12)


def test_local_hamiltonian_and_lindblad_match_promoted_solvers():
    reg = jqtc.Register([2, 4, 2])
    times = jnp.linspace(0.0, 0.3, 7)
    options = jqt.SolverOptions(
        stepsize_controller=diffrax.ConstantStepSize(),
        dt0=times[1] - times[0],
        progress_meter=None,
    )
    state = (
        jqt.basis(16, 0).reshape_qdims(*reg.dims)
        + 0.2j * jqt.basis(16, 15).reshape_qdims(*reg.dims)
    ).unit()

    for collapse in (False, True):
        c_ops = jqt.Qarray.from_list([0.04 * jqt.destroy(4)]) if collapse else None
        layer = jqtc.Layer.create(
            [
                jqtc.Operation.create(jqtc.Rx(0.12, ts=times), 0, reg),
                jqtc.Operation.create(
                    jqtc.D(4, 0.08j, ts=times, c_ops=c_ops),
                    1,
                    reg,
                ),
            ],
            default_simulate_mode=jqtc.SimulateMode.HAMILTONIAN,
        )
        initial = state.to_dm() if collapse else state
        actual = jqtc.simulate_final(
            jqtc.Circuit.create(reg, [layer]),
            initial,
            solver_options=options,
            local_operators=True,
        )
        promoted = jqtc.simulate_final(
            jqtc.Circuit.create(reg, [layer]),
            initial,
            solver_options=options,
            local_operators=False,
        )
        if collapse:
            expected = jqt.mesolve(
                layer.gen_Ht(),
                initial,
                times,
                saveat_tlist=jnp.array([]),
                c_ops=layer.gen_c_ops(),
                solver_options=options,
            )[-1]
        else:
            expected = jqt.sesolve(
                layer.gen_Ht(),
                initial,
                times,
                saveat_tlist=jnp.array([]),
                solver_options=options,
            )[-1]
        assert jnp.allclose(actual.data, expected.data, atol=1e-10)
        assert jnp.allclose(promoted.data, expected.data, atol=1e-10)

import jax
import jax.numpy as jnp
import jaxquantum as jqt

from jaxquantum.devices import (
    ATS,
    Fluxonium,
    IdealQubit,
    KNO,
    Resonator,
    SNAIL,
    Transmon,
)
from jaxquantum.devices.base.base import (
    get_vec_data_in_new_basis,
    get_vec_in_new_basis,
)


def make_transmon(ng=0.13):
    return Transmon.create(
        N=6,
        N_pre_diag=21,
        params={"Ec": 0.22, "Ej": 18.0, "ng": ng},
    )


def test_device_hamiltonian_uses_truncated_eigenvalues():
    device = make_transmon()
    values, _ = jnp.linalg.eigh(device._get_H_in_original_basis().data)
    hamiltonian = device.get_H()

    assert hamiltonian.dims == ((device.N,), (device.N,))
    assert hamiltonian.dtype == device._get_H_in_original_basis().dtype
    assert jnp.allclose(jnp.diag(hamiltonian.data), values[: device.N], atol=1e-12)
    assert jnp.allclose(
        hamiltonian.data - jnp.diag(jnp.diag(hamiltonian.data)),
        0.0,
        atol=1e-12,
    )


def test_full_ops_diagonalizes_once(monkeypatch):
    calls = 0
    original = Transmon._calculate_eig_systems

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(Transmon, "_calculate_eig_systems", counted)
    operators = make_transmon().ops

    assert calls == 1
    assert {"id", "n", "cos(\u03c6)", "sin(\u03c6)"}.issubset(operators)


def test_transmon_hamiltonian_builds_operators_once(monkeypatch):
    calls = 0
    original = Transmon.common_ops

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(Transmon, "common_ops", counted)
    make_transmon().get_H_full()
    assert calls == 1


def test_device_hamiltonians_build_operator_dictionary_once(monkeypatch):
    devices = [
        Resonator.create(4, {"Ec": 0.2, "El": 1.0}),
        Fluxonium.create(4, {"Ec": 0.2, "El": 1.0, "Ej": 3.0, "phi_ext": 0.1}),
        ATS.create(
            4,
            {
                "Ec": 0.2,
                "El": 1.0,
                "Ej": 3.0,
                "dEj": 0.1,
                "Ej2": 0.05,
                "phi_sum_ext": 0.1,
                "phi_delta_ext": 0.03,
            },
        ),
        KNO.create(4, {"f": 5.0, "α": -0.2}),
        IdealQubit.create(2, {"f": 5.0, "Δ": 0.1}),
        SNAIL.create(
            3,
            {"Ec": 0.2, "Ej": 3.0, "alpha": 0.25, "m": 2, "phi_ext": 0.1},
            N_pre_diag=5,
        ),
    ]

    for device in devices:
        cls = type(device)
        original = cls.common_ops
        calls = []

        def counted(self, fn=original):
            calls.append(None)
            return fn(self)

        monkeypatch.setattr(cls, "common_ops", counted)
        device.get_H_full()
        assert len(calls) == 1


def test_full_ops_match_direct_basis_transform():
    device = make_transmon()
    original_ops = device.linear_ops
    vectors = device.eig_systems["vecs"][:, : device.N]
    transformed_ops = device.ops

    for name, operator in original_ops.items():
        expected = vectors.conj().T @ operator.data @ vectors
        assert jnp.allclose(transformed_ops[name].data, expected, atol=1e-12)


def test_basis_transform_supports_vectors_and_column_stacks():
    vectors = jnp.stack((jnp.eye(4), jnp.flip(jnp.eye(4))))
    states = jqt.Qarray.from_list([jqt.basis(4, 0), jqt.basis(4, 1)])
    transformed = get_vec_in_new_basis(states, vectors, ((4,), (1,)))
    columns = jnp.arange(28.0).reshape(4, 7)

    assert transformed.shape == (2, 4)
    assert jnp.allclose(
        get_vec_data_in_new_basis(columns, vectors),
        jnp.swapaxes(vectors.conj(), -1, -2) @ columns,
    )


def test_device_hamiltonian_is_jittable_and_batchable():
    def hamiltonian(ng):
        return make_transmon(ng).get_H().data

    offsets = jnp.linspace(-0.2, 0.2, 5)
    actual = jax.jit(jax.vmap(hamiltonian))(offsets)
    expected = jnp.stack([hamiltonian(offset) for offset in offsets])

    assert actual.shape == (5, 6, 6)
    assert jnp.allclose(actual, expected, atol=1e-12)


def test_transmon_wavefunctions_match_loop_and_diagonalize_once(monkeypatch):
    device = make_transmon()
    phases = jnp.linspace(-0.4, 0.4, 9)
    n_labels = jnp.diag(device.original_ops["n"].data)
    vectors = device.eig_systems["vecs"]
    expected = jnp.stack(
        [
            jnp.stack(
                [
                    (1j**level / jnp.sqrt(2 * jnp.pi))
                    * jnp.sum(vectors[:, level] * jnp.exp(1j * phi * n_labels))
                    for phi in phases
                ]
            )
            for level in range(device.N_pre_diag)
        ]
    )
    calls = 0
    original = Transmon._calculate_eig_systems

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(Transmon, "_calculate_eig_systems", counted)
    actual = device.calculate_wavefunctions(phases)
    assert calls == 1
    assert jnp.allclose(actual, expected, atol=1e-12)

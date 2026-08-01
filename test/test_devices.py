import jax
import jax.numpy as jnp

from jaxquantum.devices import Transmon


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


def test_full_ops_match_direct_basis_transform():
    device = make_transmon()
    original_ops = device.linear_ops
    vectors = device.eig_systems["vecs"][:, : device.N]
    transformed_ops = device.ops

    for name, operator in original_ops.items():
        expected = vectors.conj().T @ operator.data @ vectors
        assert jnp.allclose(transformed_ops[name].data, expected, atol=1e-12)


def test_device_hamiltonian_is_jittable_and_batchable():
    def hamiltonian(ng):
        return make_transmon(ng).get_H().data

    offsets = jnp.linspace(-0.2, 0.2, 5)
    actual = jax.jit(jax.vmap(hamiltonian))(offsets)
    expected = jnp.stack([hamiltonian(offset) for offset in offsets])

    assert actual.shape == (5, 6, 6)
    assert jnp.allclose(actual, expected, atol=1e-12)

import sys
import os

# Add the jaxquantum directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jaxquantum as jqt
import jaxquantum.circuits as jqtc
import jax
import jax.numpy as jnp



def test_unitary_simulation():
    N = 10
    beta = 2
    reg = jqtc.Register([2,N])
    cirq = jqtc.Circuit.create(reg, layers=[])
    cirq.append(jqtc.X(),0)
    cirq.append(jqtc.CD(N, beta),[0,1])
    initial_state = jqt.basis(2,0) ^ jqt.basis(N,0)
    
    
    # Unitary simulation
    res = jqtc.simulate(cirq, initial_state)

    a = jqt.identity(2) ^ jqt.destroy(N)
    q = (a + a.dag())/2

    assert jnp.abs(beta/2 + jqt.overlap(res[-1][-1], q)) < 1e-6, "Overlap with q should be close to beta/2"


    # Kraus map simulation
    res = jqtc.simulate(cirq, initial_state, mode="kraus")

    a = jqt.identity(2) ^ jqt.destroy(N)
    q = (a + a.dag())/2

    assert jnp.abs(beta/2 + jqt.overlap(res[-1][-1], q)) < 1e-6, "Overlap with q should be close to beta/2"
    

def test_hamiltonian_simulation():
    N = 20
    alpha = 1
    reg = jqtc.Register([2,N])
    cirq = jqtc.Circuit.create(reg, layers=[])
    cirq.append(jqtc.X(),0)
    cirq.append(jqtc.D(N, alpha, ts=jnp.linspace(0,100,101)),1, default_simulate_mode="hamiltonian")
    initial_state = jqt.basis(2,0) ^ jqt.basis(N,0)
    res = jqtc.simulate(cirq, initial_state)

    a = jqt.identity(2) ^ jqt.destroy(N)
    q = (a + a.dag())/2

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
        jqt.basis(12, 0).reshape_qdims(*reg.dims)
        + 1j * jqt.basis(12, 11).reshape_qdims(*reg.dims)
    ).unit().to_dm()

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

# Test adopted from test_enr_state_operator.py in QuTip

import pytest
import itertools
import random
import numpy as np
import jaxquantum as jqt
import jax.numpy as jnp


def _n_enr_states(dimensions, n_excitations):
    """
    Calculate the total number of distinct ENR states for a given set of
    subspaces.  This method is not intended to be fast or efficient, it's
    intended to be obviously correct for testing purposes.
    """
    count = 0
    for excitations in itertools.product(*map(range, dimensions)):
        count += int(sum(excitations) <= n_excitations)
    return count


@pytest.fixture(params=[
    pytest.param([4], id="single"),
    pytest.param([4]*2, id="tensor-equal-2"),
    pytest.param([4]*3, id="tensor-equal-3"),
    pytest.param([4]*4, id="tensor-equal-4"),
    pytest.param([2, 3, 4], id="tensor-not-equal"),
])
def dimensions(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 1_000_000])
def n_excitations(request):
    return request.param


class TestOperator:
    def test_no_restrictions(self, dimensions):
        """
        Test that the restricted-excitation operators are equal to the standard
        operators when there aren't any restrictions.
        """
        test_operators = jqt.enr_destroy(dimensions, sum(dimensions))
        a = [jqt.destroy(n) for n in dimensions]
        iden = [jqt.identity(n) for n in dimensions]
        for i, test in enumerate(test_operators):
            expected = jqt.tensor(*(iden[:i] + [a[i]] + iden[i+1:]))
            assert (test.data == expected.data).all
            assert test.dims == [np.prod(dimensions), np.prod(dimensions)]

    def test_space_size_reduction(self, dimensions, n_excitations):
        test_operators = jqt.enr_destroy(dimensions, n_excitations)
        expected_size = _n_enr_states(dimensions, n_excitations)
        expected_shape = [[expected_size], [expected_size]]
        for test in test_operators:
            assert test.dims == expected_shape

    def test_identity(self, dimensions, n_excitations):
        iden = jqt.enr_identity(dimensions, n_excitations)
        expected_size = _n_enr_states(dimensions, n_excitations)
        expected_shape = [[expected_size], [expected_size]]
        assert (jnp.diag(iden.data) == 1).all()
        assert (iden.data - jnp.diag(jnp.diag(iden.data)) == 0).all()
        assert iden.dims == expected_shape


def test_fock_state(dimensions, n_excitations):
    """
    Test Fock state creation agrees with the number operators implied by the
    existence of the ENR annihiliation operators.
    """
    number = [a.dag()*a for a in jqt.enr_destroy(dimensions, n_excitations)]
    nstates, state2idx, idx2state = jqt.enr_state_dictionaries(dimensions, n_excitations)
    bases = list(state2idx.keys())
    n_samples = min((len(bases), 5))
    for basis in random.sample(bases, n_samples):
        state = jqt.enr_fock(dimensions, n_excitations, basis)
        for n, x in zip(number, basis):
            print(n)
            assert jnp.abs((state.dag() * n * state).data - x) < 1e-10


def test_fock_state_error():
    with pytest.raises(ValueError) as e:
        state = jqt.enr_fock([2, 2, 2], 1, [1, 1, 1])
    assert str(e.value).startswith("Given state")


def test_mesolve_ENR():
    # Ensure ENR states work with mesolve
    # We compare the output to an exact truncation of the
    # single-excitation Jaynes-Cummings model
    eps = 2 * np.pi
    omega_c = 2 * np.pi
    g = 0.1 * omega_c
    gam = 0.01 * omega_c
    tlist = np.linspace(0, 20, 100)
    N_cut = 2

    sz = jqt.sigmaz() ^ jqt.identity(N_cut)
    sm = jqt.destroy(2).dag() ^ jqt.identity(N_cut)
    a = jqt.identity(2) ^ jqt.destroy(N_cut)
    H_JC = (0.5 * eps * sz + omega_c * a.dag()*a +
            g * (a * sm.dag() + a.dag() * sm))
    psi0 = jqt.basis(2, 0) ^ jqt.basis(N_cut, 0)
    c_ops = [np.sqrt(gam) * a]

    result_psi = jqt.mesolve(psi0*psi0.dag(),
                            tlist,
                            c_ops,
                            H_JC)
    result_JC = jqt.calc_expect(sz, result_psi)

    N_exc = 1
    dims = [2, N_cut]
    d = jqt.enr_destroy(dims, N_exc)
    sz = 2*d[0].dag()*d[0]-1
    b = d[0]
    a = d[1]
    psi0 = jqt.enr_fock(dims, N_exc, [1, 0])
    H_enr = (eps * b.dag()*b + omega_c * a.dag() * a +
             g * (b.dag() * a + a.dag() * b))
    c_ops = [np.sqrt(gam) * a]

    result_enr_psi = jqt.mesolve(psi0*psi0.dag(),
                                 tlist,
                                 c_ops,
                                 H_enr)
    result_enr = jqt.calc_expect(sz, result_enr_psi)

    assert jnp.allclose(result_JC, result_enr, atol=1e-5)


# def test_steadystate_ENR():
#     # Ensure ENR states work with steadystate functions
#     # We compare the output to an exact truncation of the
#     # single-excitation Jaynes-Cummings model
#     eps = 2 * np.pi
#     omega_c = 2 * np.pi
#     g = 0.1 * omega_c
#     gam = 0.01 * omega_c
#     N_cut = 2

#     sz = qutip.sigmaz() & qutip.qeye(N_cut)
#     sm = qutip.destroy(2).dag() & qutip.qeye(N_cut)
#     a = qutip.qeye(2) & qutip.destroy(N_cut)
#     H_JC = (0.5 * eps * sz + omega_c * a.dag()*a +
#             g * (a * sm.dag() + a.dag() * sm))
#     c_ops = [np.sqrt(gam) * a]

#     result_JC = qutip.steadystate(H_JC, c_ops)
#     exp_sz_JC = qutip.expect(sz, result_JC)

#     N_exc = 1
#     dims = [2, N_cut]
#     d = qutip.enr_destroy(dims, N_exc)
#     sz = 2*d[0].dag()*d[0]-1
#     b = d[0]
#     a = d[1]
#     H_enr = (eps * b.dag()*b + omega_c * a.dag() * a +
#              g * (b.dag() * a + a.dag() * b))
#     c_ops = [np.sqrt(gam) * a]

#     result_enr = qutip.steadystate(H_enr, c_ops)
#     exp_sz_enr = qutip.expect(sz, result_enr)

#     np.testing.assert_allclose(exp_sz_JC,
#                                exp_sz_enr, atol=1e-2)

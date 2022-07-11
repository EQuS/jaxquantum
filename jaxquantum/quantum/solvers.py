"""Solvers"""

from functools import partial
from typing import Callable, List, Optional

from jax import jit, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from jaxquantum.quantum.base import dag


def spre(op: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Superoperator generator.

    Args:
        op: operator to be turned into a superoperator

    Returns:
        superoperator function
    """
    op_dag = jnp.conj(op).T
    return lambda rho: 0.5 * (
        2 * op @ rho @ op_dag - rho @ op_dag @ op - op_dag @ op @ rho
    )


@partial(
    jit,
    static_argnums=(
        4,
        5,
    ),
)
def mesolve(
    p: jnp.ndarray,
    t_list: jnp.ndarray,
    c_ops: Optional[List[jnp.ndarray]] = None,
    H0: Optional[jnp.ndarray] = None,
    Ht: Optional[Callable[[float], jnp.ndarray]] = None,
    use_density_matrix=False,
):
    """Quantum Master Equation solver.

    Args:
        p: initial state
        t_list: time list
        c_ops: list of collapse operators
        H0: time independent Hamiltonian. If H0 is not None, it will override Ht.
        Ht: time dependent Hamiltonian.
        use_density_matrix: if True, use density matrix instead of state vector

    Returns:
        list of states
    """

    # These checks slow down the function substantially.. so removing them for now.
    # use_density_matrix = use_density_matrix or c_ops is not None
    # c_ops = [] if c_ops is None else c_ops
    # if use_density_matrix and is_1d(p):
    #     p = ket2dm(p)

    def f(
        rho: jnp.ndarray,
        t: float,
        H0_val: Optional[jnp.ndarray],
        c_ops_val: List[jnp.ndarray],
    ):
        if H0 is not None:
            H = H0_val  # use H0 if given
        else:
            H = Ht(t)  # type: ignore

        rho_dot = -1j * (H @ rho)

        if use_density_matrix:
            rho_dot += -1j * (-rho @ H)

        for op in c_ops_val:
            rho_dot += spre(op)(rho)
        return rho_dot

    return odeint(f, p, t_list, H0, c_ops)


def calc_expect(op: jnp.ndarray, states: jnp.ndarray) -> jnp.ndarray:
    """Calculate expectation value of an operator given a list of states.

    Args:
        op: operator
        states: list of states

    Returns:
        list of expectation values
    """

    @jit
    def calc_expect_single(state: jnp.ndarray):
        return (dag(state) @ op @ state)[0][0]

    return vmap(calc_expect_single)(states)

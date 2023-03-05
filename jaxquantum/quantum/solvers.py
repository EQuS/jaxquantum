"""Solvers"""

from functools import partial
from typing import Callable, List, Optional

from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
from jax import jit, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from jaxquantum.quantum.base import dag
from jaxquantum.utils.utils import (
    is_1d,
    real_to_complex_iso_matrix,
    real_to_complex_iso_vector,
    complex_to_real_iso_matrix,
    complex_to_real_iso_vector,
    imag_times_iso_vector,
    imag_times_iso_matrix,
    conj_transpose_iso_matrix,
)


def spre(op: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Superoperator generator.

    Args:
        op: operator to be turned into a superoperator

    Returns:
        superoperator function
    """
    op_dag = conj_transpose_iso_matrix(op)
    return lambda rho: 0.5 * (
        2 * op @ rho @ op_dag - rho @ op_dag @ op - op_dag @ op @ rho
    )


@partial(
    jit,
    static_argnums=(4,),
)
def mesolve(
    p: jnp.ndarray,
    t_list: jnp.ndarray,
    c_ops: Optional[List[jnp.ndarray]] = jnp.array([]),
    H0: Optional[jnp.ndarray] = None,
    Ht: Optional[Callable[[float], jnp.ndarray]] = None,
):
    """Quantum Master Equation solver.

    Args:
        p: initial state, must be a density matrix. For statevector evolution, please use sesolve.
        t_list: time list
        c_ops: list of collapse operators
        H0: time independent Hamiltonian. If H0 is not None, it will override Ht.
        Ht: time dependent Hamiltonian function.

    Returns:
        list of states
    """

    p = complex_to_real_iso_matrix(p + 0.0j)
    c_ops = vmap(complex_to_real_iso_matrix)(c_ops + 0.0j)
    H0 = None if H0 is None else complex_to_real_iso_matrix(H0 + 0.0j)

    def f(
        t: float,
        rho: jnp.ndarray,
        args: jnp.ndarray,
    ):
        H0_val = args[0]
        c_ops_val = args[1]

        if H0_val is not None:
            H = H0_val  # use H0 if given
        else:
            H = Ht(t)  # type: ignore
            H = complex_to_real_iso_matrix(H + 0.0j)

        rho_dot = -1 * imag_times_iso_matrix(H @ rho - rho @ H)

        for op in c_ops_val:
            rho_dot += spre(op)(rho)

        return rho_dot

    term = ODETerm(f)
    solver = Dopri5()
    saveat = SaveAt(ts=t_list)
    stepsize_controller = PIDController(rtol=1e-7, atol=1e-7)

    sol = diffeqsolve(
        term,
        solver,
        t0=t_list[0],
        t1=t_list[-1],
        dt0=t_list[1] - t_list[0],
        y0=p,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        args=[H0, c_ops],
        max_steps=16**5,
    )

    return vmap(real_to_complex_iso_matrix)(sol.ys)


@partial(
    jit,
    static_argnums=(3,),
)
def sesolve(
    ψ: jnp.ndarray,
    t_list: jnp.ndarray,
    H0: Optional[jnp.ndarray] = None,
    Ht: Optional[Callable[[float], jnp.ndarray]] = None,
):
    """Schroedinger Equation solver.

    Args:
        ψ: initial statevector
        t_list: time list
        H0: time independent Hamiltonian. If H0 is not None, it will override Ht.
        Ht: time dependent Hamiltonian function.

    Returns:
        list of states
    """
    ψ = complex_to_real_iso_vector(ψ + 0.0j)
    H0 = None if H0 is None else complex_to_real_iso_matrix(H0 + 0.0j)

    def f(
        t: float,
        ψₜ: jnp.ndarray,
        args: jnp.ndarray,
    ):
        H0_val = args[0]

        if H0_val is not None:
            H = H0_val  # use H0 if given
        else:
            H = Ht(t)  # type: ignore
            H = complex_to_real_iso_matrix(H)
        # print("H", H.shape)
        # print("psit", ψₜ.shape)
        ψₜ_dot = -1 * imag_times_iso_vector(H @ ψₜ)

        return ψₜ_dot

    term = ODETerm(f)
    solver = Dopri5()
    saveat = SaveAt(ts=t_list)
    stepsize_controller = PIDController(rtol=1e-7, atol=1e-7)

    sol = diffeqsolve(
        term,
        solver,
        t0=t_list[0],
        t1=t_list[-1],
        dt0=t_list[1] - t_list[0],
        y0=ψ,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        args=[H0],
    )

    return vmap(real_to_complex_iso_vector)(sol.ys)


@jit
def calc_expect(op: jnp.ndarray, states: jnp.ndarray) -> jnp.ndarray:
    """Calculate expectation value of an operator given a list of states.

    Args:
        op: operator
        states: list of states

    Returns:
        list of expectation values
    """

    def calc_expect_ket_single(state: jnp.ndarray):
        return (dag(state) @ op @ state)[0][0]

    def calc_expect_dm_single(state: jnp.ndarray):
        return jnp.trace(op @ state)

    if is_1d(states[0]):
        return vmap(calc_expect_ket_single)(states)
    else:
        return vmap(calc_expect_dm_single)(states)

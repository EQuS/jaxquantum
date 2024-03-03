"""Solvers"""

from functools import partial
from typing import Callable, List, Optional

from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
from jax import jit, vmap, Array
import jax.numpy as jnp

from jaxquantum.utils.utils import (
    real_to_complex_iso_matrix,
    real_to_complex_iso_vector,
    complex_to_real_iso_matrix,
    complex_to_real_iso_vector,
    imag_times_iso_vector,
    imag_times_iso_matrix,
    conj_transpose_iso_matrix,
)

from jaxquantum.core.qarray import Qarray
from jaxquantum.core.conversions import jnps2jqts, jqts2jnps



# ----

@jit
def calc_expect(op: Qarray, states: List[Qarray]) -> Array:
    """Calculate expectation value of an operator given a list of states.

    Args:
        op: operator
        states: list of states

    Returns:
        list of expectation values
    """

    op = op.data
    is_dm = states[0].is_dm()
    states = jqts2jnps(states)

    def calc_expect_ket_single(state: Array):
        return (jnp.conj(state).T @ op @ state)[0][0]

    def calc_expect_dm_single(state: Array):
        return jnp.trace(op @ state)

    if is_dm:
        return vmap(calc_expect_dm_single)(states)
    else:
        return vmap(calc_expect_ket_single)(states)
        

# ----

# ----

def spre(op: Array) -> Callable[[Array], Array]:
    """Superoperator generator.

    Args:
        op: operator to be turned into a superoperator

    Returns:
        superoperator function
    """
    op_dag = op.conj().T
    return lambda rho: 0.5 * (
        2 * op @ rho @ op_dag - rho @ op_dag @ op - op_dag @ op @ rho
    )

@partial(
    jit,
    static_argnums=(4,),
)
def mesolve(
    ρ0: Qarray,
    t_list: Array,
    c_ops: Optional[List[Qarray]] = None,
    H0: Optional[Qarray] = None,
    Ht: Optional[Callable[[float], Qarray]] = None,
):
    """Quantum Master Equation solver.

    Args:
        ρ0: initial state, must be a density matrix. For statevector evolution, please use sesolve.
        t_list: time list
        c_ops: list of collapse operators
        H0: time independent Hamiltonian. If H0 is not None, it will override Ht.
        Ht: time dependent Hamiltonian function.

    Returns:
        list of states
    """
    dims = ρ0.dims
    ρ0 = jnp.asarray(ρ0.data) + 0.0j
    c_ops = c_ops or []
    c_ops = jnp.asarray([c_op.data for c_op in c_ops]) + 0.0j
    H0 = jnp.asarray(H0.data) + 0.0j if H0 is not None else None

    def f(
        t: float,
        rho: Array,
        args: Array,
    ):
        H0_val = args[0]
        c_ops_val = args[1]

        if H0_val is not None:
            H = H0_val  # use H0 if given
        else:
            H = Ht(t).data  # type: ignore
            H = H + 0.0j

        rho_dot = -1j * (H @ rho - rho @ H)

        for op in c_ops_val:
            rho_dot += spre(op)(rho)

        return rho_dot

    term = ODETerm(f)
    solver = Dopri5()
    saveat = SaveAt(ts=t_list)
    stepsize_controller = PIDController(rtol=1e-6, atol=1e-8)

    sol = diffeqsolve(
        term,
        solver,
        t0=t_list[0],
        t1=t_list[-1],
        dt0=t_list[1] - t_list[0],
        y0=ρ0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        args=[H0, c_ops],
        max_steps=16**5,
    )

    return jnps2jqts(sol.ys, dims=dims)

@partial(
    jit,
    static_argnums=(3,),
)
def sesolve(
    ψ: Qarray,
    t_list: Array,
    H0: Optional[Qarray] = None,
    Ht: Optional[Callable[[float], Qarray]] = None,
):
    """Schrödinger Equation solver.

    Args:
        ψ: initial statevector
        t_list: time list
        H0: time independent Hamiltonian. If H0 is not None, it will override Ht.
        Ht: time dependent Hamiltonian function.

    Returns:
        list of states
    """

    dims = ψ.dims

    ψ = jnp.asarray(ψ.data) + 0.0j
    H0 = jnp.asarray(H0.data) + 0.0j if H0 is not None else None

    def f(
        t: float,
        ψₜ: Array,
        args: Array,
    ):
        H0_val = args[0]

        if H0_val is not None:
            H = H0_val  # use H0 if given
        else:
            H = Ht(t).data  # type: ignore
        # print("H", H.shape)
        # print("psit", ψₜ.shape)
        ψₜ_dot = -1j * (H @ ψₜ)

        return ψₜ_dot

    term = ODETerm(f)
    solver = Dopri5()
    saveat = SaveAt(ts=t_list)
    stepsize_controller = PIDController(rtol=1e-6, atol=1e-6)

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

    return jnps2jqts(sol.ys, dims=dims)

# ----



# ----

def spre_iso(op: Array) -> Callable[[Array], Array]:
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
def mesolve_iso(
    ρ0: Qarray,
    t_list: Array,
    c_ops: Optional[List[Qarray]] = None,
    H0: Optional[Qarray] = None,
    Ht: Optional[Callable[[float], Qarray]] = None,
):
    """Quantum Master Equation solver.

    Args:
        ρ0: initial state, must be a density matrix. For statevector evolution, please use sesolve.
        t_list: time list
        c_ops: list of collapse operators
        H0: time independent Hamiltonian. If H0 is not None, it will override Ht.
        Ht: time dependent Hamiltonian function.

    Returns:
        list of states
    """
    
    dims = ρ0.dims

    ρ0 = jnp.asarray(ρ0.data) + 0.0j
    c_ops = c_ops or []
    c_ops = jnp.asarray([c_op.data for c_op in c_ops]) + 0.0j
    H0 = jnp.asarray(H0.data) + 0.0j if H0 is not None else None

    ρ0 = complex_to_real_iso_matrix(jnp.asarray(ρ0) + 0.0j)
    c_ops = vmap(complex_to_real_iso_matrix)(jnp.asarray(c_ops) + 0.0j)
    H0 = None if H0 is None else complex_to_real_iso_matrix(jnp.asarray(H0) + 0.0j)

    def f(
        t: float,
        rho: Array,
        args: Array,
    ):
        H0_val = args[0]
        c_ops_val = args[1]

        if H0_val is not None:
            H = H0_val  # use H0 if given
        else:
            H = Ht(t).data  # type: ignore
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
        y0=ρ0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        args=[H0, c_ops],
        max_steps=16**5,
    )

    return jnps2jqts(vmap(real_to_complex_iso_matrix)(sol.ys), dims=dims)

@partial(
    jit,
    static_argnums=(3,),
)
def sesolve_iso(
    ψ: Qarray,
    t_list: Array,
    H0: Optional[Qarray] = None,
    Ht: Optional[Callable[[float], Qarray]] = None,
):
    """Schrödinger Equation solver.

    Args:
        ψ: initial statevector
        t_list: time list
        H0: time independent Hamiltonian. If H0 is not None, it will override Ht.
        Ht: time dependent Hamiltonian function.

    Returns:
        list of states
    """
    dims = ψ.dims

    ψ = jnp.asarray(ψ.data) + 0.0j
    H0 = jnp.asarray(H0.data) + 0.0j if H0 is not None else None
    
    ψ = complex_to_real_iso_vector(jnp.asarray(ψ) + 0.0j)
    H0 = None if H0 is None else complex_to_real_iso_matrix(jnp.asarray(H0) + 0.0j)

    def f(
        t: float,
        ψₜ: Array,
        args: Array,
    ):
        H0_val = args[0]

        if H0_val is not None:
            H = H0_val  # use H0 if given
        else:
            H = Ht(t).data  # type: ignore
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

    return jnps2jqts(vmap(real_to_complex_iso_vector)(sol.ys), dims=dims)

# ----


"""Solvers"""

from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController, TqdmProgressMeter
from functools import partial
from jax import jit, vmap, Array
from typing import Callable, List, Optional
import diffrax
import jax.numpy as jnp
import warnings
import tqdm


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


class CustomProgressMeter(TqdmProgressMeter):
    @staticmethod
    def _init_bar() -> tqdm.tqdm:
        return tqdm.tqdm(total=100, unit='%', colour="MAGENTA", ascii="░▒█")
    

def solve(ρ0, f, t_list, args, solver_options):
    """ Gets teh desired solver from diffrax.

    Args:
        solver_options: dictionary with solver options

    Returns:
        solution 
    """

    # f and ts
    term = ODETerm(f)
    saveat = SaveAt(ts=t_list)

    # solver 
    solver_name = solver_options.get("solver", "Tsit5")
    solver = getattr(diffrax, solver_name)()
    stepsize_controller = PIDController(rtol=1e-6, atol=1e-6)

    # solve!
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning) # NOTE: suppresses complex dtype warning in diffrax
        sol = diffeqsolve(
            term,
            solver,
            t0=t_list[0],
            t1=t_list[-1],
            dt0=t_list[1] - t_list[0],
            y0=ρ0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            args=args,
            max_steps=solver_options.get("max_steps", 100_000),
            progress_meter=CustomProgressMeter()
        )   

    return sol

@partial(
    jit,
    static_argnums=(4,5),
)
def mesolve(
    ρ0: Qarray,
    t_list: Array,
    c_ops: Optional[List[Qarray]] = None,
    H0: Optional[Qarray] = None,
    Ht: Optional[Callable[[float], Qarray]] = None,
    solver_options: Optional[dict] = None,
):
    """Quantum Master Equation solver.

    Args:
        ρ0: initial state, must be a density matrix. For statevector evolution, please use sesolve.
        t_list: time list
        c_ops: list of collapse operators
        H0: time independent Hamiltonian. If H0 is not None, it will override Ht.
        Ht: time dependent Hamiltonian function.
        solver_options: dictionary with solver options

    Returns:
        list of states
    """
    dims = ρ0.dims
    ρ0 = jnp.asarray(ρ0.data) + 0.0j
    c_ops = c_ops or []
    c_ops = jnp.asarray([c_op.data for c_op in c_ops]) + 0.0j
    H0 = jnp.asarray(H0.data) + 0.0j if H0 is not None else None
    solver_options = solver_options or {}

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

    
    sol = solve(ρ0, f, t_list, [H0, c_ops], solver_options)

    return jnps2jqts(sol.ys, dims=dims)

@partial(
    jit,
    static_argnums=(3,4),
)
def sesolve(
    ψ: Qarray,
    t_list: Array,
    H0: Optional[Qarray] = None,
    Ht: Optional[Callable[[float], Qarray]] = None,
    solver_options: Optional[dict] = None,
):
    """Schrödinger Equation solver.

    Args:
        ψ: initial statevector
        t_list: time list
        H0: time independent Hamiltonian. If H0 is not None, it will override Ht.
        Ht: time dependent Hamiltonian function.
        solver_options: dictionary with solver options

    Returns:
        list of states
    """

    dims = ψ.dims

    ψ = jnp.asarray(ψ.data) + 0.0j
    H0 = jnp.asarray(H0.data) + 0.0j if H0 is not None else None
    solver_options = solver_options or {}


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


    sol = solve(ψ, f, t_list, [H0], solver_options)


    return jnps2jqts(sol.ys, dims=dims)

# ----
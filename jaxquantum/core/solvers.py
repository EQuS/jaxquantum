"""Solvers"""


from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController, TqdmProgressMeter, NoProgressMeter
from functools import partial
from flax import struct
from jax import vmap, Array
from typing import Callable, List, Optional, Dict, Union
import diffrax
import jax.numpy as jnp
import jax.scipy as jsp
import warnings
import tqdm
import logging



from jaxquantum.core.qarray import Qarray, Qtypes
from jaxquantum.core.conversions import jnps2jqts, jqts2jnps, jnp2jqt
from jaxquantum.utils.utils import robust_isscalar


# ----

@struct.dataclass
class SolverOptions:
    progress_meter: bool = struct.field(pytree_node=False)
    solver: str = struct.field(pytree_node=False),
    max_steps: int = struct.field(pytree_node=False),
    

    @classmethod
    def create(cls, progress_meter: bool = True, solver: str = "Tsit5", max_steps: int = 100_000):
        return cls(progress_meter, solver, max_steps)


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
        bar_format = "{desc}: {percentage:3.0f}% |{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        return tqdm.tqdm(total=100, bar_format=bar_format, unit='%', colour="MAGENTA", ascii="░▒█")
    
def solve(ρ0, f, t_list, args, solver_options: Optional[SolverOptions] = None):
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
    solver_options = solver_options or SolverOptions.create()
    
    solver_name = solver_options.solver
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
            max_steps=solver_options.max_steps,
            progress_meter=CustomProgressMeter() if solver_options.progress_meter else NoProgressMeter(),
        )   

    return sol

def mesolve(
    ρ0: Qarray,
    t_list: Array,
    c_ops: Optional[List[Qarray]] = None,
    H0: Optional[Qarray] = None,
    Ht: Optional[Callable[[float], Qarray]] = None,
    solver_options: Optional[SolverOptions] = None
):
    """Quantum Master Equation solver.

    Args:
        ρ0: initial state, must be a density matrix. For statevector evolution, please use sesolve.
        t_list: time list
        c_ops: list of collapse operators
        H0: time independent Hamiltonian. If H0 is not None, it will override Ht.
        Ht: time dependent Hamiltonian function.
        solver_options: SolverOptions with solver options

    Returns:
        list of states
    """
    
    c_ops = c_ops or []

    if len(c_ops) == 0 and ρ0.qtype != Qtypes.oper:
        logging.warning(
            "Consider using `jqt.sesolve()` instead, as `c_ops` is an empty list and the initial state is not a density matrix."
        )

    ρ0 = ρ0.to_dm()
    dims = ρ0.dims
    ρ0 = ρ0.data

    c_ops = [c_op.data for c_op in c_ops]
    H0 = jnp.asarray(H0.data) if H0 is not None else None
    Ht_data = lambda t: Ht(t).data if Ht is not None else None
    
    ys = mesolve_data(ρ0, t_list, c_ops, H0, Ht_data, solver_options)

    return jnps2jqts(ys, dims=dims)


def mesolve_data(
    ρ0: Array,
    t_list: Array,
    c_ops: Optional[List[Array]] = None,
    H0: Optional[Array] = None,
    Ht: Optional[Callable[[float], Array]] = None,
    solver_options: Optional[SolverOptions] = None
):
    """Quantum Master Equation solver.

    Args:
        ρ0: initial state, must be a density matrix. For statevector evolution, please use sesolve.
        t_list: time list
        c_ops: list of collapse operators
        H0: time independent Hamiltonian. If H0 is not None, it will override Ht.
        Ht: time dependent Hamiltonian function.
        solver_options: SolverOptions with solver options

    Returns:
        list of states
    """
    
    c_ops = c_ops or []

    if len(c_ops) == 0:
        logging.warning(
            "Consider using `jqt.sesolve()` instead, as `c_ops` is an empty list and the initial state is not a density matrix."
        )

    ρ0 = ρ0 + 0.0j

    c_ops = jnp.asarray([c_op for c_op in c_ops]) + 0.0j
    H0 = H0 + 0.0j if H0 is not None else None

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
            H = Ht(t)  # type: ignore
            H = H + 0.0j

        rho_dot = -1j * (H @ rho - rho @ H)

        for op in c_ops_val:
            rho_dot += spre(op)(rho)

        return rho_dot

    
    sol = solve(ρ0, f, t_list, [H0, c_ops], solver_options=solver_options)

    return sol.ys

def sesolve(
    ψ: Qarray,
    t_list: Array,
    H0: Optional[Qarray] = None,
    Ht: Optional[Callable[[float], Qarray]] = None,
    solver_options: Optional[SolverOptions] = None,
):
    """Schrödinger Equation solver.

    Args:
        ψ: initial statevector
        t_list: time list
        H0: time independent Hamiltonian. If H0 is not None, it will override Ht.
        Ht: time dependent Hamiltonian function.
        solver_options: SolverOptions with solver options

    Returns:
        list of states
    """

    if ψ.qtype == Qtypes.oper:
        raise ValueError(
            "Please use `jqt.mesolve` for initial state inputs in density matrix form."
        )
    
    ψ = ψ.to_ket()

    dims = ψ.dims

    ψ = ψ.data 
    H0 = H0.data if H0 is not None else None
    Ht_data = lambda t: Ht(t).data if Ht is not None else None
    
    ys = sesolve_data(ψ, t_list, H0, Ht_data, solver_options)

    return jnps2jqts(ys, dims=dims)

def sesolve_data(
    ψ: Array,
    t_list: Array,
    H0: Optional[Array] = None,
    Ht: Optional[Callable[[float], Array]] = None,
    solver_options: Optional[SolverOptions] = None,
):
    """Schrödinger Equation solver.

    Args:
        ψ: initial statevector
        t_list: time list
        H0: time independent Hamiltonian. If H0 is not None, it will override Ht.
        Ht: time dependent Hamiltonian function.
        solver_options: SolverOptions with solver options

    Returns:
        list of states
    """
    
    ψ = ψ + 0.0j
    H0 = H0 + 0.0j if H0 is not None else None
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
            H = Ht(t)  # type: ignore
        # print("H", H.shape)
        # print("psit", ψₜ.shape)
        ψₜ_dot = -1j * (H @ ψₜ)

        return ψₜ_dot


    sol = solve(ψ, f, t_list, [H0], solver_options=solver_options)
    return sol.ys

# ----


# propagators 
# ----


def propagator(
    H: Union[Qarray, Callable[[float], Qarray]],
    t: Union[float, Array],
    solver_options=None
):
    """ Generate the propagator for a time dependent Hamiltonian. 

    Args:
        H (Qarray or callable): 
            A Qarray static Hamiltonian OR
            a function that takes a time argument and returns a Hamiltonian. 
        ts (float or Array): 
            A single time point or
            an Array of time points.
        
    Returns:
        Qarray or List[Qarray]: 
            The propagator for the Hamiltonian at time t.
            OR a list of propagators for the Hamiltonian at each time in t.

    """

    t_is_scalar = robust_isscalar(t)

    if isinstance(H, Qarray):
        dims = H.dims 
        if t_is_scalar:
            return jnp2jqt(propagator_0_data(H.data,t), dims=dims)
        else:
            f = lambda t: propagator_0_data(H.data,t)
            return jnps2jqts(vmap(f)(t), dims)
    else:
        dims = H(0.0).dims
        H_data = lambda t: H(t).data
        if t_is_scalar:
            return jnp2jqt(
                propagator_t_data(H_data, t, solver_options=solver_options),
                dims=dims
            )
        else:
            f = lambda t: propagator_t_data(H_data, t, solver_options=solver_options)
            return jnps2jqts(vmap(f)(t), dims)

def propagator_0_data(
    H0: Array,
    t: float
):
    """ Generate the propagator for a time independent Hamiltonian. 

    Args:
        H0 (Qarray): The Hamiltonian.

    Returns:
        Qarray: The propagator for the time independent Hamiltonian.
    """
    return jsp.linalg.expm(-1j * H0 * t)

def propagator_t_data(
    Ht: Callable[[float], Array],
    t: float, 
    solver_options=None
):
    """ Generate the propagator for a time dependent Hamiltonian. 

    Args:
        t (float): The final time of the propagator. 
            Warning: Do not send in t. In this case, just do exp(-1j*Ht(0.0)).
        Ht (callable): A function that takes a time argument and returns a Hamiltonian. 
        solver_options (dict): Options to pass to the solver.

    Returns:
        Qarray: The propagator for the time dependent Hamiltonian for the time range [0, t_final].
    """
    ts = jnp.linspace(0,t,2)
    N = Ht(0).shape[0]
    basis_states = jnp.eye(N)

    def propogate_state(initial_state):
        return sesolve_data(initial_state, ts, Ht=Ht, solver_options=solver_options)[1]
        
    U_prop = vmap(propogate_state)(basis_states)
    return U_prop
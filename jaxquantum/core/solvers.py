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



from jaxquantum.core.qarray import Qarray, Qtypes, is_dm_data, dag_data
from jaxquantum.core.conversions import jnp2jqt
from jaxquantum.utils.utils import robust_isscalar
from jaxquantum.core.operators import identity_like
from jaxquantum.core.helpers import overlap

# ----

@struct.dataclass
class SolverOptions:
    progress_meter: bool = struct.field(pytree_node=False)
    solver: str = struct.field(pytree_node=False),
    max_steps: int = struct.field(pytree_node=False),
    

    @classmethod
    def create(cls, progress_meter: bool = True, solver: str = "Tsit5", max_steps: int = 100_000):
        return cls(progress_meter, solver, max_steps)


# Vestigial Functions ----
def calc_expect(op: Qarray, states: Qarray) -> Array:
    return overlap(op, states)

def spre(op: Qarray) -> Callable[[Qarray], Qarray]:
    """Superoperator generator.

    Args:
        op: operator to be turned into a superoperator

    Returns:
        superoperator function
    """
    op_dag = op.dag()
    return lambda rho: 0.5 * (
        2 * op @ rho @ op_dag - rho @ op_dag @ op - op_dag @ op @ rho
    )
# ---


class CustomProgressMeter(TqdmProgressMeter):
    @staticmethod
    def _init_bar() -> tqdm.tqdm:
        bar_format = "{desc}: {percentage:3.0f}% |{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        return tqdm.tqdm(total=100, bar_format=bar_format, unit='%', colour="MAGENTA", ascii="░▒█")
    
def solve(f, ρ0, tlist, args, solver_options: Optional[SolverOptions] = None):
    """ Gets teh desired solver from diffrax.

    Args:
        solver_options: dictionary with solver options

    Returns:
        solution 
    """

    # f and ts
    term = ODETerm(f)
    saveat = SaveAt(ts=tlist)

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
            t0=tlist[0],
            t1=tlist[-1],
            dt0=tlist[1] - tlist[0],
            y0=ρ0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            args=args,
            max_steps=solver_options.max_steps,
            progress_meter=CustomProgressMeter() if solver_options.progress_meter else NoProgressMeter(),
        )   

    return sol

def mesolve(
    H: Union[Qarray, Callable[[float], Qarray]],
    rho0: Qarray,
    tlist: Array,
    c_ops: Optional[Qarray] = None,
    solver_options: Optional[SolverOptions] = None
) -> Qarray:
    """Quantum Master Equation solver.

    Args:
        H: time dependent Hamiltonian function or time-independent Qarray.
        rho0: initial state, must be a density matrix. For statevector evolution, please use sesolve.
        tlist: time list
        c_ops: qarray list of collapse operators
        solver_options: SolverOptions with solver options

    Returns:
        list of states
    """
    
    c_ops = c_ops if c_ops is not None else jqt.Qarray.from_list([])

    # if isinstance(H, Qarray):
        

    if len(c_ops) == 0 and ρ0.qtype != Qtypes.oper:
        logging.warning(
            "Consider using `jqt.sesolve()` instead, as `c_ops` is an empty list and the initial state is not a density matrix."
        )

    ρ0 = rho0.to_dm()
    dims = ρ0.dims
    ρ0 = ρ0.data

    c_ops = c_ops.data

    if isinstance(H, Qarray):
        Ht_data = lambda t: H.data
    else:
        Ht_data = lambda t: H(t).data if H is not None else None
    
    ys = mesolve_data(Ht_data, ρ0, tlist, c_ops, solver_options=solver_options)

    return jnp2jqt(ys, dims=dims)


def mesolve_data(
    H: Callable[[float], Array],
    rho0: Array,
    tlist: Array,
    c_ops: Optional[Qarray] = None,
    solver_options: Optional[SolverOptions] = None
) -> Array:
    """Quantum Master Equation solver.

    Args:
        H: time dependent Hamiltonian function or time-independent Array.
        rho0: initial state, must be a density matrix. For statevector evolution, please use sesolve.
        tlist: time list
        c_ops: qarray list of collapse operators
        solver_options: SolverOptions with solver options

    Returns:
        list of states
    """
    
    c_ops = c_ops if c_ops is not None else jnp.array([])

    if len(c_ops) == 0 and not is_dm_data(rho0):
        logging.warning(
            "Consider using `jqt.sesolve()` instead, as `c_ops` is an empty list and the initial state is not a density matrix."
        )

    ρ0 = rho0 + 0.0j

    padded_dim = [1 for _ in range(len(ρ0.shape) - 2)]
    c_ops = c_ops.reshape(len(c_ops), *padded_dim, c_ops.shape[-2], c_ops.shape[-1])

    def f(
        t: float,
        rho: Array,
        c_ops_val: Array,
    ):

        H_val = H(t)  # type: ignore
        H_val = H_val + 0.0j

        rho_dot = -1j * (H_val @ rho - rho @ H_val)

        c_ops_val_dag = dag_data(c_ops_val)

        rho_dot_delta = (0.5 * (
            2 * c_ops_val @ rho @ c_ops_val_dag
            - rho @ c_ops_val_dag @ c_ops_val
            - c_ops_val_dag @ c_ops_val @ rho
        ))

        rho_dot_delta = jnp.sum(rho_dot_delta, axis=0)

        rho_dot += rho_dot_delta
        
        return rho_dot

    
    sol = solve(f, ρ0, tlist, c_ops, solver_options=solver_options)

    return sol.ys

def sesolve(
    H: Union[Qarray, Callable[[float], Qarray]],
    rho0: Qarray,
    tlist: Array,
    solver_options: Optional[SolverOptions] = None
) -> Qarray:
    """Schrödinger Equation solver.

    Args:
        H: time dependent Hamiltonian function or time-independent Qarray.
        rho0: initial state, must be a density matrix. For statevector evolution, please use sesolve.
        tlist: time list
        solver_options: SolverOptions with solver options

    Returns:
        list of states
    """

    ψ = rho0

    if ψ.qtype == Qtypes.oper:
        raise ValueError(
            "Please use `jqt.mesolve` for initial state inputs in density matrix form."
        )
    
    ψ = ψ.to_ket()
    dims = ψ.dims
    ψ = ψ.data 

    if isinstance(H, Qarray):
        Ht_data = lambda t: H.data
    else:
        Ht_data = lambda t: H(t).data if H is not None else None
    
    ys = sesolve_data(Ht_data, ψ, tlist, solver_options=solver_options)

    return jnp2jqt(ys, dims=dims)

def sesolve_data(
    H: Callable[[float], Array],
    rho0: Array,
    tlist: Array,
    solver_options: Optional[SolverOptions] = None
):
    """Schrödinger Equation solver.

    Args:
        H: time dependent Hamiltonian function or time-independent Array.
        rho0: initial state, must be a density matrix. For statevector evolution, please use sesolve.
        tlist: time list
        solver_options: SolverOptions with solver options

    Returns:
        list of states
    """
    
    ψ = rho0
    ψ = ψ + 0.0j

    def f(
        t: float,
        ψₜ: Array,
        _
    ):

        H_val = H(t)  # type: ignore
        H_val = H_val + 0.0j
        
        ψₜ_dot = -1j * (H_val @ ψₜ)

        return ψₜ_dot


    sol = solve(f, ψ, tlist, None, solver_options=solver_options)
    return sol.ys

# ----


# propagators 
# ----


# def propagator(
#     H: Union[Qarray, Callable[[float], Qarray]],
#     t: Union[float, Array],
#     solver_options=None
# ):
#     """ Generate the propagator for a time dependent Hamiltonian. 

#     Args:
#         H (Qarray or callable): 
#             A Qarray static Hamiltonian OR
#             a function that takes a time argument and returns a Hamiltonian. 
#         ts (float or Array): 
#             A single time point or
#             an Array of time points.
        
#     Returns:
#         Qarray or List[Qarray]: 
#             The propagator for the Hamiltonian at time t.
#             OR a list of propagators for the Hamiltonian at each time in t.

#     """

#     t_is_scalar = robust_isscalar(t)

#     if isinstance(H, Qarray):
#         dims = H.dims 
#         if t_is_scalar:
#             if t == 0:
#                 return identity_like(H)

#             return jnp2jqt(propagator_0_data(H.data,t), dims=dims)
#         else:
#             f = lambda t: propagator_0_data(H.data,t)
#             return jnp2jqt(vmap(f)(t), dims)
#     else:
#         dims = H(0.0).dims
#         H_data = lambda t: H(t).data
#         if t_is_scalar:
#             if t == 0:
#                 return identity_like(H(0.0))

#             ts = jnp.linspace(0,t,2)
#             return jnp2jqt(
#                 propagator_t_data(H_data, ts, solver_options=solver_options)[1],
#                 dims=dims
#             )
#         else:
#             ts = t 
#             U_props = propagator_t_data(H_data, ts, solver_options=solver_options)
#             return jnp2jqt(U_props, dims)

# def propagator_0_data(
#     H0: Array,
#     t: float
# ):
#     """ Generate the propagator for a time independent Hamiltonian. 

#     Args:
#         H0 (Qarray): The Hamiltonian.

#     Returns:
#         Qarray: The propagator for the time independent Hamiltonian.
#     """
#     return jsp.linalg.expm(-1j * H0 * t)

# def propagator_t_data(
#     Ht: Callable[[float], Array],
#     ts: Array, 
#     solver_options=None
# ):
    """ Generate the propagator for a time dependent Hamiltonian. 

    Args:
        ts (float): The final time of the propagator. 
            Warning: Do not send in t. In this case, just do exp(-1j*Ht(0.0)).
        Ht (callable): A function that takes a time argument and returns a Hamiltonian. 
        solver_options (dict): Options to pass to the solver.

    Returns:
        Qarray: The propagator for the time dependent Hamiltonian for the time range [0, t_final].
    """
    N = Ht(0).shape[0]
    basis_states = jnp.eye(N)

    def propogate_state(initial_state):
        return sesolve_data(initial_state, ts, Ht=Ht, solver_options=solver_options)
        
    U_prop = vmap(propogate_state)(basis_states)
    U_prop = U_prop.transpose(1,0,2) # move time axis to the front
    return U_prop
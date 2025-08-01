"""Solvers"""

from diffrax import (
    diffeqsolve,
    ODETerm,
    SaveAt,
    PIDController,
    TqdmProgressMeter,
    NoProgressMeter,
)
from flax import struct
from jax import Array
from typing import Callable, Optional, Union
import diffrax
import jax.numpy as jnp
import warnings
import tqdm
import logging


from jaxquantum.core.qarray import Qarray, Qtypes, dag_data
from jaxquantum.core.conversions import jnp2jqt
from jaxquantum.core.operators import identity_like, multi_mode_basis_set
from jaxquantum.utils.utils import robust_isscalar

# ----


@struct.dataclass
class SolverOptions:
    progress_meter: bool = struct.field(pytree_node=False)
    solver: str = (struct.field(pytree_node=False),)
    max_steps: int = (struct.field(pytree_node=False),)
    rtol: float = (struct.field(pytree_node=False),)
    atol: float = (struct.field(pytree_node=False),)

    @classmethod
    def create(
        cls,
        progress_meter: bool = True,
        solver: str = "Tsit5",
        max_steps: int = 100_000,
        rtol: float = 1e-7,
        atol: float = 1e-9,
    ):
        return cls(progress_meter, solver, max_steps, rtol, atol)


class CustomProgressMeter(TqdmProgressMeter):
    @staticmethod
    def _init_bar() -> tqdm.tqdm:
        bar_format = "{desc}: {percentage:3.0f}% |{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        return tqdm.tqdm(
            total=100, bar_format=bar_format, unit="%", colour="MAGENTA", ascii="░▒█"
        )


def solve(f, ρ0, tlist, args, solver_options: Optional[SolverOptions] = None):
    """Gets teh desired solver from diffrax.

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
    stepsize_controller = PIDController(rtol=solver_options.rtol, atol=solver_options.atol)

    # solve!
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                message="Complex dtype support in Diffrax",
                                category=UserWarning)  # NOTE: suppresses complex dtype warning in diffrax
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
            progress_meter=CustomProgressMeter()
            if solver_options.progress_meter
            else NoProgressMeter(),
        )

    return sol


def mesolve(
    H: Union[Qarray, Callable[[float], Qarray]],
    rho0: Qarray,
    tlist: Array,
    c_ops: Optional[Qarray] = None,
    solver_options: Optional[SolverOptions] = None,
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

    c_ops = c_ops if c_ops is not None else Qarray.from_list([])

    # if isinstance(H, Qarray):

    if len(c_ops) == 0 and rho0.qtype != Qtypes.oper:
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

    ys = _mesolve_data(Ht_data, ρ0, tlist, c_ops, solver_options=solver_options)

    return jnp2jqt(ys, dims=dims)


def _mesolve_data(
    H: Callable[[float], Array],
    rho0: Array,
    tlist: Array,
    c_ops: Optional[Qarray] = None,
    solver_options: Optional[SolverOptions] = None,
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

    # check is in mesolve
    # if len(c_ops) == 0 and not is_dm_data(rho0):
    #     logging.warning(
    #         "Consider using `jqt.sesolve()` instead, as `c_ops` is an empty list and the initial state is not a density matrix."
    #     )

    ρ0 = rho0 + 0.0j

    if len(c_ops) == 0:
        test_data = H(0.0) @ ρ0
    else:
        test_data = c_ops[0] @ H(0.0) @ ρ0

    ρ0 = jnp.resize(ρ0, test_data.shape)  # ensure correct shape

    if len(c_ops) != 0:
        c_ops_bdims = c_ops.shape[:-2]
        c_ops = c_ops.reshape(*c_ops_bdims, c_ops.shape[-2], c_ops.shape[-1])

    def f(
        t: float,
        rho: Array,
        c_ops_val: Array,
    ):
        H_val = H(t)  # type: ignore
        H_val = H_val + 0.0j

        rho_dot = -1j * (H_val @ rho - rho @ H_val)

        if len(c_ops_val) == 0:
            return rho_dot

        c_ops_val_dag = dag_data(c_ops_val)

        rho_dot_delta = 0.5 * (
            2 * c_ops_val @ rho @ c_ops_val_dag
            - rho @ c_ops_val_dag @ c_ops_val
            - c_ops_val_dag @ c_ops_val @ rho
        )

        rho_dot_delta = jnp.sum(rho_dot_delta, axis=0)

        rho_dot += rho_dot_delta

        return rho_dot

    sol = solve(f, ρ0, tlist, c_ops, solver_options=solver_options)

    return sol.ys


def sesolve(
    H: Union[Qarray, Callable[[float], Qarray]],
    rho0: Qarray,
    tlist: Array,
    solver_options: Optional[SolverOptions] = None,
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

    ys = _sesolve_data(Ht_data, ψ, tlist, solver_options=solver_options)

    return jnp2jqt(ys, dims=dims)


def _sesolve_data(
    H: Callable[[float], Array],
    rho0: Array,
    tlist: Array,
    solver_options: Optional[SolverOptions] = None,
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

    def f(t: float, ψₜ: Array, _):
        H_val = H(t)  # type: ignore
        H_val = H_val + 0.0j

        ψₜ_dot = -1j * (H_val @ ψₜ)

        return ψₜ_dot

    ψ_test = f(0, ψ, None)
    ψ = jnp.resize(ψ, ψ_test.shape)  # ensure correct shape

    sol = solve(f, ψ, tlist, None, solver_options=solver_options)
    return sol.ys

# ----

# propagators
# ----

def propagator(
    H: Union[Qarray, Callable[[float], Qarray]],
    ts: Union[float, Array],
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
    

    ts_is_scalar = robust_isscalar(ts)
    H_is_qarray = isinstance(H, Qarray)

    if H_is_qarray:
        return (-1j * H * ts).expm()
    else:
        
        if ts_is_scalar:
            H_first = H(0.0)
            if ts == 0:
                return identity_like(H_first)
            ts = jnp.array([0.0, ts])
        else:
            H_first = H(ts[0])

        basis_states = multi_mode_basis_set(H_first.space_dims)
        results = sesolve(H, basis_states, ts)
        propagators_data = results.data.squeeze(-1)
        propagators = Qarray.create(propagators_data, dims=H_first.space_dims)
        
        return propagators

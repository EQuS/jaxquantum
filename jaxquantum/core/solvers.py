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


from jax.experimental import sparse as jax_sparse
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


def solve(f, ρ0, tlist, saveat_tlist, args, solver_options: Optional[
    SolverOptions] = None):
    """Gets teh desired solver from diffrax.

    Args:
        f: function defining the ODE
        ρ0: initial state
        tlist: time list
        saveat_tlist: list of times at which to save the state
            pass in [-1] to save only at final time
        args: additional arguments to f
        solver_options: dictionary with solver options

    Returns:
        solution
    """

    # f and ts
    term = ODETerm(f)
    
    if saveat_tlist.shape[0] == 1 and saveat_tlist == -1:
        saveat = SaveAt(t1=True)
    else:
        saveat = SaveAt(ts=saveat_tlist)

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
    saveat_tlist: Optional[Array] = None,
    c_ops: Optional[Qarray] = None,
    solver_options: Optional[SolverOptions] = None,
) -> Qarray:
    """Quantum Master Equation solver.

    Args:
        H: time dependent Hamiltonian function or time-independent Qarray.
        rho0: initial state, must be a density matrix. For statevector evolution, please use sesolve.
        tlist: time list
        saveat_tlist: list of times at which to save the state.
            If -1 or [-1], save only at final time.
            If None, save at all times in tlist. Default: None.
        c_ops: qarray list of collapse operators
        solver_options: SolverOptions with solver options

    Returns:
        list of states
    """

    saveat_tlist = saveat_tlist if saveat_tlist is not None else tlist

    saveat_tlist = jnp.atleast_1d(saveat_tlist)

    c_ops = c_ops if c_ops is not None else Qarray.from_list([])

    # if isinstance(H, Qarray):

    if len(c_ops) == 0 and rho0.qtype != Qtypes.oper:
        logging.warning(
            "Consider using `jqt.sesolve()` instead, as `c_ops` is an empty list and the initial state is not a density matrix."
        )

    ρ0 = rho0.to_dm()

    if robust_isscalar(H):
        H = H * identity_like(ρ0)  # treat scalar H as a multiple of the identity

    dims = ρ0.dims
    ρ0 = ρ0.data

    c_ops = c_ops.data

    if isinstance(H, Qarray):
        Ht_data = lambda t: H.data
    else:
        Ht_data = lambda t: H(t).data

    ys = _mesolve_data(Ht_data, ρ0, tlist, saveat_tlist, c_ops,
                       solver_options=solver_options)

    return jnp2jqt(ys, dims=dims)


def _mesolve_data(
    H: Callable[[float], Array],
    rho0: Array,
    tlist: Array,
    saveat_tlist: Array,
    c_ops: Optional[Qarray] = None,
    solver_options: Optional[SolverOptions] = None,
) -> Array:
    """Quantum Master Equation solver.

    Args:
        H: time dependent Hamiltonian function or time-independent Array.
        rho0: initial state, must be a density matrix. For statevector evolution, please use sesolve.
        tlist: time list
        saveat_tlist: list of times at which to save the state
            If -1 or [-1], save only at final time.
            If None, save at all times in tlist. Default: None.
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

    ρ0 = rho0

    # Shape inference: when c_ops contains batched operators (e.g. shape
    # (1, B, N, N)), the initial state ρ0 must be broadcast to (B, N, N) so
    # that the ODE RHS produces consistently shaped output.
    #
    # The output batch shape is the broadcast of:
    #   c_ops[0] batch dims  →  c_ops.shape[1:-2]  (outer batch index stripped)
    #   H batch dims         →  H(0.0).shape[:-2]
    #   ρ0 batch dims        →  ρ0.shape[:-2]
    # This is a pure shape calculation — no array values are materialised.
    H0_shape = H(0.0).shape
    if len(c_ops) == 0:
        batch_shape = jnp.broadcast_shapes(H0_shape[:-2], ρ0.shape[:-2])
    else:
        # c_ops.shape[1:-2]: strip the outermost (c_op index) dim and the two
        # matrix dims to get the batch dims that will be broadcast into ρ.
        batch_shape = jnp.broadcast_shapes(
            c_ops.shape[1:-2], H0_shape[:-2], ρ0.shape[:-2]
        )
    ρ0 = jnp.resize(ρ0, batch_shape + ρ0.shape[-2:])  # ensure correct shape

    if len(c_ops) != 0:
        c_ops_bdims = c_ops.shape[:-2]
        c_ops = c_ops.reshape(*c_ops_bdims, c_ops.shape[-2], c_ops.shape[-1])

    # Precompute the adjoint once, outside the ODE hot-loop.
    # dag_data dispatches to the correct impl (dense or sparse) automatically,
    # so c_ops_dag is BCOO when c_ops is sparse and a dense array otherwise.
    c_ops_dag = dag_data(c_ops) if len(c_ops) != 0 else c_ops

    def f(
        t: float,
        rho: Array,
        args,
    ):
        c_ops_val, c_ops_dag_val = args
        H_val = H(t)  # type: ignore

        rho_dot = -1j * (H_val @ rho - rho @ H_val)

        if len(c_ops_val) == 0:
            return rho_dot

        # Compute the Lindblad dissipator D[L](ρ) = L ρ L† - ½(L†L ρ + ρ L†L)
        # using only  (sparse L) @ (dense rho)  operations to support BCOO
        # collapse operators natively — no dense @ sparse required:
        #
        #   L ρ L†  = dag( L @ dag(L @ ρ) )     avoids the dense @ L† step
        #   L†L ρ   = L† @ (L @ ρ)              BCOO @ dense → dense ✓
        #   ρ L†L   = dag(L†L ρ)                dag of dense ✓  (ρ Hermitian)
        Lrho = c_ops_val @ rho
        LrhoLdag = dag_data(c_ops_val @ dag_data(Lrho))
        LdagLrho = c_ops_dag_val @ Lrho
        rhoLdagL = dag_data(LdagLrho)

        rho_dot_delta = 0.5 * (2 * LrhoLdag - LdagLrho - rhoLdagL)

        rho_dot_delta = jnp.sum(rho_dot_delta, axis=0)

        rho_dot += rho_dot_delta

        return rho_dot

    sol = solve(f, ρ0, tlist, saveat_tlist, (c_ops, c_ops_dag),
                solver_options=solver_options)

    return sol.ys


def sesolve(
    H: Union[Qarray, Callable[[float], Qarray]],
    rho0: Qarray,
    tlist: Array,
    saveat_tlist: Optional[Array] = None,
    solver_options: Optional[SolverOptions] = None,
) -> Qarray:
    """Schrödinger Equation solver.

    Args:
        H: time dependent Hamiltonian function or time-independent Qarray.
        rho0: initial state, must be a density matrix. For statevector evolution, please use sesolve.
        tlist: time list
        saveat_tlist: list of times at which to save the state.
            If -1 or [-1], save only at final time.
            If None, save at all times in tlist. Default: None.
        solver_options: SolverOptions with solver options

    Returns:
        list of states
    """

    saveat_tlist = saveat_tlist if saveat_tlist is not None else tlist

    saveat_tlist = jnp.atleast_1d(saveat_tlist)

    ψ = rho0

    if ψ.qtype == Qtypes.oper:
        raise ValueError(
            "Please use `jqt.mesolve` for initial state inputs in density matrix form."
        )

    ψ = ψ.to_ket()

    if robust_isscalar(H):
        H = H * identity_like(ψ)  # treat scalar H as a multiple of the identity

    dims = ψ.dims
    ψ = ψ.data

    if isinstance(H, Qarray):
        Ht_data = lambda t: H.data
    else:
        Ht_data = lambda t: H(t).data

    ys = _sesolve_data(Ht_data, ψ, tlist, saveat_tlist,
                       solver_options=solver_options)

    return jnp2jqt(ys, dims=dims)


def _sesolve_data(
    H: Callable[[float], Array],
    rho0: Array,
    tlist: Array,
    saveat_tlist: Array,
    solver_options: Optional[SolverOptions] = None,
):
    """Schrödinger Equation solver.

    Args:
        H: time dependent Hamiltonian function or time-independent Array.
        rho0: initial state, must be a density matrix. For statevector evolution, please use sesolve.
        tlist: time list
        saveat_tlist: list of times at which to save the state.
            If -1 or [-1], save only at final time.
            If None, save at all times in tlist. Default: None.
        solver_options: SolverOptions with solver options

    Returns:
        list of states
    """

    ψ = rho0

    def f(t: float, ψₜ: Array, _):
        H_val = H(t)  # type: ignore

        ψₜ_dot = -1j * (H_val @ ψₜ)

        return ψₜ_dot

    ψ_test = f(0, ψ, None)
    ψ = jnp.resize(ψ, ψ_test.shape)  # ensure correct shape

    sol = solve(f, ψ, tlist, saveat_tlist, None, solver_options=solver_options)
    return sol.ys

# ----

# propagators
# ----

def propagator(
    H: Union[Qarray, Callable[[float], Qarray]],
    ts: Union[float, Array],
    saveat_tlist: Optional[Array] = None,
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
        saveat_tlist: list of times at which to save the state.
            If -1 or [-1], save only at final time.
            If None, save at all times in tlist. Default: None.

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
        results = sesolve(H, basis_states, ts, saveat_tlist=saveat_tlist)
        propagators_data = results.data.squeeze(-1).mT
        propagators = Qarray.create(propagators_data, dims=H_first.space_dims)
        
        return propagators

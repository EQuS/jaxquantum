"""Solvers"""

from diffrax import (
    diffeqsolve,
    ODETerm,
    SaveAt,
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


from jax.experimental import sparse as _sparse

from jaxquantum.core.qarray import DenseImpl, Qarray, Qtypes, dag_data
from jaxquantum.core.operators import identity_like, multi_mode_basis_set
from jaxquantum.utils.utils import robust_isscalar


def _is_dense_array(x) -> bool:
    """True for a dense JAX array (not BCOO, not SparseDIA data)."""
    return not isinstance(x, _sparse.BCOO) and not getattr(x, "_is_sparse_dia", False)

# ----


@struct.dataclass
class SolverOptions:
    progress_meter: bool = struct.field(pytree_node=False, default=True)
    solver: str = struct.field(pytree_node=False, default="Tsit5")
    max_steps: int = struct.field(pytree_node=False, default=100_000)
    # Name of the diffrax stepsize controller and the kwargs fed into it. The
    # kwargs are controller-specific, so the two must be set together (e.g. the
    # default rtol/atol below only make sense for PIDController). Any other
    # diffrax controller works by pairing its name with its own kwargs, e.g.
    # stepsize_controller="ConstantStepSize", stepsize_controller_kwargs={}.
    stepsize_controller: str = struct.field(pytree_node=False, default="PIDController")
    stepsize_controller_kwargs: dict = struct.field(
        pytree_node=False,
        default_factory=lambda: {"rtol": 1e-7, "atol": 1e-9},
    )

    @classmethod
    def create(
        cls,
        progress_meter: bool = True,
        solver: str = "Tsit5",
        max_steps: int = 100_000,
        stepsize_controller: str = "PIDController",
        stepsize_controller_kwargs: Optional[dict] = None,
    ):
        if stepsize_controller_kwargs is None:
            # PIDController needs tolerances; other controllers get no kwargs
            # by default (pass them explicitly if the controller needs any).
            stepsize_controller_kwargs = (
                {"rtol": 1e-7, "atol": 1e-9}
                if stepsize_controller == "PIDController"
                else {}
            )
        return cls(
            progress_meter,
            solver,
            max_steps,
            stepsize_controller,
            stepsize_controller_kwargs,
        )



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
            pass in an empty list to save only at final time
        args: additional arguments to f
        solver_options: dictionary with solver options

    Returns:
        solution
    """

    # f and ts
    term = ODETerm(f)
    
    # An empty saveat_tlist means "save only the final state". We key on the
    # *static* length, which is jit-safe: under jit saveat_tlist is a traced
    # array whose element values can't be read in a Python `if` (e.g. `== -1`
    # would raise TracerBoolConversionError), but its length is part of the
    # static shape and is always known at trace time. A non-empty saveat_tlist
    # is passed straight through to SaveAt(ts=...), so a length-1 list is now an
    # ordinary single-time save rather than a final-only sentinel.
    if len(saveat_tlist) == 0:
        saveat = SaveAt(t1=True)
    else:
        saveat = SaveAt(ts=saveat_tlist)

    # solver
    solver_options = solver_options or SolverOptions.create()

    solver_name = solver_options.solver
    solver = getattr(diffrax, solver_name)()

    # Build the diffrax stepsize controller generically from its name and the
    # controller-specific kwargs dict (fall back to PIDController if unset).
    stepsize_controller_name = solver_options.stepsize_controller or "PIDController"
    stepsize_controller = getattr(diffrax, stepsize_controller_name)(
        **solver_options.stepsize_controller_kwargs
    )

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
            If empty (e.g. jnp.array([])), save only at final time.
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
        logging.warning(  # noqa: LOG015
            "Consider using `jqt.sesolve()` instead, as `c_ops` is an empty list and the initial state is not a density matrix."
        )

    ρ0 = rho0.to_dm().to_dense()

    if robust_isscalar(H):
        H = H * identity_like(ρ0)  # treat scalar H as a multiple of the identity

    qdims = ρ0.qdims
    ρ0 = ρ0.data

    c_ops = c_ops.data

    if isinstance(H, Qarray):
        Ht_data = lambda t: H.data
    else:
        Ht_data = lambda t: H(t).data

    ys = _mesolve_data(Ht_data, ρ0, tlist, saveat_tlist, c_ops,
                       solver_options=solver_options)

    return Qarray._from_impl(DenseImpl._make(ys), qdims)


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
            If empty (e.g. jnp.array([])), save only at final time.
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
    ρ0 = jnp.broadcast_to(ρ0, batch_shape + ρ0.shape[-2:])

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
            If empty (e.g. jnp.array([])), save only at final time.
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

    ψ = ψ.to_ket().to_dense()

    if robust_isscalar(H):
        H = H * identity_like(ψ)  # treat scalar H as a multiple of the identity

    qdims = ψ.qdims
    ψ = ψ.data

    if isinstance(H, Qarray):
        Ht_data = lambda t: H.data
    else:
        Ht_data = lambda t: H(t).data

    ys = _sesolve_data(Ht_data, ψ, tlist, saveat_tlist,
                       solver_options=solver_options)

    return Qarray._from_impl(DenseImpl._make(ys), qdims)


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
            If empty (e.g. jnp.array([])), save only at final time.
            If None, save at all times in tlist. Default: None.
        solver_options: SolverOptions with solver options

    Returns:
        list of states
    """

    ψ = rho0

    def f(t: float, ψₜ: Array, _):
        H_val = H(t)  # type: ignore

        # State vectors live on a single trailing axis (..., N). For a dense
        # Hamiltonian contract that axis directly via einsum (batch-safe, and no
        # (N,1) is ever materialised — keeps the scan carry 1-D). For a sparse
        # Hamiltonian use a transient column local to this RHS (sparse is not the
        # TPU-padding path).
        if _is_dense_array(H_val):
            ψₜ_dot = -1j * jnp.einsum("...ij,...j->...i", H_val, ψₜ)
        else:
            ψₜ_dot = -1j * (H_val @ ψₜ[..., None])[..., 0]

        return ψₜ_dot

    batch_shape = jnp.broadcast_shapes(H(0.0).shape[:-2], ψ.shape[:-1])
    ψ = jnp.broadcast_to(ψ, batch_shape + ψ.shape[-1:])

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
            If empty (e.g. jnp.array([])), save only at final time.
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
        # results.data is (T, M, M): T times, M evolved basis kets (batch), M
        # ket components. Transpose the last two axes so each time slice is a
        # propagator whose columns are the evolved basis states. No squeeze:
        # kets no longer carry a trailing singleton.
        propagators_data = results.data.mT
        propagators = Qarray.create(propagators_data, dims=H_first.space_dims)
        
        return propagators

import logging
import warnings
from collections.abc import Callable
from typing import Any, Literal

import diffrax
import jax.numpy as jnp
import tqdm
from flax import struct
from jax import Array
from jax.experimental import sparse as _sparse

from jaxquantum.core.dims import Qdims
from jaxquantum.core.operators import identity_like, multi_mode_basis_set
from jaxquantum.core.qarray import DenseImpl, Qarray, Qtypes, dag_data
from jaxquantum.utils.utils import robust_isscalar


def _is_dense_array(x) -> bool:
    """True for a dense JAX array (not BCOO, not SparseDIA data)."""
    return not isinstance(x, _sparse.BCOO) and not getattr(x, "_is_sparse_dia", False)


def _default_stepsize_controller() -> diffrax.AbstractStepSizeController:
    return diffrax.PIDController(rtol=1e-7, atol=1e-9)


@struct.dataclass
class SolverOptions:
    """Options forwarded to :func:`diffrax.diffeqsolve`.

    Attributes:
        solver: Native Diffrax solver; strings are deprecated.
        stepsize_controller: Native Diffrax controller; strings are deprecated.
        stepsize_controller_kwargs: Deprecated controller constructor arguments.
        saveat: Custom save policy; overrides JAXQuantum's save-time handling.
        dt0: Initial step, ``"tlist"`` for the first interval, or ``None`` for
            Diffrax's automatic choice.
        adjoint: Differentiation strategy. ``None`` uses Diffrax's default.
        event: Native Diffrax termination event.
        max_steps: Maximum solver steps.
        throw: Whether unsuccessful solves raise an exception.
        progress_meter: ``None``, ``"default"``, or a native progress meter.
            Booleans are deprecated.
        solver_state: Solver state used to continue a previous solve.
        controller_state: Controller state used to continue a previous solve.
        made_jump: Previous jump state used when continuing a solve.

    ``saveat=None`` saves at ``saveat_tlist`` (or ``tlist`` when omitted).
    ``adjoint=None`` and ``progress_meter=None`` preserve Diffrax's defaults.
    Native Diffrax objects pass through unchanged. Legacy values still work and
    issue a ``FutureWarning``.
    """

    progress_meter: bool | Literal["default"] | diffrax.AbstractProgressMeter | None = (
        struct.field(pytree_node=False, default="default")
    )
    solver: diffrax.AbstractSolver | str = struct.field(
        pytree_node=False, default_factory=diffrax.Tsit5
    )
    max_steps: int | None = struct.field(pytree_node=False, default=100_000)
    stepsize_controller: diffrax.AbstractStepSizeController | str = struct.field(
        pytree_node=False, default_factory=_default_stepsize_controller
    )
    stepsize_controller_kwargs: dict[str, Any] | None = struct.field(
        pytree_node=False, default=None
    )
    saveat: diffrax.SaveAt | None = struct.field(pytree_node=False, default=None)
    dt0: float | Array | None | Literal["tlist"] = struct.field(
        pytree_node=False, default="tlist"
    )
    adjoint: diffrax.AbstractAdjoint | None = struct.field(
        pytree_node=False, default=None
    )
    event: diffrax.Event | None = struct.field(pytree_node=False, default=None)
    throw: bool = struct.field(pytree_node=False, default=True)
    solver_state: Any = None
    controller_state: Any = None
    made_jump: bool | Array | None = None

    @classmethod
    def create(
        cls,
        progress_meter: bool = True,
        solver: str = "Tsit5",
        max_steps: int = 100_000,
        stepsize_controller: str = "PIDController",
        stepsize_controller_kwargs: dict[str, Any] | None = None,
    ) -> "SolverOptions":
        """Create options with the deprecated string-based interface."""
        warnings.warn(
            "SolverOptions.create() is deprecated; use SolverOptions with native "
            "objects, such as solver=diffrax.Tsit5() and "
            "progress_meter='default'.",
            FutureWarning,
            stacklevel=2,
        )
        return cls(
            solver=_diffrax_object(solver, diffrax.AbstractSolver),
            stepsize_controller=_diffrax_object(
                stepsize_controller,
                diffrax.AbstractStepSizeController,
                _legacy_controller_kwargs(
                    stepsize_controller, stepsize_controller_kwargs
                ),
            ),
            max_steps=max_steps,
            progress_meter="default" if progress_meter else None,
        )


class CustomProgressMeter(diffrax.TqdmProgressMeter):
    """JAXQuantum's default Diffrax progress bar."""

    @staticmethod
    def _init_bar() -> tqdm.tqdm:
        bar_format = (
            "{desc}: {percentage:3.0f}% |{bar}| "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        return tqdm.tqdm(
            total=100, bar_format=bar_format, unit="%", colour="MAGENTA", ascii="░▒█"
        )


def _resolve_saveat(
    options: SolverOptions,
    tlist: Array,
    saveat_tlist: Array | None,
) -> diffrax.SaveAt:
    if options.saveat is not None:
        if saveat_tlist is not None:
            raise ValueError(
                "Pass save times through saveat_tlist or SolverOptions.saveat, not both."
            )
        return options.saveat

    times = tlist if saveat_tlist is None else jnp.atleast_1d(saveat_tlist)
    return diffrax.SaveAt(t1=True) if len(times) == 0 else diffrax.SaveAt(ts=times)


def _resolve_progress_meter(
    progress_meter: (bool | Literal["default"] | diffrax.AbstractProgressMeter | None),
) -> diffrax.AbstractProgressMeter | None:
    if progress_meter is None:
        return None
    if isinstance(progress_meter, bool):
        return CustomProgressMeter() if progress_meter else None
    if isinstance(progress_meter, str):
        if progress_meter == "default":
            return CustomProgressMeter()
        raise ValueError(
            "progress_meter must be None, 'default', or a Diffrax progress meter."
        )
    if not isinstance(progress_meter, diffrax.AbstractProgressMeter):
        raise TypeError(
            "progress_meter must be a Diffrax AbstractProgressMeter instance."
        )
    return progress_meter


def _diffrax_object(
    name: str,
    expected_type: type,
    kwargs: dict[str, Any] | None = None,
):
    try:
        value = getattr(diffrax, name)(**(kwargs or {}))
    except AttributeError as error:
        raise ValueError(f"Unknown Diffrax type: {name!r}.") from error
    if not isinstance(value, expected_type):
        raise TypeError(f"diffrax.{name} is not a {expected_type.__name__}.")
    return value


def _legacy_controller_kwargs(
    name: str, kwargs: dict[str, Any] | None
) -> dict[str, Any]:
    if kwargs is not None:
        return kwargs
    return {"rtol": 1e-7, "atol": 1e-9} if name == "PIDController" else {}


def _uses_legacy_options(options: SolverOptions) -> bool:
    return (
        isinstance(options.solver, str)
        or isinstance(options.stepsize_controller, str)
        or options.stepsize_controller_kwargs is not None
        or isinstance(options.progress_meter, bool)
    )


def _resolve_solver(solver: diffrax.AbstractSolver | str) -> diffrax.AbstractSolver:
    if isinstance(solver, str):
        return _diffrax_object(solver, diffrax.AbstractSolver)
    if not isinstance(solver, diffrax.AbstractSolver):
        raise TypeError("solver must be a Diffrax AbstractSolver instance.")
    return solver


def _resolve_stepsize_controller(
    options: SolverOptions,
) -> diffrax.AbstractStepSizeController:
    controller = options.stepsize_controller
    if isinstance(controller, str):
        return _diffrax_object(
            controller,
            diffrax.AbstractStepSizeController,
            _legacy_controller_kwargs(controller, options.stepsize_controller_kwargs),
        )
    if options.stepsize_controller_kwargs is not None:
        raise ValueError(
            "Pass controller arguments when constructing stepsize_controller."
        )
    if not isinstance(controller, diffrax.AbstractStepSizeController):
        raise TypeError(
            "stepsize_controller must be a Diffrax AbstractStepSizeController instance."
        )
    return controller


def _resolve_dt0(options: SolverOptions, tlist: Array) -> float | Array | None:
    if isinstance(options.dt0, str):
        if options.dt0 != "tlist":
            raise ValueError("dt0 must be a number, None, or 'tlist'.")
        return tlist[1] - tlist[0]
    return options.dt0


def solve(
    f: Callable,
    y0: Array,
    tlist: Array,
    saveat_tlist: Array | None = None,
    args: Any = None,
    solver_options: SolverOptions | None = None,
) -> diffrax.Solution:
    """Solve an ODE using native Diffrax configuration from ``SolverOptions``."""
    options = SolverOptions() if solver_options is None else solver_options
    if _uses_legacy_options(options):
        warnings.warn(
            "String and boolean SolverOptions values are deprecated; use native "
            "objects such as solver=diffrax.Tsit5(), "
            "stepsize_controller=diffrax.PIDController(...), and "
            "progress_meter='default' or None.",
            FutureWarning,
            stacklevel=2,
        )
    kwargs = {
        "saveat": _resolve_saveat(options, tlist, saveat_tlist),
        "stepsize_controller": _resolve_stepsize_controller(options),
        "args": args,
        "max_steps": options.max_steps,
        "throw": options.throw,
    }
    optional = {
        "adjoint": options.adjoint,
        "event": options.event,
        "progress_meter": _resolve_progress_meter(options.progress_meter),
        "solver_state": options.solver_state,
        "controller_state": options.controller_state,
        "made_jump": options.made_jump,
    }
    kwargs.update(
        (name, value) for name, value in optional.items() if value is not None
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Complex dtype support in Diffrax",
            category=UserWarning,
        )
        return diffrax.diffeqsolve(
            diffrax.ODETerm(f),
            _resolve_solver(options.solver),
            t0=tlist[0],
            t1=tlist[-1],
            dt0=_resolve_dt0(options, tlist),
            y0=y0,
            **kwargs,
        )


def mesolve(
    H: Qarray | Callable[[float], Qarray],
    rho0: Qarray,
    tlist: Array,
    saveat_tlist: Array | None = None,
    c_ops: Qarray | None = None,
    solver_options: SolverOptions | None = None,
) -> Qarray:
    """Solve a Lindblad master equation and return the saved states.

    Args:
        H: Static Hamiltonian or callable ``H(t)``.
        rho0: Initial ket or density matrix.
        tlist: Integration interval; also the default save times.
        saveat_tlist: Save times. An empty array saves only the final state.
        c_ops: Collapse operators.
        solver_options: Native Diffrax configuration.

    Returns:
        Saved density matrices as a batched ``Qarray``.

    See Also:
        :func:`mesolve_result` returns the complete Diffrax solution.
    """
    solution = mesolve_result(
        H,
        rho0,
        tlist,
        saveat_tlist=saveat_tlist,
        c_ops=c_ops,
        solver_options=solver_options,
    )
    qdims = Qdims((rho0.space_dims, rho0.space_dims))
    return Qarray._from_impl(DenseImpl._make(solution.ys), qdims)


def mesolve_result(
    H: Qarray | Callable[[float], Qarray],
    rho0: Qarray,
    tlist: Array,
    saveat_tlist: Array | None = None,
    c_ops: Qarray | None = None,
    solver_options: SolverOptions | None = None,
) -> diffrax.Solution:
    """Solve a Lindblad master equation and return its Diffrax solution.

    Use this form for solver statistics, events, dense interpolation, custom
    ``SaveAt`` functions, or continuation state.
    """
    collapse_ops = c_ops if c_ops is not None else Qarray.from_list([])

    if len(collapse_ops) == 0 and rho0.qtype != Qtypes.oper:
        logging.warning(  # noqa: LOG015
            "Consider sesolve(): no collapse operators were provided and the "
            "initial state is not a density matrix."
        )

    rho_data = rho0.to_dm().to_dense()

    if robust_isscalar(H):
        H = H * identity_like(rho_data)

    if isinstance(H, Qarray):
        H_data = lambda t: H.data
    else:
        H_data = lambda t: H(t).data

    return _mesolve_result_data(
        H_data,
        rho_data.data,
        tlist,
        saveat_tlist,
        collapse_ops.data,
        solver_options=solver_options,
    )


def _mesolve_result_data(
    H: Callable[[float], Array],
    rho0: Array,
    tlist: Array,
    saveat_tlist: Array | None,
    c_ops: Array | None = None,
    solver_options: SolverOptions | None = None,
) -> diffrax.Solution:
    """Array-level master-equation implementation."""

    c_ops = c_ops if c_ops is not None else jnp.array([])

    # Shape inference: when c_ops contains batched operators (e.g. shape
    # (1, B, N, N)), the initial state ρ0 must be broadcast to (B, N, N) so
    # that the ODE RHS produces consistently shaped output.
    #
    # The output batch shape is the broadcast of:
    #   c_ops[0] batch dims  →  c_ops.shape[1:-2]  (outer batch index stripped)
    #   H batch dims         →  H(tlist[0]).shape[:-2]
    #   ρ0 batch dims        →  ρ0.shape[:-2]
    # This is a pure shape calculation — no array values are materialised.
    H0_shape = H(tlist[0]).shape
    if len(c_ops) == 0:
        batch_shape = jnp.broadcast_shapes(H0_shape[:-2], rho0.shape[:-2])
    else:
        # c_ops.shape[1:-2]: strip the outermost (c_op index) dim and the two
        # matrix dims to get the batch dims that will be broadcast into ρ.
        batch_shape = jnp.broadcast_shapes(
            c_ops.shape[1:-2], H0_shape[:-2], rho0.shape[:-2]
        )
    rho = jnp.broadcast_to(rho0, batch_shape + rho0.shape[-2:])

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

    return solve(
        f,
        rho,
        tlist,
        saveat_tlist,
        (c_ops, c_ops_dag),
        solver_options=solver_options,
    )


def sesolve(
    H: Qarray | Callable[[float], Qarray],
    rho0: Qarray,
    tlist: Array,
    saveat_tlist: Array | None = None,
    solver_options: SolverOptions | None = None,
) -> Qarray:
    """Solve a Schrödinger equation and return the saved states.

    Args:
        H: Static Hamiltonian or callable ``H(t)``.
        rho0: Initial ket.
        tlist: Integration interval; also the default save times.
        saveat_tlist: Save times. An empty array saves only the final state.
        solver_options: Native Diffrax configuration.

    Returns:
        Saved kets as a batched ``Qarray``.

    See Also:
        :func:`sesolve_result` returns the complete Diffrax solution.
    """
    solution = sesolve_result(
        H,
        rho0,
        tlist,
        saveat_tlist=saveat_tlist,
        solver_options=solver_options,
    )
    return Qarray._from_impl(DenseImpl._make(solution.ys), rho0.qdims)


def sesolve_result(
    H: Qarray | Callable[[float], Qarray],
    rho0: Qarray,
    tlist: Array,
    saveat_tlist: Array | None = None,
    solver_options: SolverOptions | None = None,
) -> diffrax.Solution:
    """Solve a Schrödinger equation and return its Diffrax solution.

    Use this form for solver statistics, events, dense interpolation, custom
    ``SaveAt`` functions, or continuation state.
    """
    if rho0.qtype == Qtypes.oper:
        raise ValueError("Use mesolve() for an initial density matrix.")

    state = rho0.to_ket().to_dense()

    if robust_isscalar(H):
        H = H * identity_like(state)

    if isinstance(H, Qarray):
        H_data = lambda t: H.data
    else:
        H_data = lambda t: H(t).data

    return _sesolve_result_data(
        H_data,
        state.data,
        tlist,
        saveat_tlist,
        solver_options=solver_options,
    )


def _sesolve_result_data(
    H: Callable[[float], Array],
    rho0: Array,
    tlist: Array,
    saveat_tlist: Array | None,
    solver_options: SolverOptions | None = None,
) -> diffrax.Solution:
    """Array-level Schrödinger-equation implementation."""

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

    batch_shape = jnp.broadcast_shapes(H(tlist[0]).shape[:-2], rho0.shape[:-1])
    state = jnp.broadcast_to(rho0, batch_shape + rho0.shape[-1:])

    return solve(
        f,
        state,
        tlist,
        saveat_tlist,
        solver_options=solver_options,
    )


# propagators


def propagator(
    H: Qarray | Callable[[float], Qarray],
    ts: float | Array,
    saveat_tlist: Array | None = None,
    solver_options: SolverOptions | None = None,
) -> Qarray:
    """Generate a propagator for a Hamiltonian.

    Args:
        H (Qarray or callable):
            A Qarray static Hamiltonian OR
            a function that takes a time argument and returns a Hamiltonian.
        ts (float or Array):
            A single time point or
            an Array of time points.
        saveat_tlist: Times at which to save the propagator.
        solver_options: Native Diffrax configuration for time-dependent input.

    Returns:
        The propagator at each saved time.
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
        results = sesolve(
            H,
            basis_states,
            ts,
            saveat_tlist=saveat_tlist,
            solver_options=solver_options,
        )
        # results.data is (T, M, M): T times, M evolved basis kets (batch), M
        # ket components. Transpose the last two axes so each time slice is a
        # propagator whose columns are the evolved basis states. No squeeze:
        # kets no longer carry a trailing singleton.
        propagators_data = results.data.mT
        return Qarray.create(propagators_data, dims=H_first.space_dims)

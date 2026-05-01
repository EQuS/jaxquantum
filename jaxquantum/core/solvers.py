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


from jaxquantum.core.qarray import Qarray, Qtypes, QarrayImplType, dag_data
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

    # Dispatch to the cuquantum solver path when H is cuquantum-backed.
    # The cuquantum path needs the Qarray (for OperatorTerm + dims), so it
    # branches *before* the .data extraction used by the dense / sparse path.
    H_test = H if isinstance(H, Qarray) else (H(0.0) if callable(H) else None)
    if isinstance(H_test, Qarray) and H_test.impl_type == QarrayImplType.CUQUANTUM:
        return _mesolve_cuquantum(H, rho0, tlist, saveat_tlist, c_ops, solver_options)

    ρ0 = rho0.to_dm().to_dense()

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

    # Dispatch to the cuquantum solver path when H is cuquantum-backed.
    H_test = H if isinstance(H, Qarray) else (H(0.0) if callable(H) else None)
    if isinstance(H_test, Qarray) and H_test.impl_type == QarrayImplType.CUQUANTUM:
        return _sesolve_cuquantum(H, ψ, tlist, saveat_tlist, solver_options)

    ψ = ψ.to_ket().to_dense()

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


# ---------------------------------------------------------------------------
# cuQuantum solver path
# ---------------------------------------------------------------------------

# def _build_cuquantum_dissipator_terms(L_qarray):
#     """Generate Ls_term entries that encode ``D[L]ρ`` in cuquantum form.

#     Mirrors the dissipator recipe used in
#     ``local/lattice_2d_operator_action.py``:

#       D[L]ρ = LρL† − ½ L†L ρ − ½ ρ L†L

#     Yields ``(matrices, modes, duals, coeff)`` tuples that should be
#     appended to a single ``Ls_term`` ``OperatorTerm``.

#     Args:
#         L_qarray: Qarray[CuquantumImpl] holding the collapse operator.

#     Yields:
#         Successive term tuples ready to be passed to OperatorTerm.append
#         (after wrapping each matrix in an ElementaryOperator).
#     """
#     L_data = L_qarray._impl._data            # OperatorTerm
#     LdagL_data = (L_qarray.dag() @ L_qarray)._impl._data  # OperatorTerm

#     def _iter_products(op_term):
#         """Yield ``(matrices, modes, coeff_scalar)`` for each product in op_term."""
#         for op_prod, modes, coeff_arr in zip(
#             op_term.op_prods, op_term.modes, op_term.coeffs
#         ):
#             mats = tuple(jnp.squeeze(elem.data, axis=0) for elem in op_prod)
#             yield mats, tuple(modes), coeff_arr[0] if coeff_arr.ndim else coeff_arr

#     # L ρ L† = sum over (a, b) of cₐ * conj(c_b) * (L_a) ρ (L_b)†.
#     # In cuQuantum's elementary product convention, ``[X, Y]`` on the same mode
#     # acts as the matrix ``X @ Y`` — i.e. operators are composed left-to-right
#     # in the tuple — so to encode ``L_a ρ L_b†`` we put ``L_a``'s factors with
#     # ``dual=False`` then ``L_b``'s daggered factors (in reverse order) with
#     # ``dual=True``.
#     L_products = list(_iter_products(L_data))
#     for (mats_a, modes_a, coeff_a) in L_products:
#         for (mats_b, modes_b, coeff_b) in L_products:
#             mats_b_dag_rev = tuple(_matrix_dag(m) for m in reversed(mats_b))
#             modes_b_rev = tuple(reversed(modes_b))
#             yield (
#                 tuple(mats_a) + mats_b_dag_rev,
#                 tuple(modes_a) + modes_b_rev,
#                 (False,) * len(mats_a) + (True,) * len(mats_b_dag_rev),
#                 coeff_a * jnp.conj(coeff_b),
#             )

#     # -½ L†L ρ : L†L applied on the left side (all duals=False).
#     # -½ ρ L†L : L†L applied on the right side (all duals=True).
#     for (mats, modes, coeff) in _iter_products(LdagL_data):
#         yield (mats, modes, (False,) * len(mats), -0.5 * coeff)
#         yield (mats, modes, (True,) * len(mats), -0.5 * coeff)


def _matrix_dag(matrix):
    return jnp.conj(jnp.swapaxes(matrix, -1, -2))


def _mesolve_cuquantum(
    H,
    rho0: Qarray,
    tlist: Array,
    saveat_tlist: Array,
    c_ops: Qarray,
    solver_options: Optional[SolverOptions] = None,
) -> Qarray:
    """Master-equation solver dispatched to cuQuantum's ``operator_action``.

    The Hamiltonian and (optional) collapse operators must be cuquantum-backed
    Qarrays; the state ``ρ0`` is densified before being handed to diffrax.
    Inside the RHS we build a fresh ``Operator`` each step, append the
    Hamiltonian commutator and the dissipator superterm, and call
    ``operator_action`` — exactly mirroring
    ``local/lattice_2d_operator_action.py``.
    """
    from jaxquantum.core.cuquantum_impl import CuquantumImpl, _cuqnt_dag
    from cuquantum.densitymat.jax import (
        ElementaryOperator,
        Operator,
        operator_action,
    )

    from jaxquantum.utils.cuquantum_util import OperatorTerm 


    # --- snapshot the static metadata we need outside the RHS ---
    H_test = H if isinstance(H, Qarray) else H(0.0)
    space_dims = tuple(int(d) for d in H_test.space_dims)

    rho0_dense = rho0.to_dm().to_dense()
    dims_meta = rho0_dense.dims
    rho0_arr = rho0_dense.data

    # Normalise c_ops to a Python list of cuquantum-backed Qarrays.
    # ``Qarray.from_list`` does not preserve the cuquantum impl (no batched
    # ``OperatorTerm`` exists), so cuquantum users should pass c_ops as a
    # list directly:  ``mesolve(..., c_ops=[L1, L2])``.  We also accept an
    # empty Qarray placeholder (which the public ``mesolve`` substitutes
    # when c_ops is None).
    if c_ops is None or (hasattr(c_ops, "__len__") and len(c_ops) == 0):
        c_ops_list: list = []
    elif isinstance(c_ops, (list, tuple)):
        c_ops_list = list(c_ops)
    elif isinstance(c_ops, Qarray) and c_ops.impl_type == QarrayImplType.CUQUANTUM:
        c_ops_list = [c_ops[k] for k in range(len(c_ops))]
    else:
        raise TypeError(
            "cuquantum mesolve requires c_ops to be a Python list of "
            "cuquantum-backed Qarrays (Qarray.from_list densifies them)"
        )

    # Pre-build the dissipator OperatorTerm — it is static across t.
    Ls_term: Optional["OperatorTerm"] = None
    if c_ops_list:
        Ls_term = OperatorTerm(space_dims)
        for c_op in c_ops_list:
            if not isinstance(c_op, Qarray) or c_op.impl_type != QarrayImplType.CUQUANTUM:
                raise TypeError(
                    "_mesolve_cuquantum requires every c_op to be a "
                    "cuquantum-backed Qarray"
                )
            _l = c_op._impl._data
            _ld = _cuqnt_dag(c_op._impl._data)
            
            if len(_l.op_prods) > 1:
                raise ValueError("currently, only one term is allowed in the collapse operator") 

            l = _l[0]
            ld =_ld[0]

            # l = ElementaryOperator((_l[0][0].data * _l.coeffs[0]).astype(_l[0][0].dtype))
            # ld = ElementaryOperator(_ld[0][0].data * _ld.coeffs[0])

            modes = c_op._impl._data.modes[0]
            coeff = 1.0 * _l.coeffs[0] * jnp.conj(_l.coeffs[0])
            
            Ls_term.append([*l, *ld], modes=modes+modes, duals=[False, True], coeff=1.0 * coeff)
            Ls_term.append([*ld, *l], modes=modes+modes, duals=[True, True], coeff=-0.5 * coeff)
            Ls_term.append([*l, *ld], modes=modes+modes, duals=[False, False], coeff=-0.5 * coeff)


    # The Hamiltonian's OperatorTerm depends on t; build it inside the RHS.
    if isinstance(H, Qarray):
        H_qarray_fn = lambda t: H
    else:
        H_qarray_fn = lambda t: H(t)

    def f(t, rho, args):
        H_term = H_qarray_fn(t)._impl.to_operator_term()
        liouvillian = Operator(space_dims)
        liouvillian.append(H_term, dual=False, coeff=-1j)
        liouvillian.append(H_term, dual=True,  coeff= 1j)
        if Ls_term is not None:
            liouvillian.append(Ls_term, dual=False, coeff=1.0)
        rho_shape = rho.shape
        rho_dot = operator_action(
            liouvillian, rho.reshape(*space_dims, *space_dims)
        )
        return rho_dot.reshape(rho_shape)

    try:
        sol = solve(
            f, rho0_arr, tlist, saveat_tlist, args=None,
            solver_options=solver_options,
        )
    except ValueError as e:
        if "operator terms" in str(e):
            raise ValueError(
                "please make sure the Hamiltonian and collapse operators are of the same dtype"
            )
        else:
            raise e
    return jnp2jqt(sol.ys, dims=dims_meta)


def _sesolve_cuquantum(
    H,
    psi0: Qarray,
    tlist: Array,
    saveat_tlist: Array,
    solver_options: Optional[SolverOptions] = None,
) -> Qarray:
    """Schrödinger-equation solver dispatched to cuQuantum's ``operator_action``.

    Like ``_mesolve_cuquantum`` but evolves a ket: ``i ψ̇ = H ψ``.  No
    dissipator is involved; the RHS reduces to a single ``-1j * H * ψ``
    application.
    """
    from cuquantum.densitymat.jax import Operator, operator_action

    H_test = H if isinstance(H, Qarray) else H(0.0)
    space_dims = tuple(int(d) for d in H_test.space_dims)

    psi0_ket = psi0.to_ket().to_dense()
    dims_meta = psi0_ket.dims
    psi0_arr = psi0_ket.data

    if isinstance(H, Qarray):
        H_qarray_fn = lambda t: H
    else:
        H_qarray_fn = lambda t: H(t)

    def f(t, psi, args):
        H_term = H_qarray_fn(t)._impl.to_operator_term()
        op = Operator(space_dims)
        op.append(H_term, dual=False, coeff=-1j)
        psi_shape = psi.shape
        # ψ has shape (*batch, full_dim, 1); collapse trailing 1 for the
        # state grid then restore it after the action.
        flat = psi.reshape(*psi.shape[:-2], *space_dims)
        psi_dot = operator_action(op, flat)
        return psi_dot.reshape(psi_shape)

    sol = solve(
        f, psi0_arr, tlist, saveat_tlist, args=None,
        solver_options=solver_options,
    )
    return jnp2jqt(sol.ys, dims=dims_meta)


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

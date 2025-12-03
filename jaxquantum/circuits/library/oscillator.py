"""Oscillator gates."""

from jaxquantum.core.operators import (displace, basis, destroy, create, num,
                                       identity)
from jaxquantum.circuits.gates import Gate
from jax.scipy.special import factorial, gammaln
import jax.numpy as jnp
from jaxquantum import Qarray
from jaxquantum.utils import hermgauss
from functools import partial
from jax import jit
import jax

def diag_expm(diag_matrix):
    """Computes expm of a diagonal matrix efficiently (O(N) instead of O(N^3))."""
    # Extract diagonal, exponentiate elements, put back on diagonal
    return jnp.diag(jnp.exp(jnp.diagonal(diag_matrix)))



def D(N, alpha, ts=None, c_ops=None):
    """Displacement gate.

    Args:
        N: Hilbert space dimension.
        alpha: Displacement amplitude.
        ts: Optional time array for hamiltonian simulation.
        c_ops: Optional collapse operators.

    Returns:
        Displacement gate.
    """
    gen_Ht = None
    if ts is not None:
        delta_t = ts[-1] - ts[0]
        amp = 1j * alpha / delta_t
        a = destroy(N)
        gen_Ht = lambda params: (lambda t: jnp.conj(amp) * a + amp * a.dag())

    return Gate.create(
        N,
        name="D",
        params={"alpha": alpha},
        gen_U=lambda params: displace(N, params["alpha"]),
        gen_Ht=gen_Ht,
        ts=ts,
        gen_c_ops=lambda params: Qarray.from_list([]) if c_ops is None else c_ops,
        num_modes=1,
    )


def CD(N, beta, ts=None):
    """Conditional displacement gate.

    Args:
        N: Hilbert space dimension.
        beta: Conditional displacement amplitude.
        ts: Optional time sequence for hamiltonian simulation.

    Returns:
        Conditional displacement gate.
    """
    g = basis(2, 0)
    e = basis(2, 1)

    gg = g @ g.dag()
    ee = e @ e.dag()

    gen_Ht = None
    if ts is not None:
        delta_t = ts[-1] - ts[0]
        amp = 1j * beta / delta_t / 2
        a = destroy(N)
        gen_Ht = lambda params: lambda t: (
            gg
            ^ (jnp.conj(amp) * a + amp * a.dag()) + ee
            ^ (jnp.conj(-amp) * a + (-amp) * a.dag())
        )

    return Gate.create(
        [2, N],
        name="CD",
        params={"beta": beta},
        gen_U=lambda params: (gg ^ displace(N, params["beta"] / 2))
        + (ee ^ displace(N, -params["beta"] / 2)),
        gen_Ht=gen_Ht,
        ts=ts,
        num_modes=2,
    )


def ECD(N, beta, ts=None):
    """Echoed conditional displacement gate.

    Args:
        N: Hilbert space dimension.
        beta: Conditional displacement amplitude.
        ts: Optional time sequence for hamiltonian simulation.

    Returns:
        Echoed conditional displacement gate.
    """
    g = basis(2, 0)
    e = basis(2, 1)

    eg = e @ g.dag()
    ge = g @ e.dag()

    # gen_Ht = None
    # if ts is not None:
    #     delta_t = ts[-1] - ts[0]
    #     amp = 1j * beta / delta_t / 2
    #     a = destroy(N)
    #     gen_Ht = lambda params: lambda t: (
    #         eg
    #         ^ (jnp.conj(amp) * a + amp * a.dag()) + ge
    #         ^ (jnp.conj(-amp) * a + (-amp) * a.dag())
    #     )

    return Gate.create(
        [2, N],
        name="ECD",
        params={"beta": beta},
        gen_U=lambda params: (eg ^ displace(N, params["beta"] / 2))
        + (ge ^ displace(N, -params["beta"] / 2)),
        gen_Ht=None,
        ts=ts,
        num_modes=2,
    )

def CR(N, theta):
    """Conditional rotation gate.

    Args:
        N: Hilbert space dimension.
        theta: Conditional rotation angle.

    Returns:
        Conditional rotation gate.
    """
    g = basis(2, 0)
    e = basis(2, 1)

    gg = g @ g.dag()
    ee = e @ e.dag()


    return Gate.create(
        [2, N],
        name="CR",
        params={"theta": theta},
        gen_U=lambda params: (gg ^ (-1.j*theta/2*create(N)@destroy(N)).expm())
        + (ee ^ (1.j*theta/2*create(N)@destroy(N)).expm()),
        num_modes=2,
    )


# --- 2. Optimized Kernels (Using diag_expm) ---

@partial(jax.jit, static_argnames=["N", "max_l"])
def _Amp_Damp_Kraus_Map_JIT(N, err_prob, max_l):
    n_op = num(N).data
    a_op = destroy(N).data
    
    log_term = jnp.log(jnp.sqrt(1.0 - err_prob))
    # FIX: Use diag_expm
    middle_op = diag_expm(n_op * log_term)

    def compute_op(l):
        prefactor = jnp.sqrt(jnp.power(err_prob, l) / jnp.exp(gammaln(l + 1)))
        a_pow_l = jnp.linalg.matrix_power(a_op, l)
        return prefactor * (middle_op @ a_pow_l)

    ls = jnp.arange(max_l + 1)
    return jax.vmap(compute_op)(ls)

def Amp_Damp(N, err_prob, max_l):
    kmap = lambda params: Qarray.create(
        _Amp_Damp_Kraus_Map_JIT(params["N"], params["err_prob"], params["max_l"]),
        dims=[[N], [N]],
        bdims=(params["max_l"] + 1,)
    )
    return Gate.create(
        N,
        name="Amp_Damp",
        params={"err_prob": err_prob, "max_l": max_l, "N": N},
        gen_KM=kmap,
        num_modes=1,
    )


@partial(jax.jit, static_argnames=["N", "max_l"])
def _Amp_Gain_Kraus_Map_JIT(N, err_prob, max_l):
    n_op = num(N).data
    adag_op = create(N).data
    
    log_term = jnp.log(jnp.sqrt(1.0 - err_prob))
    # FIX: Use diag_expm
    middle_op = diag_expm(n_op * log_term)

    def compute_op(l):
        prefactor = jnp.sqrt(jnp.power(err_prob, l) / jnp.exp(gammaln(l + 1)))
        adag_pow_l = jnp.linalg.matrix_power(adag_op, l)
        return prefactor * (adag_pow_l @ middle_op)

    ls = jnp.arange(max_l + 1)
    return jax.vmap(compute_op)(ls)

def Amp_Gain(N, err_prob, max_l):
    kmap = lambda params: Qarray.create(
        _Amp_Gain_Kraus_Map_JIT(params["N"], params["err_prob"], params["max_l"]),
        dims=[[N], [N]],
        bdims=(params["max_l"] + 1,)
    )
    return Gate.create(
        N,
        name="Amp_Gain",
        params={"err_prob": err_prob, "max_l": max_l, "N": N},
        gen_KM=kmap,
        num_modes=1,
    )


@partial(jax.jit, static_argnames=["N", "max_l"])
def _Thermal_Ch_Kraus_Map_JIT(N, err_prob, n_bar, max_l):
    a_op = destroy(N).data
    adag_op = create(N).data
    n_op = num(N).data
    
    a_powers = jnp.stack([jnp.linalg.matrix_power(a_op, i) for i in range(max_l + 1)])
    adag_powers = jnp.stack([jnp.linalg.matrix_power(adag_op, i) for i in range(max_l + 1)])
    
    log_term = jnp.log(jnp.sqrt(1.0 - err_prob))
    # FIX: Use diag_expm
    middle_op = diag_expm(n_op * log_term)

    def compute_single_op(idx):
        l = idx // (max_l + 1)
        k = idx % (max_l + 1)
        
        fact_l = jnp.exp(gammaln(l + 1))
        fact_k = jnp.exp(gammaln(k + 1))
        
        term_k = jnp.power(err_prob * (1.0 + n_bar), k)
        term_l = jnp.power(err_prob * n_bar, l)
        
        prefactor = jnp.sqrt( (term_k * term_l) / (fact_k * fact_l) )
        op_k = a_powers[k]
        op_l = adag_powers[l]
        
        return prefactor * (middle_op @ op_k @ op_l)

    indices = jnp.arange((max_l + 1)**2)
    return jax.vmap(compute_single_op)(indices)

def Thermal_Ch(N, err_prob, n_bar, max_l):
    kmap = lambda params: Qarray.create(
        _Thermal_Ch_Kraus_Map_JIT(params["N"], params["err_prob"], params["n_bar"], params["max_l"]),
        dims=[[N], [N]],
        bdims=((params["max_l"] + 1)**2,)
    )
    return Gate.create(
        N,
        name="Thermal_Ch",
        params={"err_prob": err_prob, "n_bar": n_bar, "max_l": max_l, "N": N},
        gen_KM=kmap,
        num_modes=1,
    )


@partial(jax.jit, static_argnames=["N", "max_l"])
def _Dephasing_Ch_Kraus_Map_JIT(N, ws, phis, max_l):
    n_op = num(N).data
    def compute_op(w, phi):
        # FIX: Use diag_expm
        op = diag_expm(1.0j * phi * n_op)
        return jnp.sqrt(w) * op
    return jax.vmap(compute_op)(ws, phis)

def Dephasing_Ch(N, err_prob, max_l):
    xs, ws_raw = hermgauss(max_l)
    phis = jnp.sqrt(2*err_prob)*xs
    ws = 1/jnp.sqrt(jnp.pi)*ws_raw

    kmap = lambda params: Qarray.create(
        _Dephasing_Ch_Kraus_Map_JIT(params["N"], ws, phis, params["max_l"]),
        dims=[[N], [N]],
        bdims=(params["max_l"],)
    )
    return Gate.create(
        N,
        name="Dephasing_Ch",
        params={"err_prob": err_prob, "max_l": max_l, "N": N},
        gen_KM=kmap,
        num_modes=1,
    )


def selfKerr(N, K):
    a = destroy(N)
    return Gate.create(
        N,
        name="selfKerr",
        params={"Kerr": K},
        gen_U=lambda params: (-1.0j * K / 2 * (a.dag() @ a.dag() @ a @ a)).expm(),
        num_modes=1,
    )


@partial(jax.jit, static_argnames=["N", "max_l"])
def _Dephasing_Reset_Kraus_Map_JIT(N, p, t_rst, chi, max_l):
    g = basis(2, 0).data
    e = basis(2, 1).data
    gg = g @ jnp.conj(g.T)
    ee = e @ jnp.conj(e.T)
    ge = g @ jnp.conj(e.T)
    
    n_op = num(N).data
    I_N = jnp.eye(N)
    
    ls_all = jnp.arange(2, max_l)
    norm_terms = -(jnp.log(p) * jnp.power(p, (ls_all - 2) / (max_l - 1))) / (max_l - 1)
    normalization_factor = (1 - p) / jnp.sum(norm_terms)

    def compute_op(l):
        def branch_0(_):
            return jnp.kron(gg, I_N)

        def branch_1(_):
            # FIX: Use diag_expm
            op_osc = diag_expm(-1.0j * chi * t_rst * n_op)
            return jnp.sqrt(p) * jnp.kron(ee, op_osc)

        def branch_rest(_):
            term_val = -(jnp.log(p) * jnp.power(p, (l - 2) / (max_l - 1))) / (max_l - 1)
            prefactor = jnp.sqrt(term_val) * jnp.sqrt(normalization_factor)
            
            exponent = -1.0j * chi * t_rst * (l - 2) / (max_l - 1)
            # FIX: Use diag_expm
            op_osc = diag_expm(exponent * n_op)
            return prefactor * jnp.kron(ge, op_osc)

        return jax.lax.cond(
            l == 0, 
            branch_0,
            lambda _: jax.lax.cond(l == 1, branch_1, branch_rest, operand=None),
            operand=None
        )

    ls = jnp.arange(max_l)
    return jax.vmap(compute_op)(ls)

def Dephasing_Reset(N, p, t_rst, chi, max_l):
    kmap = lambda params: Qarray.create(
        _Dephasing_Reset_Kraus_Map_JIT(
            params["N"], params["p"], params["t_rst"], params["chi"], params["max_l"]
        ),
        dims=[[2, N], [2, N]],
        bdims=(params["max_l"],)
    )
    
    return Gate.create(
        [2, N],
        name="Dephasing_Reset",
        params={"p": p, "t_rst": t_rst, "chi": chi, "max_l": max_l, "N": N},
        gen_KM=kmap,
        num_modes=2,
    )

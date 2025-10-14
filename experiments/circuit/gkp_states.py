import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import gammaln
from functools import partial
from jax.scipy.linalg import eigh_tridiagonal


import jaxquantum as jqt


def log_hermite_jax(n, xs):
    float_dtype = xs.dtype
    
    def n_is_zero_branch(_xs):
        return jnp.ones_like(_xs, dtype=jnp.int32), jnp.zeros_like(_xs, dtype=float_dtype)

    def n_is_not_zero_branch(_xs):
        
        zeros = hermroots(n)
        xs_minus_zeros = _xs[:, None] - zeros[None, :]
        num_negative_factors = (xs_minus_zeros < 0).sum(axis=1)
        signs = (1 - 2 * (num_negative_factors % 2))
        abs_xs_minus_zeros = jnp.abs(xs_minus_zeros)
        log_of_2 = jnp.log(jnp.array(2.0, dtype=float_dtype))
        log_abs = n * log_of_2 + jnp.log(abs_xs_minus_zeros).sum(axis=1)
        
        return signs.astype(jnp.int32), log_abs.astype(float_dtype)

    return jax.lax.cond(n == 0, n_is_zero_branch, n_is_not_zero_branch, operand=xs)


def log_ho_basis_prefactors_jax(n, xs):
    return (-n / 2.0 * jnp.log(2.0)
            - 0.5 * gammaln(n + 1.0)
            - 0.25 * jnp.log(jnp.pi)
            - xs**2 / 2.0)


def log_ho_basis_function_jax(n, xs):
    signs_herm, log_abs_herm = log_hermite_jax(n, xs)
    log_prefactors = log_ho_basis_prefactors_jax(n, xs)
    return signs_herm, log_abs_herm + log_prefactors


@partial(jit, static_argnames=['n'])
def _calculate_single_amplitude_jitted(n, xs, delta):
    logenv = -delta**2 * n
    sgns, logamps = log_ho_basis_function_jax(n, xs)
    amps = sgns * jnp.exp(logenv + logamps)
    return amps.sum()


def finite_gkp(mu, delta, dim):
    """
    Constructs a finite-energy GKP state in the Fock basis.
    """
    l = 2.0 * jnp.sqrt(jnp.pi)
    xs = l * (jnp.arange(-5000, 5001, dtype=jnp.float32) + mu / 2.0)

    amplitudes = []
    for n in range(dim):
        amp = _calculate_single_amplitude_jitted(n, xs, delta)
        amplitudes.append(amp)
    
    amplitudes = jnp.array(amplitudes)
    
    norm = jnp.linalg.norm(amplitudes)
    normalized_amplitudes = amplitudes / (norm + 1e-10)
    
    return jqt.Qarray.create(normalized_amplitudes)


def hermroots(n):
    """
    Computes the roots of the nth-degree physicist's Hermite polynomial H_n(x).
    """
    return eigh_tridiagonal(jnp.zeros(n), jnp.sqrt(jnp.arange(1, n) / 2.0), eigvals_only=True)

    
import jax.numpy as jnp
from jax import vmap, config
from jax.scipy.special import factorial
import jax



def wigner(psi, xvec, yvec, method="clenshaw", g=2):
    """Wigner function for a state vector or density matrix at points
    `xvec + i * yvec`.

    Parameters
    ----------

    state : Qarray
        A state vector or density matrix.

    xvec : array_like
        x-coordinates at which to calculate the Wigner function.

    yvec : array_like
        y-coordinates at which to calculate the Wigner function.

    g : float, default: 2
        Scaling factor for `a = 0.5 * g * (x + iy)`, default `g = 2`.
        The value of `g` is related to the value of `hbar` in the commutation
        relation `[x, y] = i * hbar` via `hbar=2/g^2`.

    method : string {'clenshaw', 'iterative', 'laguerre', 'fft'}, default: 'clenshaw'
        Only 'clenshaw' is currently supported.
        Select method 'clenshaw' 'iterative', 'laguerre', or 'fft', where 'clenshaw'
        and 'iterative' use an iterative method to evaluate the Wigner functions for density
        matrices :math:`|m><n|`, while 'laguerre' uses the Laguerre polynomials
        in scipy for the same task. The 'fft' method evaluates the Fourier
        transform of the density matrix. The 'iterative' method is default, and
        in general recommended, but the 'laguerre' method is more efficient for
        very sparse density matrices (e.g., superpositions of Fock states in a
        large Hilbert space). The 'clenshaw' method is the preferred method for
        dealing with density matrices that have a large number of excitations
        (>~50). 'clenshaw' is a fast and numerically stable method.

    Returns
    -------

    W : array
        Values representing the Wigner function calculated over the specified
        range [xvec,yvec].


    References
    ----------

    Ulf Leonhardt,
    Measuring the Quantum State of Light, (Cambridge University Press, 1997)

    """

    if not (psi.is_vec() or psi.is_dm()):
        raise TypeError("Input state is not a valid operator.")

    if method == "fft":
        raise NotImplementedError("Only the 'clenshaw' method is implemented.")

    if method == "iterative":
        raise NotImplementedError("Only the 'clenshaw' method is implemented.")

    elif method == "laguerre":
        raise NotImplementedError("Only the 'clenshaw' method is implemented.")

    elif method == "clenshaw":
        rho = psi.to_dm()
        rho = rho.data

        vmapped_wigner_clenshaw = [_wigner_clenshaw]

        for _ in rho.shape[:-2]:
            vmapped_wigner_clenshaw.append(
                vmap(
                    vmapped_wigner_clenshaw[-1],
                    in_axes=(0, None, None, None),
                    out_axes=0,
                )
            )
        return vmapped_wigner_clenshaw[-1](rho, xvec, yvec, g)

    else:
        raise TypeError("method must be either 'iterative', 'laguerre', or 'fft'.")


def _wigner_clenshaw(rho, xvec, yvec, g):
    r"""
    Using Clenshaw summation - numerically stable and efficient
    iterative algorithm to evaluate polynomial series.

    The Wigner function is calculated as
    :math:`W = e^(-0.5*x^2)/pi * \sum_{L} c_L (2x)^L / \sqrt(L!)` where
    :math:`c_L = \sum_n \rho_{n,L+n} LL_n^L` where
    :math:`LL_n^L = (-1)^n \sqrt(L!n!/(L+n)!) LaguerreL[n,L,x]`
    Heavily inspired by Qutip and Dynamiqs
    https://github.com/dynamiqs/dynamiqs
    https://github.com/qutip/qutip
    """

    M = jnp.prod(rho.shape[0])
    X, Y = jnp.meshgrid(xvec, yvec)
    A = 0.5 * g * (X + 1.0j * Y)
    B = jnp.abs(2*A)

    B *= B

    w0 = (2 * rho[0, -1]) * jnp.ones_like(A)

    # calculation of \sum_{L} c_L (2x)^L / \sqrt(L!)
    # using Horner's method

    rho = rho * (2 * jnp.ones((M, M)) - jnp.diag(jnp.ones(M)))
    def loop(i: int, w: jax.Array) -> jax.Array:
        i = M - 2 - i
        w = w * (2 * A * (i + 1) ** (-0.5))
        return w + _wig_laguerre_val(i, B, rho, M)

    w = jax.lax.fori_loop(0, M - 1, loop, w0)

    return w.real * jnp.exp(-B * 0.5) * (g * g * 0.5 / jnp.pi)

def _extract_diag_element(rho: jnp.array, L: int, n:int):
    """"
    Extract element at index n from diagonal L of matrix rho.
    Heavily inspired from https://github.com/dynamiqs/dynamiqs
    """
    N = rho.shape[0]
    n = jax.lax.select(n < 0, N - jnp.abs(L) - jnp.abs(n), n)
    row = jnp.maximum(-L, 0) + n
    col = jnp.maximum(L, 0) + n
    return rho[row, col]

def _wig_laguerre_val(L, x, rho, N):
    r"""
    Evaluate Laguerre polynomials.
    Implementation in Jax from https://github.com/dynamiqs/dynamiqs
    """

    def len_c_1():
        return _extract_diag_element(rho, L, 0) * jnp.ones_like(x)

    def len_c_2():
        c0 = _extract_diag_element(rho, L, 0)
        c1 = _extract_diag_element(rho, L, 1)
        return (c0 - c1 * (L + 1 - x) * (L + 1) ** (-0.5)) * jnp.ones_like(x)

    def len_c_other():
        cm2 = _extract_diag_element(rho, L, -2)
        cm1 = _extract_diag_element(rho, L, -1)
        y0 = cm2 * jnp.ones_like(x)
        y1 = cm1 * jnp.ones_like(x)

        def loop(j: int, args: tuple[jax.Array, jax.Array]) -> tuple[
            jax.Array, jax.Array]:
            def body() -> tuple[jax.Array, jax.Array]:
                k = N + 1 - L - j
                y0, y1 = args
                ckm1 = _extract_diag_element(rho, L, -j)
                y0, y1 = (
                    ckm1 - y1 * (k * (L + k) / ((L + k + 1) * (k + 1))) ** 0.5,
                    y0 - y1 * (L + 2 * k - x + 1) * (
                                (L + k + 1) * (k + 1)) ** -0.5,
                )

                return y0, y1

            return jax.lax.cond(j >= N + 1 - L, lambda: args, body)

        y0, y1 = jax.lax.fori_loop(3, N + 1, loop, (y0, y1))

        return y0 - y1 * (L + 1 - x) * (L + 1) ** (-0.5)


    return jax.lax.cond(N - L == 1, len_c_1, lambda: jax.lax.cond(N - L == 2,
                                                               len_c_2,
                                                       len_c_other))


def qfunc(psi, xvec, yvec, g=2):
    r"""
    Husimi-Q function of a given state vector or density matrix at phase-space
    points ``0.5 * g * (xvec + i*yvec)``.

    Parameters
    ----------
    state : Qarray
        A state vector or density matrix. This cannot have tensor-product
        structure.

    xvec, yvec : array_like
        x- and y-coordinates at which to calculate the Husimi-Q function.

    g : float, default: 2
        Scaling factor for ``a = 0.5 * g * (x + iy)``.  The value of `g` is
        related to the value of :math:`\hbar` in the commutation relation
        :math:`[x,\,y] = i\hbar` via :math:`\hbar=2/g^2`.

    Returns
    -------
    jnp.ndarray
        Values representing the Husimi-Q function calculated over the specified
        range ``[xvec, yvec]``.

    """

    alpha_grid, prefactor = _qfunc_coherent_grid(xvec, yvec, g)

    if psi.is_vec():
        psi = psi.to_ket()

        def _compute_qfunc(psi, alpha_grid, prefactor, g):
            out = _qfunc_iterative_single(psi, alpha_grid, prefactor, g)
            out /= jnp.pi
            return out
    else:

        def _compute_qfunc(psi, alpha_grid, prefactor, g):
            values, vectors = jnp.linalg.eigh(psi)
            vectors = vectors.T
            out = values[0] * _qfunc_iterative_single(
                vectors[0], alpha_grid, prefactor, g
            )
            for value, vector in zip(values[1:], vectors[1:]):
                out += value * _qfunc_iterative_single(vector, alpha_grid, prefactor, g)
            out /= jnp.pi

            return out

    psi = psi.data

    vmapped_compute_qfunc = [_compute_qfunc]

    for _ in psi.shape[:-2]:
        vmapped_compute_qfunc.append(
            vmap(
                vmapped_compute_qfunc[-1],
                in_axes=(0, None, None, None),
                out_axes=0,
            )
        )
    return vmapped_compute_qfunc[-1](psi, alpha_grid, prefactor, g)


def _qfunc_iterative_single(
    vector,
    grid,
    prefactor,
    g,
):
    r"""
    Get the Q function (without the :math:`\pi` scaling factor) of a single
    state vector, using the iterative algorithm which recomputes the powers of
    the coherent-state matrix.
    """
    vector = vector.squeeze()
    ns = jnp.arange(vector.shape[-1])
    out = jnp.polyval(
        (vector / jnp.sqrt(factorial(ns)))[::-1],
        grid,
    )
    out *= prefactor
    return jnp.abs(out) ** 2


def _qfunc_coherent_grid(xvec, yvec, g):
    x, y = jnp.meshgrid(0.5 * g * xvec, 0.5 * g * yvec)
    grid = jnp.empty(x.shape, dtype=jnp.complex128)
    grid += x
    # We produce the adjoint of the coherent states to save an operation
    # later when computing dot products, hence the negative imaginary part.
    grid += -y * 1.0j
    prefactor = jnp.exp(-0.5 * (x * x + y * y)).astype(jnp.complex128)
    return grid, prefactor

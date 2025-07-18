import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import factorial


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


def _wigner_clenshaw(rho, xvec, yvec, g=jnp.sqrt(2)):
    r"""
    Using Clenshaw summation - numerically stable and efficient
    iterative algorithm to evaluate polynomial series.

    The Wigner function is calculated as
    :math:`W = e^(-0.5*x^2)/pi * \sum_{L} c_L (2x)^L / \sqrt(L!)` where
    :math:`c_L = \sum_n \rho_{n,L+n} LL_n^L` where
    :math:`LL_n^L = (-1)^n \sqrt(L!n!/(L+n)!) LaguerreL[n,L,x]`
    """

    M = jnp.prod(rho.shape[0])
    X, Y = jnp.meshgrid(xvec, yvec)
    # A = 0.5 * g * (X + 1.0j * Y)
    A2 = g * (X + 1.0j * Y)  # this is A2 = 2*A

    B = jnp.abs(A2)

    B *= B

    w0 = (2 * rho[0, -1]) * jnp.ones_like(A2)

    L = M - 1
    # calculation of \sum_{L} c_L (2x)^L / \sqrt(L!)
    # using Horner's method

    rho = rho * (2 * jnp.ones((M, M)) - jnp.diag(jnp.ones(M)))
    while L > 0:
        L -= 1
        # here c_L = _wig_laguerre_val(L, B, np.diag(rho, L))
        w0 = _wig_laguerre_val(L, B, jnp.diag(rho, L)) + w0 * A2 * (L + 1) ** -0.5

    return w0.real * jnp.exp(-B * 0.5) * (g * g * 0.5 / jnp.pi)


def _wig_laguerre_val(L, x, c):
    r"""
    this is evaluation of polynomial series inspired by hermval from numpy.
    Returns polynomial series

    .. math:
        \sum_n b_n LL_n^L,

    where

    .. math:
        LL_n^L = (-1)^n \sqrt(L!n!/(L+n)!) LaguerreL[n,L,x]

    The evaluation uses Clenshaw recursion.
    """

    if len(c) == 1:
        y0 = c[0]
        y1 = 0
    elif len(c) == 2:
        y0 = c[0]
        y1 = c[1]
    else:
        k = len(c)
        y0 = c[-2]
        y1 = c[-1]
        for i in range(3, len(c) + 1):
            k -= 1
            y0, y1 = (
                c[-i] - y1 * (float((k - 1) * (L + k - 1)) / ((L + k) * k)) ** 0.5,
                y0 - y1 * ((L + 2 * k - 1) - x) * ((L + k) * k) ** -0.5,
            )

    return y0 - y1 * ((L + 1) - x) * (L + 1) ** -0.5


def husimi(psi, xvec, yvec, g=2):
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
    ns = jnp.arange(vector.shape[0])
    out = jnp.polyval(
        (0.5 * g * vector / jnp.sqrt(factorial(ns)))[::-1],
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

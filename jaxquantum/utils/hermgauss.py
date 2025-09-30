import jax.numpy as jnp
from jax import jit, lax
from jax._src.numpy.util import promote_dtypes_inexact

"""
The following code is sourced from https://github.com/f0uriest/orthax/
and is licensed under the MIT license.

Copyright (c) 2024 Rory Conlin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def as_series(*arrs):
    """Return arguments as a list of 1-d arrays.

    The returned list contains array(s) of dtype double, complex double, or
    object.  A 1-d argument of shape ``(N,)`` is parsed into ``N`` arrays of
    size one; a 2-d argument of shape ``(M,N)`` is parsed into ``M`` arrays
    of size ``N`` (i.e., is "parsed by row"); and a higher dimensional array
    raises a Value Error if it is not first reshaped into either a 1-d or 2-d
    array.

    Parameters
    ----------
    arrs : array_like
        1- or 2-d array_like
    trim : boolean, optional
        When True, trailing zeros are removed from the inputs.
        When False, the inputs are passed through intact.

    Returns
    -------
    a1, a2,... : 1-D arrays
        A copy of the input data as 1-d arrays.

    """
    arrays = tuple(jnp.array(a, ndmin=1) for a in arrs)
    arrays = promote_dtypes_inexact(*arrays)
    if len(arrays) == 1:
        return arrays[0]
    return tuple(arrays)


@jit
def hermcompanion(c):
    """Return the scaled companion matrix of c.

    The basis polynomials are scaled so that the companion matrix is
    symmetric when `c` is an Hermite basis polynomial. This provides
    better eigenvalue estimates than the unscaled case and for basis
    polynomials the eigenvalues are guaranteed to be real if
    `jax.numpy.linalg.eigvalsh` is used to obtain them.

    Parameters
    ----------
    c : array_like
        1-D array of Hermite series coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Scaled companion matrix of dimensions (deg, deg).

    """
    c = as_series(c)
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return jnp.array([[-0.5 * c[0] / c[1]]])

    n = len(c) - 1
    mat = jnp.zeros((n, n), dtype=c.dtype)
    scl = jnp.hstack((1.0, 1.0 / jnp.sqrt(2.0 * jnp.arange(n - 1, 0, -1))))
    scl = jnp.cumprod(scl)[::-1]
    shp = mat.shape
    mat = mat.flatten()
    mat = mat.at[1 :: n + 1].set(jnp.sqrt(0.5 * jnp.arange(1, n)))
    mat = mat.at[n :: n + 1].set(jnp.sqrt(0.5 * jnp.arange(1, n)))
    mat = mat.reshape(shp)
    mat = mat.at[:, -1].add(-scl * c[:-1] / (2.0 * c[-1]))
    return mat


@jit
def _normed_hermite_n(x, n):
    """
    Evaluate a normalized Hermite polynomial.

    Compute the value of the normalized Hermite polynomial of degree ``n``
    at the points ``x``.


    Parameters
    ----------
    x : ndarray of double.
        Points at which to evaluate the function
    n : int
        Degree of the normalized Hermite function to be evaluated.

    Returns
    -------
    values : ndarray
        The shape of the return value is described above.

    Notes
    -----
    This function is needed for finding the Gauss points and integration
    weights for high degrees. The values of the standard Hermite functions
    overflow when n >= 207.

    """

    def truefun():
        return jnp.full(x.shape, 1 / jnp.sqrt(jnp.sqrt(jnp.pi)))

    def falsefun():
        c0 = jnp.zeros_like(x)
        c1 = jnp.ones_like(x) / jnp.sqrt(jnp.sqrt(jnp.pi))
        nd = jnp.array(n).astype(float)

        def body(i, val):
            c0, c1, nd = val
            tmp = c0
            c0 = -c1 * jnp.sqrt((nd - 1.0) / nd)
            c1 = tmp + c1 * x * jnp.sqrt(2.0 / nd)
            nd = nd - 1.0
            return c0, c1, nd

        c0, c1, _ = lax.fori_loop(0, n - 1, body, (c0, c1, nd))
        return c0 + c1 * x * jnp.sqrt(2)

    return lax.cond(n == 0, truefun, falsefun)


def hermgauss(deg):
    r"""Gauss-Hermite quadrature.

    Computes the sample points and weights for Gauss-Hermite quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[-\inf, \inf]`
    with the weight function :math:`f(x) = \exp(-x^2)`.

    Parameters
    ----------
    deg : int
        Number of sample points and weights. It must be >= 1.

    Returns
    -------
    x : ndarray
        1-D ndarray containing the sample points.
    y : ndarray
        1-D ndarray containing the weights.

    Notes
    -----
    The results have only been tested up to degree 100, higher degrees may
    be problematic. The weights are determined by using the fact that

    .. math:: w_k = c / (H'_n(x_k) * H_{n-1}(x_k))

    where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
    is the k'th root of :math:`H_n`, and then scaling the results to get
    the right value when integrating 1.

    """
    deg = int(deg)
    if deg <= 0:
        raise ValueError("deg must be a positive integer")

    # first approximation of roots. We use the fact that the companion
    # matrix is symmetric in this case in order to obtain better zeros.
    c = jnp.zeros(deg + 1).at[-1].set(1)
    m = hermcompanion(c)
    x = jnp.linalg.eigvalsh(m)

    # improve roots by one application of Newton
    dy = _normed_hermite_n(x, deg)
    df = _normed_hermite_n(x, deg - 1) * jnp.sqrt(2 * deg)
    x -= dy / df

    # compute the weights. We scale the factor to avoid possible numerical
    # overflow.
    fm = _normed_hermite_n(x, deg - 1)
    fm /= jnp.abs(fm).max()
    w = 1 / (fm * fm)

    # for Hermite we can also symmetrize
    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    # scale w to get the right value
    w *= jnp.sqrt(jnp.pi) / w.sum()

    return x, w

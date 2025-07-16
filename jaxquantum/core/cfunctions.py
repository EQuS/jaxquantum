import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import factorial
import jaxquantum as jqt


def cf_wigner(psi, xvec, yvec):
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


    Returns
    -------

    W : array
        Values representing the Wigner function calculated over the specified
        range [xvec,yvec].


    """
    N = psi.dims[0][0]
    x, y = jnp.meshgrid(xvec, yvec)
    alpha = x + 1.0j * y
    displacement = jqt.displace(N, alpha)

    vmapped_overlap = [vmap(vmap(jqt.overlap, in_axes=(None, 0)), in_axes=(
        None, 0))]
    for _ in psi.bdims:
        vmapped_overlap.append(vmap(vmapped_overlap[-1], in_axes=(0, None)))

    cf = vmapped_overlap[-1](psi, displacement)
    return cf

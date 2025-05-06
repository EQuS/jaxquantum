""" Oscillator gates. """

from jaxquantum.core.operators import displace, basis
from jaxquantum.circuits.gates import Gate

def D(N, alpha):
    return Gate.create(
        N, 
        name="D",
        params={"alpha": alpha},
        gen_U = lambda params: displace(N,params["alpha"]),
        num_modes=1
    )

def CD(N, beta):
    g = basis(2,0)
    e = basis(2,1)
    
    gg = g @ g.dag()
    ee = e @ e.dag()

    return Gate.create(
        [2,N], 
        name="CD",
        params={"beta": beta},
        gen_U = lambda params: (gg ^ displace(N, params["beta"]/2)) + (ee ^ displace(N, -params["beta"]/2)),
        num_modes=2
    )
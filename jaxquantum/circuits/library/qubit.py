""" qubit gates. """

from jaxquantum.core.operators import sigmax, sigmay, sigmaz, basis
from jaxquantum.circuits.gates import Gate

def X():
    return Gate.create(
        2, 
        name="X",
        gen_U = lambda params: sigmax(),
        num_modes=1
    )

def Y():
    return Gate.create(
        2, 
        name="Y",
        gen_U = lambda params: sigmay(),
        num_modes=1
    )

def Z():
    return Gate.create(
        2, 
        name="Z",
        gen_U = lambda params: sigmaz(),
        num_modes=1
    )

def CX():
    g = basis(2,0)
    e = basis(2,1)
    
    gg = g @ g.dag()
    ee = e @ e.dag()

    op =  (gg ^ identity(2)) + (ee ^ sigmax())
    
    return Gate.create(
        [2,2],
        name="CX",
        gen_U = lambda params: op,
        num_modes=2
    )

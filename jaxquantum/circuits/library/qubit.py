""" qubit gates. """

from jaxquantum.core.operators import sigmax, sigmay, sigmaz, basis, hadamard, qubit_rotation
from jaxquantum.circuits.gates import Gate
from jaxquantum.core.qarray import QarrayArray

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

def H():
    return Gate.create(
        2, 
        name="H",
        gen_U = lambda params: hadamard(),
        num_modes=1
    )

def Rx(theta):
    return Gate.create(
        2, 
        name="Rx",
        params={"theta": theta},
        gen_U = lambda params: qubit_rotation(params["theta"], 1, 0, 0),
        num_modes=1
    )

def Ry(theta):
    return Gate.create(
        2, 
        name="Ry",
        params={"theta": theta},
        gen_U = lambda params: qubit_rotation(params["theta"], 0, 1, 0),
        num_modes=1
    )

def Rz(theta):
    return Gate.create(
        2, 
        name="Rz",
        params={"theta": theta},
        gen_U = lambda params: qubit_rotation(params["theta"], 0, 0, 1),
        num_modes=1
    )

def MZ():

    g = basis(2,0)
    e = basis(2,1)

    gg = g @ g.dag()
    ee = e @ e.dag()

    kmap = QarrayArray.create([gg, ee])

    return Gate.create(
        2, 
        name="MZ",
        gen_KM = lambda params: kmap,
        num_modes=1
    )

def MX():

    g = basis(2,0)
    e = basis(2,1)

    plus = (g + e).unit()
    minus = (g - e).unit()

    pp = plus @ plus.dag()
    mm = minus @ minus.dag()

    kmap = QarrayArray.create([pp, mm])

    return Gate.create(
        2, 
        name="MX",
        gen_KM = lambda params: kmap,
        num_modes=1
    )

def MX_plus():

    g = basis(2,0)
    e = basis(2,1)
    plus = (g + e).unit()
    pp = plus @ plus.dag()
    kmap = QarrayArray.create([2*pp])

    return Gate.create(
        2, 
        name="MXplus",
        gen_KM = lambda params: kmap,
        num_modes=1
    )

def MZ_plus():
    g = basis(2,0)
    plus = g
    pp = plus @ plus.dag()
    kmap = QarrayArray.create([2*pp])

    return Gate.create(
        2, 
        name="MZplus",
        gen_KM = lambda params: kmap,
        num_modes=1
    )

def Reset():
    g = basis(2,0)
    e = basis(2,1)

    gg = g @ g.dag()
    ge = g @ e.dag()

    kmap = QarrayArray.create([gg, ge])
    return Gate.create(
        2, 
        name="Reset",
        gen_KM = lambda params: kmap,
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

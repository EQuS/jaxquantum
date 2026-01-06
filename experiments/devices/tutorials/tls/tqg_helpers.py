from jax import jit, grad, config
import jaxquantum as jqt
import jaxquantum.devices as jqtd
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt


jqt.set_precision("double")

def gen_devices(params):

    Ecq_1 = params.get("Qubit1__Ec", 0.125)
    Ej1q_1 = params.get("Qubit1__Ej1", 8.2)
    Ej2q_1 = params.get("Qubit1__Ej2", 23)

    Ecq_2 = params.get("Qubit2__Ec", 0.125)
    Ej1q_2 = params.get("Qubit2__Ej1", 8.2)
    Ej2q_2 = params.get("Qubit2__Ej2", 23)

    phi_ext_1 = params.get("Qubit1__phi_ext", 0.0)
    phi_ext_2 = params.get("Qubit2__phi_ext", 0.0)

    qubit_1 = jqtd.TunableTransmon.create(
        3, # final Hilbert space dimension, after diagonalization and truncation
        {"Ec": Ecq_1, "Ej1": Ej1q_1, "Ej2": Ej2q_1, "phi_ext": phi_ext_1},
        basis=jqtd.BasisTypes.charge,
        hamiltonian=jqtd.HamiltonianTypes.full,
        N_pre_diag=23, # pre-diagonalization Hilbert space dimension
        label=0,
    )

    qubit_2 = jqtd.TunableTransmon.create(
        3, # final Hilbert space dimension, after diagonalization and truncation
        {"Ec": Ecq_2, "Ej1": Ej1q_2, "Ej2": Ej2q_2, "phi_ext": phi_ext_2},
        basis=jqtd.BasisTypes.charge,
        hamiltonian=jqtd.HamiltonianTypes.full,
        N_pre_diag=23, # pre-diagonalization Hilbert space dimension
        label=1,
    )

    tls = jqtd.IdealQubit.create(
            2,
            {"f": params.get("TLS__f", 5.0), "Δ": 0.0}, # GHz
            basis=jqtd.BasisTypes.fock,
            hamiltonian=jqtd.HamiltonianTypes.full,
            N_pre_diag=2,
        )

    return qubit_1, qubit_2, tls

def gen_system(params):
    qubit_1, qubit_2, tls = gen_devices(params)

    λ = params.get("Qubit2_TLS__λ", 0.0) # NOTE: dimensionless coupling strength to between qubit2 and the TLS, for now this is set to zero
    g = params.get("Qubit1_Qubit2__g", 5e-3) # GHz # qubit1 - qubit2 coupling strength

    devices = [qubit_1, qubit_2, tls]
    q1_indx = 0
    q2_indx = 1
    t_indx = 2

    Ns = [device.N for device in devices]


    n1 = jqtd.promote(qubit_1.ops["n"], q1_indx, Ns)
    n2 = jqtd.promote(qubit_2.ops["n"], q2_indx, Ns)
    σz = jqtd.promote(tls.ops["sigmaz"], t_indx, Ns)

    couplings = [] 

    # coupling between qubit 2 and tls 
    couplings.append(8*qubit_2.params["Ec"]*λ * n1 @ σz)


    J = g / (qubit_1.n_zpf() * qubit_2.n_zpf())

    # charge coupling between qubit 1 and qubit 2
    couplings.append(J * n1 @ n2)


    system = jqtd.System.create(devices, couplings=couplings)
    system.params["λ"] = λ

    return system 


def get_metrics(params):
    system = gen_system(params)

    Es, kets = system.calculate_eig()
    H_full = system.get_H()

    metrics = {
        "system": system,
        "Es": Es,
        "kets": kets,
        "H_full": H_full,
    }

    return metrics
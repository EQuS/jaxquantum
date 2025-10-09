import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxquantum.codes as jqtb
import jaxquantum as jqt
import jaxquantum.circuits as jqtc
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
import numpy as np
from jaxquantum.circuits.gates import Gate
from jax_tqdm import scan_tqdm
from jaxopt import GaussNewton
from functools import partial



def _CD_Ancilla_Decay_Kraus_Op(N, beta, gamma_t, l, max_l):
    if l==0:
        K_0 =  jnp.sqrt(jnp.exp(-gamma_t))*jqtc.CD(N, beta).U
        return K_0

    K_i = (
        jnp.sqrt(gamma_t/max_l*jnp.exp(-gamma_t*l/max_l))
        * (jqt.sigmax() ^ jqt.identity(N))
        @ jqtc.CD(N, beta*(2*l/max_l-1)).U
    )
    # sigma_down = jqt.Qarray.create(jnp.array([[0, 1], [0, 0]]))
    # K_i = (
    #     jnp.sqrt(gamma_t/max_l*jnp.exp(-gamma_t*l/max_l))
    #     * (sigma_down ^ jqt.identity(N))
    #     @ jqtc.CD(N, beta*(2*l/max_l-1)).U
    # )
    # sigma_up = jqt.Qarray.create(jnp.array([[0, 0], [1, 0]]))
    # K_i = (
    #     jnp.sqrt(gamma_t/max_l*jnp.exp(-gamma_t*l/max_l))
    #     * (sigma_up ^ jqt.identity(N))
    #     @ jqtc.CD(N, beta*(2*l/max_l-1)).U
    # )
    return K_i


def CD_Ancilla_Decay(N, beta, gamma_t, max_l):
    kmap = lambda params: jqt.Qarray.from_list(
        [_CD_Ancilla_Decay_Kraus_Op(N, beta, gamma_t, l, max_l) for l in range(max_l)]
    )
    return Gate.create(
        [2, N],
        name="CD_Ancilla_Decay",
        params={"beta": beta, "gamma_t": gamma_t, "max_l": max_l},
        gen_KM=kmap,
        num_modes=2,
    )





def sbs(initial_state,
        delta,
        sd_ratio,
        T,
        observable,
        t_sqg,
        t_CD_floor,
        t_CD_p,
        t_rst,
        error_channels,
        ):

    l = jnp.sqrt(2*jnp.pi)
    epsilon = jnp.sinh(delta*delta)*l

    alphas_real = jnp.array([epsilon/2, 0., sd_ratio*epsilon/2, 0., l, 0.])
    alphas_imag = jnp.array([0., -l, 0., epsilon/2, 0., sd_ratio*epsilon/2])
    alphas = alphas_real + alphas_imag * 1.j
    phis = jnp.array([0., 0., 0., 0.])
    thetas = jnp.array([jnp.pi/2, -jnp.pi/2, jnp.pi/2, -jnp.pi/2])

    def t_CD(beta):
        return jax.lax.max(t_CD_floor, jnp.polyval(t_CD_p, beta)) #ns

    t_CDs = jax.vmap(t_CD)(jnp.abs(alphas))
    t_CDs = jnp.array([456, 1344, 456, 456, 1344, 456])
    t_round = t_CDs.sum()+t_sqg*8+t_rst*2
    
    exp = jnp.array([jqt.overlap(observable, initial_state.ptrace(1))])
    
    initial_carry = initial_state.to_dm()

    N = initial_state.space_dims[1]
    
    #print("Building circuits")
    
    cirq_Z = sBs_half_round_circuit(N, alphas[0:3], phis[0:2],
                                  thetas[0:2], t_sqg, t_CDs[0:3], t_rst, error_channels)
    
    cirq_X = sBs_half_round_circuit(N, alphas[3:6], phis[2:4],
                                  thetas[2:4], t_sqg, t_CDs[3:6], t_rst, error_channels)
    #print("Start loop")
    @scan_tqdm(T)
    @jax.jit
    def sbs_round(carry, _):
        current_state = carry
        current_state = run_sBs_half_round(cirq_Z, current_state)
        current_state = run_sBs_half_round(cirq_X, current_state)
        exp = jqt.overlap(observable, current_state.ptrace(1))
        return (current_state, exp)

    final_carry, hist = jax.lax.scan(sbs_round, initial_carry, jnp.arange(T), length=T)

    exp = jnp.concatenate((exp, hist))

    ts = t_rst + jnp.linspace(0, T*t_round, T+1)
    
    return ts, exp


def sBs_half_round_circuit(N,
                   alphas, 
                   phis,
                   thetas,
                   t_sqg,
                   t_CD,
                   t_rst,
                   error_channels):
    
    
    reg = jqtc.Register([2, N])

    cirq = jqtc.Circuit.create(reg, layers=[])

    n_bar_qb = error_channels["fluxonium"]["n_bar"]
    n_bar_osc = error_channels["resonator"]["n_bar"]
    max_l = 5
    
    cirq.append(jqtc.Ry(jnp.pi / 2), 0)
    cirq.append(jqtc.Thermal_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["T1"]), n_bar_qb), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["Tphi"])), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")

    cirq.append(CD_Ancilla_Decay(N, alphas[0], 1-jnp.exp(-t_CD[0]/error_channels["fluxonium"]["T1"]), max_l), [0, 1], default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch_Qb(1-jnp.exp(-t_CD[0]/error_channels["fluxonium"]["Tphi"])), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_CD[0]/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_CD[0]/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")
    
    # cirq.append(jqtc.Ry(phis[0]), 0)
    # cirq.append(jqtc.Thermal_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["T1"]), n_bar_qb), 0, default_simulate_mode="kraus")
    # cirq.append(jqtc.Dephasing_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["Tphi"])), 0, default_simulate_mode="kraus")
    # cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    # cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")
    
    cirq.append(jqtc.Rx(thetas[0]), 0)
    cirq.append(jqtc.Thermal_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["T1"]), n_bar_qb), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["Tphi"])), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")
    
    cirq.append(CD_Ancilla_Decay(N, alphas[1], 1-jnp.exp(-t_CD[1]/error_channels["fluxonium"]["T1"]), max_l), [0, 1], default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch_Qb(1-jnp.exp(-t_CD[1]/error_channels["fluxonium"]["Tphi"])), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_CD[1]/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_CD[1]/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")
    
    # cirq.append(jqtc.Ry(phis[1]), 0)
    # cirq.append(jqtc.Thermal_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["T1"]), n_bar_qb), 0, default_simulate_mode="kraus")
    # cirq.append(jqtc.Dephasing_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["Tphi"])), 0, default_simulate_mode="kraus")
    # cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    # cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")
    
    cirq.append(jqtc.Rx(thetas[1]), 0)
    cirq.append(jqtc.Thermal_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["T1"]), n_bar_qb), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["Tphi"])), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")
    
    cirq.append(CD_Ancilla_Decay(N, alphas[2], 1-jnp.exp(-t_CD[2]/error_channels["fluxonium"]["T1"]), max_l), [0, 1], default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch_Qb(1-jnp.exp(-t_CD[2]/error_channels["fluxonium"]["Tphi"])), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_CD[2]/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_CD[2]/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")
    
    cirq.append(jqtc.IP_Reset(error_channels["fluxonium"]["reset_p_eg"], error_channels["fluxonium"]["reset_p_ee"]), 0, default_simulate_mode="kraus")
    # cirq.append(jqtc.CR(N, error_channels["resonator"]["chi"]*t_rst), [0, 1], default_simulate_mode="unitary")
    cirq.append(jqtc.Thermal_Ch_Qb(1-jnp.exp(-t_rst/error_channels["fluxonium"]["T1"]), n_bar_qb), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch_Qb(1-jnp.exp(-t_rst/error_channels["fluxonium"]["Tphi"])), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_rst/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_rst/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")

    return cirq
    

def run_sBs_half_round(cirq, initial_state):
    
    res = jqtc.simulate(cirq, initial_state, mode='default')

    final_state = res[-1][-1]


    return final_state












    
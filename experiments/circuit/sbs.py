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



@partial(jax.jit, static_argnames=["N", "max_l"])
def _CD_Ancilla_Decay_Kraus_Map_JIT(N, beta, gamma_t, n_th, max_l):
    # Calculate thermal probabilities
    # p_down: Relaxation |e> -> |g>
    # p_up:   Excitation |g> -> |e>
    p_down = gamma_t * (1.0 + n_th) / (1.0 + 2*n_th)
    p_up = gamma_t * n_th / (1.0 + 2*n_th)
    

    def compute_op(idx):
        # --- Branch 0: No Jump ---
        def branch_no_jump(_):
            # Qubit Projectors: Pg=|0><0|, Pe=|1><1|
            # Note: We assume standard basis |0>=g, |1>=e
            Pg = (jqt.basis(2, 0)@jqt.basis(2, 0).dag()).data
            Pe = (jqt.basis(2, 1)@jqt.basis(2, 1).dag()).data
            
            # State-dependent damping amplitudes
            # If in |g>, probability to leave is p_up. Survivor: sqrt(1-p_up)
            # If in |e>, probability to leave is p_down. Survivor: sqrt(1-p_down)
            amp_g = jnp.sqrt(1.0 - p_up)
            amp_e = jnp.sqrt(1.0 - p_down)
            
            q_damp = amp_g * Pg + amp_e * Pe
            
            # M0 = (Damping on Qubit) @ CD_Unitary
            # q_damp must be tensored with Identity on Cavity
            damp_expanded = jnp.kron(q_damp, jnp.eye(N))
            
            return damp_expanded @ jqtc.CD(N, beta).U.data

        # --- Branch > 0: Quantum Jumps ---
        def branch_jump(_):
            # Determine if Relaxation (indices 1..L) or Excitation (indices L+1..2L)
            is_relax = idx <= max_l
            
            # Normalize l to 1..L for displacement calc
            l = jax.lax.select(is_relax, idx, idx - max_l)
            
            # Calculate effective displacement alpha = beta * (2*l/N - 1)
            alpha = beta * (2.0 * l / max_l - 1.0)
            
            # Oscillator Operator: Displacement D(alpha)
            # We use jqt.displace instead of CD because the error is a simple displacement
            d_op = jqtc.CD(N, alpha).U.data
            
            # Qubit Operator & Weight selection
            # Relaxation: sigmam (|0><1|), weight ~ p_down
            # Excitation: sigmap (|1><0|), weight ~ p_up
            q_op = jax.lax.select(
                is_relax, 
                jqt.sigmam().data, 
                jqt.sigmap().data
            )
            
            weight = jax.lax.select(
                is_relax,
                jnp.sqrt(p_down / max_l),
                jnp.sqrt(p_up / max_l)
            )
            
            # Kraus Op = weight * (Q_jump tensor D_alpha)
            return weight * (jnp.kron(q_op, jnp.eye(N))) @ d_op

        return jax.lax.cond(idx == 0, branch_no_jump, branch_jump, operand=None)

    # Map over total indices: 0 (No Jump) + L (Relax) + L (Excite)
    idxs = jnp.arange(2 * max_l + 1)
    
    return jax.lax.map(compute_op, idxs)

# 2. Update the Gate definition
def CD_Ancilla_Decay(N, beta, gamma_t, n_th, max_l):
    # CRITICAL FIX: Pass dims=[[2, N], [2, N]] so Qarray knows it's 2 modes
    kmap = lambda params: jqt.Qarray.create(
        _CD_Ancilla_Decay_Kraus_Map_JIT(
            params["N"], 
            params["beta"], 
            params["gamma_t"], 
            params["n_th"], 
            params["max_l"]
        ),
        dims=[[2, N], [2, N]],        
        bdims=(2 * params["max_l"] + 1,)  # Updated batch dim for 2*L+1 operators
    )
    return Gate.create(
        [2, N],
        name="CD_Ancilla_Decay",
        params={"beta": beta, "gamma_t": gamma_t, "n_th": n_th, "max_l": max_l, "N": N},
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
        speedup
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
    t_CDs = jnp.array([288, 1640, 384, 288, 1640, 384]) / speedup
    t_round = t_CDs.sum()+t_sqg*6+t_rst*2
    
    exp = jnp.array([jqt.overlap(observable, initial_state.ptrace(1))])
    
    initial_carry = initial_state.to_dm()

    N = initial_state.space_dims[1]
    
    #print("Building circuits")
    
    cirq_Z = sBs_half_round_circuit(N, alphas[0:3], phis[0:2],
                                  thetas[0:2], t_sqg, t_CDs[0:3], t_rst, error_channels)
    
    cirq_X = sBs_half_round_circuit(N, alphas[3:6], phis[2:4],
                                  thetas[2:4], t_sqg, t_CDs[3:6], t_rst, error_channels)
    #print("Start loop")
    # @scan_tqdm(T)
    # @jax.jit
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
    max_l = 12
    
    cirq.append(jqtc.Ry(jnp.pi / 2), 0)
    cirq.append(jqtc.Thermal_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["T1"]), n_bar_qb), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch_Qb((1-jnp.exp(-t_sqg/error_channels["fluxonium"]["Tphi"]))/2), 0, default_simulate_mode="kraus")
    # cirq.append(jqtc.Amp_Damp(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["T1"]), max_l), 1, default_simulate_mode="kraus")
    # cirq.append(jqtc.CR(N, t_sqg * error_channels["resonator"]["chi"]), [0, 1], default_simulate_mode="unitary")
    cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    #cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")

    cirq.append(CD_Ancilla_Decay(N, alphas[0], t_CD[0]/error_channels["fluxonium"]["T1"], n_bar_qb, max_l), [0, 1], default_simulate_mode="kraus")
    
    cirq.append(jqtc.Thermal_Ch_Qb(1-jnp.exp(-t_CD[0]/error_channels["fluxonium"]["T1"]), n_bar_qb), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch_Qb((1-jnp.exp(-t_CD[0]/error_channels["fluxonium"]["Tphi"]))/2), 0, default_simulate_mode="kraus")
    # cirq.append(jqtc.Amp_Damp(N, 1-jnp.exp(-t_CD[0]/error_channels["resonator"]["T1"]), max_l), 1, default_simulate_mode="kraus")
    # cirq.append(jqtc.CR(N, t_CD[0] * error_channels["resonator"]["chi"]), [0, 1], default_simulate_mode="unitary")
    cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_CD[0]/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    #cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_CD[0]/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")
    
    # # cirq.append(jqtc.Ry(phis[0]), 0)
    # # cirq.append(jqtc.Thermal_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["T1"]), n_bar_qb), 0, default_simulate_mode="kraus")
    # # cirq.append(jqtc.Dephasing_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["Tphi"])), 0, default_simulate_mode="kraus")
    # # cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    # # cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")
    
    cirq.append(jqtc.Rx(thetas[0]), 0)
    cirq.append(jqtc.Thermal_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["T1"]), n_bar_qb), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch_Qb((1-jnp.exp(-t_sqg/error_channels["fluxonium"]["Tphi"]))/2), 0, default_simulate_mode="kraus")
    # cirq.append(jqtc.Amp_Damp(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["T1"]), max_l), 1, default_simulate_mode="kraus")
    # cirq.append(jqtc.CR(N, t_sqg* error_channels["resonator"]["chi"]), [0, 1], default_simulate_mode="unitary")
    cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    #cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")
    
    cirq.append(CD_Ancilla_Decay(N, alphas[1], t_CD[1]/error_channels["fluxonium"]["T1"], n_bar_qb, max_l), [0, 1], default_simulate_mode="kraus")
    cirq.append(jqtc.Thermal_Ch_Qb(1-jnp.exp(-t_CD[1]/error_channels["fluxonium"]["T1"]), n_bar_qb), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch_Qb((1-jnp.exp(-t_CD[1]/error_channels["fluxonium"]["Tphi"]))/2), 0, default_simulate_mode="kraus")
    # cirq.append(jqtc.Amp_Damp(N, 1-jnp.exp(-t_CD[1]/error_channels["resonator"]["T1"]), max_l), 1, default_simulate_mode="kraus")
    # cirq.append(jqtc.CR(N, t_CD[1] * error_channels["resonator"]["chi"]), [0, 1], default_simulate_mode="unitary")
    cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_CD[1]/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    #cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_CD[1]/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")
    
    # # cirq.append(jqtc.Ry(phis[1]), 0)
    # # cirq.append(jqtc.Thermal_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["T1"]), n_bar_qb), 0, default_simulate_mode="kraus")
    # # cirq.append(jqtc.Dephasing_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["Tphi"])), 0, default_simulate_mode="kraus")
    # # cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    # # cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")
    
    cirq.append(jqtc.Rx(thetas[1]), 0)
    cirq.append(jqtc.Thermal_Ch_Qb(1-jnp.exp(-t_sqg/error_channels["fluxonium"]["T1"]), n_bar_qb), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch_Qb((1-jnp.exp(-t_sqg/error_channels["fluxonium"]["Tphi"]))/2), 0, default_simulate_mode="kraus")
    # cirq.append(jqtc.Amp_Damp(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["T1"]), max_l), 1, default_simulate_mode="kraus")
    # cirq.append(jqtc.CR(N, t_sqg * error_channels["resonator"]["chi"]), [0, 1], default_simulate_mode="unitary")
    cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    #cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_sqg/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")
    
    cirq.append(CD_Ancilla_Decay(N, alphas[2], t_CD[2]/error_channels["fluxonium"]["T1"], n_bar_qb, max_l), [0, 1], default_simulate_mode="kraus")
    cirq.append(jqtc.Thermal_Ch_Qb(1-jnp.exp(-t_CD[2]/error_channels["fluxonium"]["T1"]), n_bar_qb), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch_Qb((1-jnp.exp(-t_CD[2]/error_channels["fluxonium"]["Tphi"]))/2), 0, default_simulate_mode="kraus")
    # cirq.append(jqtc.Amp_Damp(N, 1-jnp.exp(-t_CD[2]/error_channels["resonator"]["T1"]), max_l), 1, default_simulate_mode="kraus")
    # cirq.append(jqtc.CR(N, t_CD[2] * error_channels["resonator"]["chi"]), [0, 1], default_simulate_mode="unitary")
    cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_CD[2]/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    #cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_CD[2]/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")
    
    cirq.append(jqtc.Dephasing_Reset(N, error_channels["fluxonium"]["reset_p_ee"], error_channels["fluxonium"]["t_rst"], error_channels["resonator"]["chi"], max_l), [0, 1], default_simulate_mode="kraus")
    cirq.append(jqtc.Thermal_Ch_Qb(1-jnp.exp(-t_rst/error_channels["fluxonium"]["T1"]), n_bar_qb), 0, default_simulate_mode="kraus")
    cirq.append(jqtc.Dephasing_Ch_Qb((1-jnp.exp(-t_rst/error_channels["fluxonium"]["Tphi"]))/2), 0, default_simulate_mode="kraus")
    # cirq.append(jqtc.Amp_Damp(N, 1-jnp.exp(-t_rst/error_channels["resonator"]["T1"]), max_l), 1, default_simulate_mode="kraus")
    # cirq.append(jqtc.CR(N, t_sqg * error_channels["resonator"]["chi"]), [0, 1], default_simulate_mode="unitary")
    cirq.append(jqtc.Thermal_Ch(N, 1-jnp.exp(-t_rst/error_channels["resonator"]["T1"]), n_bar_osc, max_l), 1, default_simulate_mode="kraus")
    #cirq.append(jqtc.Dephasing_Ch(N, 1-jnp.exp(-t_rst/error_channels["resonator"]["Tphi"]), max_l), 1, default_simulate_mode="kraus")

    return cirq
    

def run_sBs_half_round(cirq, initial_state):
    
    res = jqtc.simulate(cirq, initial_state, mode='default')

    final_state = res[-1][-1]


    return final_state












    
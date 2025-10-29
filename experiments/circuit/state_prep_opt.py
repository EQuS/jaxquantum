import jaxquantum as jqt
import jaxquantum.circuits as jqtc
import jaxquantum.codes as jqtb
from jax import jit, grad, vmap
import jax.numpy as jnp
import numpy as np
from math import prod
import matplotlib.pyplot as plt
from jax_tqdm import scan_tqdm
from tqdm import tqdm

import jax
import optax
from functools import partial
import datetime



def run_circuit(params, N, measure=None):
    gammas_x = 2*jnp.pi*params[0]
    gammas_y = 2*jnp.pi*params[1]
    betas_re = params[2]
    betas_im = params[3]

    betas = betas_re + 1j*betas_im
    
    reg = jqtc.Register([2,N])
    cirq = jqtc.Circuit.create(reg, layers=[])

    for i in range(len(gammas_x)):
        cirq.append(jqtc.Rx(gammas_x[i]), 0)
        cirq.append(jqtc.Ry(gammas_y[i]), 0)
        cirq.append(jqtc.CD(N, betas[i]), [0, 1])
        cirq.append(jqtc.Rx(jnp.pi), 0)
    
    if measure == 'x':
        cirq.append(jqtc.MX_plus(), 0, default_simulate_mode="kraus")
        cirq.append(jqtc.Ry(-jnp.pi/2), 0)
        initial_state = jqt.basis(2,0) ^ jqt.basis(N,0)
        res = jqtc.simulate(cirq, initial_state, mode="default")
        return res[-1][-1]/res[-1][-1].trace()
        
    if measure == 'z':
        cirq.append(jqtc.MZ_plus(), 0, default_simulate_mode="kraus")
        initial_state = jqt.basis(2,0) ^ jqt.basis(N,0)
        res = jqtc.simulate(cirq, initial_state, mode="default")
        return res[-1][-1]/res[-1][-1].trace()
        
    initial_state = jqt.basis(2,0) ^ jqt.basis(N,0)
    res = jqtc.simulate(cirq, initial_state, mode="default")

    return res[-1][-1].unit()
    

def fid_metric(params, target_state):
    N = target_state.dims[0][1]
    prepared_state = run_circuit(params, N)
    return jnp.real(jqt.overlap(target_state, prepared_state))

fid_metric_vmap = jax.vmap(fid_metric, in_axes=(0, None))


def regularize(params, reg_mode, reg_strength):
    betas_re = params[2]
    betas_im = params[3]

    betas_amp = jnp.abs(betas_re + 1j*betas_im)

    if reg_mode=='avg':
        reg_loss = reg_strength * jnp.average(betas_amp**2)
    if reg_mode=='max':
        reg_loss = reg_strength * jnp.max(betas_amp)**2
        
    return reg_loss

regularize_vmap = jax.vmap(regularize, in_axes=(0, None, None))

def metric(params, target_state, reg_mode, reg_strength):
    fids = fid_metric_vmap(params, target_state)
    losses = jnp.log10(1 - fids)
    #losses = 1.-fids
    avg_loss = losses
    reg_loss = regularize_vmap(params, reg_mode, reg_strength)
    
    return jnp.average(avg_loss + reg_loss)

def metric_non_avg(params, target_state, reg_mode, reg_strength):
    fids = fid_metric_vmap(params, target_state)
    losses = jnp.log10(1 - fids)
    #losses = 1.-fids
    avg_loss = losses
    reg_loss = regularize_vmap(params, reg_mode, reg_strength)

    return avg_loss + reg_loss

    
def cf_tomography_circuit(state, beta, measure_real=True):
    N = state.dims[0][1]
    reg = jqtc.Register([2,N])
    cirq = jqtc.Circuit.create(reg, layers=[])

    cirq.append(jqtc.Ry(jnp.pi/2), 0)
    cirq.append(jqtc.CD(N, beta), [0,1])
    
    if measure_real:
        cirq.append(jqtc.Ry(jnp.pi/2), 0)
    else:
        cirq.append(jqtc.Rx(jnp.pi/2), 0)

    res = jqtc.simulate(cirq, state)
    final_state = res[-1][-1]
    sigmaz = jqt.sigmaz() ^ jqt.identity(N)
    sigmaz_exp = final_state.dag() @ sigmaz @ final_state
    return sigmaz_exp.data[0][0].real

def sim_cf(osc_state, betas_re=None, betas_im=None):
    if len(osc_state.dims[0]) == 1:
        if osc_state.is_dm():
            state = jqt.ket2dm(jqt.basis(2,0)) ^ osc_state
        else:
            state = jqt.basis(2,0) ^ osc_state
    else:
        state = osc_state

    # Plot CF
    betas_re = betas_re if betas_re is not None else jnp.linspace(-4,4, 101)
    betas_im = betas_re if betas_re is not None else  jnp.linspace(-4,4, 101)
    betas = betas_re.reshape(-1,1) + 1j*betas_im.reshape(1,-1)
    betas_flat = betas.flatten()

    cf_tomography_circuit_vmap = jit(vmap(lambda beta: cf_tomography_circuit(state, beta, measure_real=True)))
    tomo_res_real = cf_tomography_circuit_vmap(betas_flat)

    cf_tomography_circuit_vmap = jit(vmap(lambda beta: cf_tomography_circuit(state, beta, measure_real=False)))
    tomo_res_imag = cf_tomography_circuit_vmap(betas_flat)
    
    tomo_res_real = tomo_res_real.reshape(*betas.shape)
    tomo_res_imag = tomo_res_imag.reshape(*betas.shape)

    tomo_res = tomo_res_real + 1j*tomo_res_imag

    return tomo_res, betas_re, betas_im


def calculate_cf(osc_state, betas_re=None, betas_im=None):
    # Plot CF
    N = osc_state.dims[0][0]

    betas_re = betas_re if betas_re is not None else jnp.linspace(-4,4, 41)
    betas_im = betas_im if betas_im is not None else jnp.linspace(-4,4, 41)
    betas = betas_re.reshape(-1,1) + 1j*betas_im.reshape(1,-1)
    
    cf_vals = np.zeros((len(betas_re), len(betas_im)), dtype=jnp.complex64)
    for j in tqdm(range(len(betas_re))):
        for k in range(len(betas_im)):
            cf_vals[j,k] = jqt.overlap(jqt.displace(N, betas[j,k]), osc_state)
    return cf_vals, betas_re, betas_im

metric_val_and_grad = jit(jax.value_and_grad(metric), static_argnames="reg_mode")
metric_non_avg_jit = jit(metric_non_avg, static_argnames="reg_mode")
fid_metric_vmap_jit = jit(fid_metric_vmap)

def optimize(initial_params, settings):
    start_learning_rate = settings["learning_rate"]
    optimizer = optax.adam(start_learning_rate)
    

    opt_state = optimizer.init(initial_params)
    
    
    epochs = settings["epochs"]
    
    @scan_tqdm(epochs)
    @jax.jit
    def scan_step(carry, _):
        """One step of the training loop for lax.scan."""
        params, opt_state = carry
        metrics, grads = metric_val_and_grad(params, settings["target_state"], settings["reg_mode"], settings["reg_strength"])
        fids = fid_metric_vmap_jit(params, settings["target_state"])
        metric_batch = metric_non_avg_jit(params, settings['target_state'], settings["reg_mode"], settings["reg_strength"])

        
        history_slice = {
            "params": params,
            "fids": fids,
            "metric": metrics,
            "metric_batch": metric_batch
        }

        
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        new_carry = (new_params, new_opt_state)

        return new_carry, history_slice

    initial_carry = (initial_params, opt_state)
    final_carry, history = jax.lax.scan(
        scan_step, initial_carry, jnp.arange(epochs), length=epochs
    )
    
    return history
    
















        
"""System."""

from typing import List, Optional, Dict, Any, Union
import math

from flax import struct
from jax import vmap, Array
from jax import config

import jax.numpy as jnp

from jaxquantum.devices.base.base import Device
from jaxquantum.devices.superconducting.drive import Drive
from jaxquantum.core.qarray import Qarray, tensor
from jaxquantum.core.operators import identity




def calculate_eig(Ns, H: Qarray):
    N_tot = math.prod(Ns)
    edxs = jnp.arange(N_tot)

    vals, kets = jnp.linalg.eigh(H.data)
    kets = kets.T

    def calc_quantum_number(edx):
        argmax = jnp.argmax(jnp.abs(kets[edx]))
        val = vals[edx]  # - vals[0]
        return val, argmax, kets[edx]

    quantum_numbers = vmap(calc_quantum_number)(edxs)

    def calc_order(edx):
        indx = jnp.argmin(jnp.abs(edx - quantum_numbers[1]))
        return quantum_numbers[0][indx], quantum_numbers[2][indx]

    Es, kets = vmap(calc_order)(edxs)

    kets = jnp.reshape(kets, (N_tot, N_tot, 1))
    kets = Qarray.create(kets)
    kets = kets.reshape_qdims(*Ns)
    kets = kets.reshape_bdims(*Ns)

    return (
        jnp.reshape(Es, Ns),
        kets,
    )


def promote(op: Qarray, device_num, Ns):
    I_ops = [identity(N) for N in Ns]
    return tensor(*I_ops[:device_num], op, *I_ops[device_num + 1 :])


@struct.dataclass
class System:
    Ns: List[int] = struct.field(pytree_node=False)
    devices: List[Union[Device, Drive]]
    couplings: List[Array]
    params: Dict[str, Any]

    @classmethod
    def create(
        cls,
        devices: List[Union[Device, Drive]],
        couplings: Optional[List[Array]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        labels = [device.label for device in devices]
        unique_labels = set(labels)
        if len(labels) != len(unique_labels):
            raise ValueError("Devices must have unique labels.")

        Ns = tuple([device.N for device in devices])
        couplings = couplings if couplings is not None else []
        params = params if params is not None else {}
        return cls(Ns, devices, couplings, params)

    def promote(self, op, device_num):
        return promote(op, device_num, self.Ns)

    def get_H_bare(self):
        H = 0
        for j, device in enumerate(self.devices):
            H += self.promote(device.get_H(), j)
        return H

    def get_H_couplings(self):
        H = 0
        for coupling in self.couplings:
            H += coupling
        return H

    def get_H(self):
        H_bare = self.get_H_bare()
        H_couplings = self.get_H_couplings()
        return H_bare + H_couplings

    def calculate_eig(self):
        H = self.get_H()
        return calculate_eig(self.Ns, H)

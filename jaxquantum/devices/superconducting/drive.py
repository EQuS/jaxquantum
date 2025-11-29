"""Base Drive."""

from abc import ABC
from typing import Dict

from flax import struct
from jax import config
import jax.numpy as jnp

from jaxquantum.core.qarray import Qarray
from jaxquantum.core.conversions import jnp2jqt

config.update("jax_enable_x64", True)


@struct.dataclass
class Drive(ABC):
    N: int = struct.field(pytree_node=False)
    fd: float
    _label: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, M_max, fd, label=0):
        cls.M_max = M_max
        N = 2 * M_max + 1
        return cls(N, fd, label)

    @property
    def label(self):
        return self.__class__.__name__ + str(self._label)

    @property
    def ops(self):
        return self.common_ops()

    def common_ops(self) -> Dict[str, Qarray]:
        ops = {}

        M_max = self.M_max

        # Construct M = ∑ₘ m|m><m| operator in drive charge basis
        ops["M"] = jnp2jqt(jnp.diag(jnp.arange(-M_max, M_max + 1)))

        # Construct Id = ∑ₘ|m><m| in the drive charge basis
        ops["id"] = jnp2jqt(jnp.identity(2 * M_max + 1))

        # Construct M₊ ≡ exp(iθ) and M₋ ≡ exp(-iθ) operators for drive
        ops["M-"] = jnp2jqt(jnp.eye(2 * M_max + 1, k=1))
        ops["M+"] = jnp2jqt(jnp.eye(2 * M_max + 1, k=-1))

        # Construct cos(θ) ≡ 1/2 * [M₊ + M₋] = 1/2 * ∑ₘ|m+1><m| + h.c
        ops["cos(θ)"] = 0.5 * (ops["M+"] + ops["M-"])

        # Construct sin(θ) ≡ -i/2 * [M₊ - M₋] = -i/2 * ∑ₘ|m+1><m| + h.c
        ops["sin(θ)"] = -0.5j * (ops["M+"] - ops["M-"])

        # Construct more general drive operators cos(kθ) and sin(kθ)
        for k in range(2, M_max + 1):
            ops[f"M_+{k}"] = jnp2jqt(jnp.eye(2 * M_max + 1, k=-k))
            ops[f"M_-{k}"] = jnp2jqt(jnp.eye(2 * M_max + 1, k=k))
            ops[f"cos({k}θ)"] = 0.5 * (ops[f"M_+{k}"] + ops[f"M_-{k}"])
            ops[f"sin({k}θ)"] = -0.5j * (ops[f"M_+{k}"] - ops[f"M_-{k}"])

        return ops

    #############################################################

    def get_H(self):
        """
        Bare "drive" Hamiltonian (fd * M) in the extended Hilbert space.
        """
        return self.fd * self.ops["M"]

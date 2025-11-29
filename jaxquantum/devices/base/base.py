"""Base device."""

from abc import abstractmethod, ABC
from enum import Enum
from typing import Dict, Any, List

from flax import struct
from jax import config, Array
import jax.numpy as jnp

from jaxquantum.core.qarray import Qarray
from jaxquantum.core.dims import Qtypes

config.update("jax_enable_x64", True)


class BasisTypes(str, Enum):
    fock = "fock"
    charge = "charge"
    singlecharge = "single_charge"
    singlecharge_even = "singlecharge_even"
    singlecharge_odd = "singlecharge_odd"

    @classmethod
    def from_str(cls, string: str):
        return cls(string)

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __hash__(self):
        return hash(self.value)


class HamiltonianTypes(str, Enum):
    linear = "linear"
    truncated = "truncated"
    full = "full"

    @classmethod
    def from_str(cls, string: str):
        return cls(string)

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __hash__(self):
        return hash(self.value)


@struct.dataclass
class Device(ABC):
    DEFAULT_BASIS = BasisTypes.fock
    DEFAULT_HAMILTONIAN = HamiltonianTypes.full

    N: int = struct.field(pytree_node=False)
    N_pre_diag: int = struct.field(pytree_node=False)
    params: Dict[str, Any]
    _label: int = struct.field(pytree_node=False)
    _basis: BasisTypes = struct.field(pytree_node=False)
    _hamiltonian: HamiltonianTypes = struct.field(pytree_node=False)

    @classmethod
    def param_validation(cls, N, N_pre_diag, params, hamiltonian, basis):
        """This can be overridden by subclasses."""
        pass

    @classmethod
    def create(
        cls,
        N,
        params,
        label=0,
        N_pre_diag=None,
        use_linear=False,
        hamiltonian: HamiltonianTypes = None,
        basis: BasisTypes = None,
    ):
        """Create a device.

        Args:
            N (int): dimension of Hilbert space.
            params (dict): parameters of the device.
            label (int, optional): label for the device. Defaults to 0. This is useful when you have multiple of the same device type in the same system.
            N_pre_diag (int, optional): dimension of Hilbert space before diagonalization. Defaults to None, in which case it is set to N. This must be greater than or rqual to N.
            use_linear (bool): whether to use the linearized device. Defaults to False. This will override the hamiltonian keyword argument. This is a bit redundant with hamiltonian, but it is kept for backwards compatibility.
            hamiltonian (HamiltonianTypes, optional): type of Hamiltonian. Defaults to None, in which case the full hamiltonian is used.
            basis (BasisTypes, optional): type of basis. Defaults to None, in which case the fock basis is used.
        """

        if N_pre_diag is None:
            N_pre_diag = N

        assert N_pre_diag >= N, "N_pre_diag must be greater than or equal to N."

        _basis = basis if basis is not None else cls.DEFAULT_BASIS
        _hamiltonian = (
            hamiltonian if hamiltonian is not None else cls.DEFAULT_HAMILTONIAN
        )

        if use_linear:
            _hamiltonian = HamiltonianTypes.linear

        cls.param_validation(N, N_pre_diag, params, _hamiltonian, _basis)

        return cls(N, N_pre_diag, params, label, _basis, _hamiltonian)

    @property
    def basis(self):
        return self._basis

    @property
    def hamiltonian(self):
        return self._hamiltonian

    @property
    def label(self):
        return self.__class__.__name__ + str(self._label)

    @property
    def linear_ops(self):
        return self.common_ops()

    @property
    def original_ops(self):
        return self.common_ops()

    @property
    def ops(self):
        return self.full_ops()

    @abstractmethod
    def common_ops(self) -> Dict[str, Qarray]:
        """Set up common ops in the specified basis."""

    @abstractmethod
    def get_linear_frequency(self):
        """Get frequency of linear terms."""

    @abstractmethod
    def get_H_linear(self):
        """Return linear terms in H."""

    @abstractmethod
    def get_H_full(self):
        """Return full H."""

    def get_H(self):
        """
        Return diagonalized H. Explicitly keep only diagonal elements of matrix.
        """
        return self.get_op_in_H_eigenbasis(
            self._get_H_in_original_basis()
        ).keep_only_diag_elements()

    def _get_H_in_original_basis(self):
        """This returns the Hamiltonian in the original specified basis. This can be overridden by subclasses."""

        if self.hamiltonian == HamiltonianTypes.linear:
            return self.get_H_linear()
        elif self.hamiltonian == HamiltonianTypes.full:
            return self.get_H_full()

    def _calculate_eig_systems(self):
        evs, evecs = jnp.linalg.eigh(self._get_H_in_original_basis().data)  # Hermitian
        idxs_sorted = jnp.argsort(evs)
        return evs[idxs_sorted], evecs[:, idxs_sorted]

    @property
    def eig_systems(self):
        eig_systems = {}
        eig_systems["vals"], eig_systems["vecs"] = self._calculate_eig_systems()

        eig_systems["vecs"] = eig_systems["vecs"]
        eig_systems["vals"] = eig_systems["vals"]
        return eig_systems

    def get_op_in_H_eigenbasis(self, op: Qarray):
        evecs = self.eig_systems["vecs"][:, : self.N]
        dims = [[self.N], [self.N]]
        return get_op_in_new_basis(op, evecs, dims)

    def get_op_data_in_H_eigenbasis(self, op: Array):
        evecs = self.eig_systems["vecs"][:, : self.N]
        return get_op_data_in_new_basis(op, evecs)

    def get_vec_in_H_eigenbasis(self, vec: Qarray):
        evecs = self.eig_systems["vecs"][:, : self.N]
        if vec.qtype == Qtypes.ket:
            dims = [[self.N], [1]]
        else:
            dims = [[1], [self.N]]
        return get_vec_in_new_basis(vec, evecs, dims)

    def get_vec_data_in_H_eigenbasis(self, vec: Array):
        evecs = self.eig_systems["vecs"][:, : self.N]
        return get_vec_data_in_new_basis(vec, evecs)

    def full_ops(self):
        # TODO: use JAX vmap here

        linear_ops = self.linear_ops
        ops = {}
        for name, op in linear_ops.items():
            ops[name] = self.get_op_in_H_eigenbasis(op)

        return ops


def get_op_in_new_basis(op: Qarray, evecs: Array, dims: List[List[int]]) -> Qarray:
    data = get_op_data_in_new_basis(op.data, evecs)
    return Qarray.create(data, dims=dims)


def get_op_data_in_new_basis(op_data: Array, evecs: Array) -> Array:
    return jnp.dot(jnp.conjugate(evecs.transpose()), jnp.dot(op_data, evecs))


def get_vec_in_new_basis(vec: Qarray, evecs: Array, dims: List[List[int]]) -> Qarray:
    return Qarray.create(get_vec_data_in_new_basis(vec.data, evecs), dims=dims)


def get_vec_data_in_new_basis(vec_data: Array, evecs: Array) -> Array:
    return jnp.dot(jnp.conjugate(evecs.transpose()), vec_data)

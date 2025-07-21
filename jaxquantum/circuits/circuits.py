"""Circuits.

Inspired by a mix of Cirq and Qiskit circuits.
"""

from flax import struct
from jax import config
from typing import List, Optional, Union
from copy import deepcopy
from numpy import argsort
import jax.numpy as jnp

from jaxquantum.core.operators import identity
from jaxquantum.circuits.gates import Gate
from jaxquantum.circuits.constants import SimulateMode
from jaxquantum.core.qarray import Qarray, concatenate


config.update("jax_enable_x64", True)


@struct.dataclass
class Register:
    dims: List[int] = struct.field(pytree_node=False)

    @classmethod
    def create(cls, dims: List[int]):
        return Register(dims=dims)

    def __eq__(self, other):
        if isinstance(other, Register):
            return self.dims == other.dims
        return False


@struct.dataclass
class Operation:
    gate: Gate
    indices: List[int] = struct.field(pytree_node=False)
    register: Register

    @classmethod
    def create(cls, gate: Gate, indices: Union[int, List[int]], register: Register):
        if isinstance(indices, int):
            indices = [indices]

        assert gate.num_modes == len(indices), (
            "Number of indices must match gate's num_modes."
        )
        assert gate.dims == [register.dims[ind] for ind in indices], (
            "Indices must match register dimensions."
        )

        if any(
            (0 > ind and ind >= len(register.dims)) or not isinstance(ind, int)
            for ind in indices
        ):
            raise ValueError("Indices must be integers within the register.")

        return Operation(gate=gate, indices=indices, register=register)


    def promote(self, op: Qarray) -> Qarray:
        indices_order = self.indices
        missing_indices = [
            i for i in range(len(self.register.dims)) if i not in indices_order
        ]
        for j in missing_indices:
            op = op ^ identity(self.register.dims[j])
        combined_indices = indices_order + missing_indices
        sorted_ind = list(argsort(combined_indices))
        op = op.transpose(sorted_ind)
        return op


@struct.dataclass
class Layer:
    operations: List[Operation] = struct.field(pytree_node=False)
    _unique_indices: List[int] = struct.field(pytree_node=False)
    _default_simulate_mode: SimulateMode = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls, operations: List[Operation], default_simulate_mode=SimulateMode.UNITARY
    ):
        all_indices = [ind for op in operations for ind in op.indices]
        unique_indices = list(set(all_indices))

        if default_simulate_mode != SimulateMode.HAMILTONIAN:
            if len(all_indices) != len(unique_indices):
                raise ValueError("Operations must not have overlapping indices.")

        return Layer(
            operations=operations,
            _unique_indices=unique_indices,
            _default_simulate_mode=default_simulate_mode,
        )

    def add(self, operation: Operation):
        if self._default_simulate_mode != SimulateMode.HAMILTONIAN:
            if any(ind in self._unique_indices for ind in operation.indices):
                raise ValueError("Operations must not have overlapping indices.")
        self.operations.append(operation)
        self._unique_indices.extend(operation.indices)

    def gen_U(self):
        U = None

        if len(self.operations) == 0:
            return None
            
        indices_order = []
        for operation in self.operations:
            indices_order += operation.indices

            if U is None:
                U = operation.gate.U
            else:
                U = U ^ operation.gate.U

        register = self.operations[0].register
        missing_indices = [
            i for i in range(len(register.dims)) if i not in indices_order
        ]

        for j in missing_indices:
            U = U ^ identity(register.dims[j])

        combined_indices = indices_order + missing_indices

        sorted_ind = list(argsort(combined_indices))
        U = U.transpose(sorted_ind)
        return U

    def gen_Ht(self):
        Ht = lambda t: 0

        if len(self.operations) == 0:
            return Ht
        
        for operation in self.operations:
            def Ht(t, prev_Ht=Ht, prev_operation=operation):
                return prev_Ht(t) + prev_operation.promote(prev_operation.gate.Ht(t))
        
        return Ht

    def gen_KM(self):
        KM_label = "KM"
        KM = Qarray.from_list([])

        if len(self.operations) == 0:
            return KM

        indices_order = []
        for operation in self.operations:
            if len(getattr(operation.gate, KM_label)) == 0:
                continue

            indices_order += operation.indices

            if len(KM) == 0:
                KM = deepcopy(getattr(operation.gate, KM_label))
            else:
                KM = KM ^ getattr(operation.gate, KM_label)

        if len(KM) == 0:
            return KM

        register = self.operations[0].register
        missing_indices = [
            i for i in range(len(register.dims)) if i not in indices_order
        ]

        for j in missing_indices:
            KM = KM ^ identity(register.dims[j])

        combined_indices = indices_order + missing_indices
        sorted_ind = list(argsort(combined_indices))

        KM = KM.transpose(sorted_ind)

        return KM

    def gen_c_ops(self):
        c_ops = Qarray.from_list([])

        if len(self.operations) == 0:
            return c_ops

        for operation in self.operations:
            if len(operation.gate.c_ops) == 0:
                continue
            promoted_c_ops = operation.promote(operation.gate.c_ops)
            c_ops = concatenate([c_ops, promoted_c_ops])

        return c_ops

    def gen_ts(self):
        ts = None

        for operation in self.operations:
            if operation.gate.ts is not None and len(operation.gate.ts) > 0:
                if ts is None:
                    ts = operation.gate.ts
                else:
                    assert jnp.array_equal(ts, operation.gate.ts), (
                        "All operations in a layer must have the same specified time steps, but not all operations need to have time steps."
                    )
        return ts
        
@struct.dataclass
class Circuit:
    register: Register
    layers: List[Layer] = struct.field(pytree_node=False)

    @classmethod
    def create(cls, register: Register, layers: Optional[List[Layer]] = None):
        if layers is None:
            layers = []

        return Circuit(
            register=register,
            layers=layers,
        )

    def append_layer(self, layer: Layer):
        self.layers.append(layer)

    def append_operation(
        self, operation: Operation, default_simulate_mode: Optional[SimulateMode] = None, new_layer: bool =True
    ):
        assert operation.register == self.register, (
            f"Mismatch in operation register {operation.register} and circuit register {self.register}."
        )

        new_layer = new_layer or len(self.layers) == 0

        if new_layer:
            default_simulate_mode = default_simulate_mode if default_simulate_mode is not None else SimulateMode.UNITARY
            self.append_layer(
                Layer.create([operation], default_simulate_mode=default_simulate_mode)
            )
        else:
            if default_simulate_mode is not None:
                assert (
                    self.layers[-1]._default_simulate_mode == default_simulate_mode
                ), "Cannot append operation to last layer with different default simulate mode."

            self.layers[-1].add(operation)

    def append(
        self,
        gate: Gate,
        indices: Union[int, List[int]],
        default_simulate_mode: Optional[SimulateMode] = None,
        new_layer: bool = True,
    ):
        operation = Operation.create(gate, indices, self.register)
        self.append_operation(operation, default_simulate_mode=default_simulate_mode, new_layer=new_layer)

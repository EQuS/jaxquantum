""" Circuits. 

Inspired by a mix of Cirq and Qiskit circuits.
"""

from flax import struct
from jax import config
from typing import List, Optional, Union
from copy import deepcopy
from numpy import argsort

from jaxquantum.core.operators import identity
from jaxquantum.circuits.gates import Gate
from jaxquantum.circuits.constants import SimulateMode

config.update("jax_enable_x64", True)

@struct.dataclass
class Register:
    dims: List[int] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls, dims: List[int]
    ):
        return Register(
            dims = dims
        )

    def __eq__(
        self, other
    ):
        if isinstance(other, Register):
            return self.dims == other.dims
        return False

@struct.dataclass
class Operation:
    gate: Gate
    indices: List[int] = struct.field(pytree_node=False)
    register: Register

    @classmethod
    def create(
        cls,
        gate: Gate,
        indices: Union[int,List[int]],
        register: Register
    ):
        if isinstance(indices, int):
            indices = [indices]

        assert gate.num_modes == len(indices), "Number of indices must match gate's num_modes."
        assert gate.dims == [register.dims[ind] for ind in indices], "Indices must match register dimensions."

        if any(
            (0 > ind and ind >= len(register.dims)) or not isinstance(ind, int) for ind in indices):
            raise ValueError("Indices must be integers within the register.")

        return Operation(
            gate = gate,
            indices = indices,
            register = register
        )

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
        if len(all_indices) != len(unique_indices):
            raise ValueError("Operations must not have overlapping indices.")
        return Layer(
            operations = operations,
            _unique_indices = unique_indices,
            _default_simulate_mode = default_simulate_mode
        )

    def add(self, operation: Operation):
        if any(ind in self._unique_indices for ind in operation.indices):
            raise ValueError("Operations must not have overlapping indices.")
        self.operations.append(operation)
        self._unique_indices.extend(operation.indices)

    def gen_U(self):
        U = None

        if len(self.operations) == 0:
            return U
        
        indices_order = []
        for operation in self.operations:
            indices_order += operation.indices

            if U is None:
                U = operation.gate.U
            else:
                U = U ^ operation.gate.U
            
        register = self.operations[0].register
        missing_indices = [i for i in range(len(register.dims)) if i not in indices_order]

        for j in missing_indices:
            U = U ^ identity(register.dims[j])

        combined_indices = indices_order + missing_indices

        sorted_ind = list(argsort(combined_indices))
        U = U.transpose(sorted_ind)
        return U

    def gen_KM(self):
        KM = []

        if len(self.operations) == 0:
            return KM

        
        indices_order = []
        for operation in self.operations:
            indices_order += operation.indices

            if len(KM) == 0:
                KM = deepcopy(operation.gate.KM)
            else:
                # updated_KM = []                
                # for op1 in KM:
                #     for op2 in operation.gate.KM:
                #         updated_KM.append(op1^op2)
                # KM = updated_KM
                KM = KM.arraytensor(operation.gate.KM)
        
        register = self.operations[0].register
        missing_indices = [i for i in range(len(register.dims)) if i not in indices_order]
        
        for j in missing_indices:
            KM = KM ^ identity(register.dims[j])
        
        combined_indices = indices_order + missing_indices
        sorted_ind = list(argsort(combined_indices))
                
        KM = KM.transpose(sorted_ind)

        return KM


@struct.dataclass
class Circuit:
    register: Register
    layers: List[Layer]  = struct.field(pytree_node=False)
    

    @classmethod
    def create(
        cls, register: Register, layers: Optional[List[Layer]] = None
    ):  
        if layers is None:
            layers = []

        return Circuit(
            register = register,
            layers = layers,   
        )

    def append_layer(self, layer: Layer):
        self.layers.append(layer)

    def append_operation(self, operation: Operation, default_simulate_mode=SimulateMode.UNITARY):
        assert operation.register == self.register, (f"Mismatch in operation register {operation.register} and circuit register {self.register}.")
        self.append_layer(Layer.create([operation], default_simulate_mode=default_simulate_mode))

    def append(self, gate: Gate, indices: Union[int, List[int]], default_simulate_mode=SimulateMode.UNITARY):
        operation = Operation.create(gate, indices, self.register)
        self.append_operation(operation, default_simulate_mode=default_simulate_mode)
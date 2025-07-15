"""Gates."""

from flax import struct
from jax import Array, config
from typing import List, Dict, Any, Optional, Callable, Union
import jax.numpy as jnp


from jaxquantum.core.qarray import Qarray

config.update("jax_enable_x64", True)


@struct.dataclass
class Gate:
    dims: List[int] = struct.field(pytree_node=False)
    _U: Optional[Array] # Unitary
    _Ht: Optional[Array] # Hamiltonian
    _KM: Optional[Qarray] # Kraus map
    _params: Dict[str, Any]
    _ts: Array
    _name: str = struct.field(pytree_node=False)
    num_modes: int = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        dims: Union[int, List[int]],
        name: str = "Gate",
        params: Optional[Dict[str, Any]] = None,
        ts: Optional[Array] = None,
        gen_U: Optional[Callable[[Dict[str, Any]], Qarray]] = None,
        gen_Ht: Optional[Callable[[Dict[str, Any]], Qarray]] = None,
        gen_KM: Optional[Callable[[Dict[str, Any]], List[Qarray]]] = None,
        num_modes: int = 1,
    ):
        """Create a gate.

        Args:
            dims: Dimensions of the gate.
            name: Name of the gate.
            params: Parameters of the gate.
            ts: Times of the gate.
            gen_U: Function to generate the unitary of the gate.
            gen_Ht: Function to generate a function Ht(t) that takes in a time t and outputs a Hamiltonian Qarray.
            gen_KM: Function to generate the Kraus map of the gate.
            num_modes: Number of modes of the gate.
        """

        # TODO: add params to device?

        if isinstance(dims, int):
            dims = [dims]

        assert len(dims) == num_modes, (
            "Number of dimensions must match number of modes."
        )

        # Unitary
        _U = gen_U(params) if gen_U is not None else None 
        _Ht = gen_Ht(params) if gen_H is not None else None 

        if gen_KM is not None:
            _KM = gen_KM(params)
        elif _U is not None:
            _KM = Qarray.from_list([_U])

        return Gate(
            dims = dims,
            _U = _U,
            _Ht = _Ht,
            _KM = _KM,
            _params = params if params is not None else {},
            _ts=ts if ts is not None else jnp.array([]),
            _name=name,
            num_modes=num_modes,
        )

    def __str__(self):
        return self._name

    def __repr__(self):
        return self._name

    @property
    def U(self):
        return self._U

    @property
    def Ht(self):
        return self._Ht

    @property
    def KM(self):
        return self._KM

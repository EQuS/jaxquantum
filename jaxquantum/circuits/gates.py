"""Gates."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import jax.numpy as jnp
from flax import struct
from jax import Array, config

from jaxquantum.core.qarray import Qarray, concatenate

config.update("jax_enable_x64", True)


@struct.dataclass
class Gate:
    dims: list[int] = struct.field(pytree_node=False)
    _U: Array | None  # Unitary
    _Ht: Array | None  # Hamiltonian
    _KM: Qarray | None  # Kraus map
    _c_ops: Qarray | None
    _params: dict[str, Any]
    _ts: Array
    _name: str = struct.field(pytree_node=False)
    num_modes: int = struct.field(pytree_node=False)
    _gen_KM: Callable | None = struct.field(pytree_node=False, default=None)
    _channel_apply: Callable | None = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        dims: int | list[int],
        name: str = "Gate",
        params: dict[str, Any] | None = None,
        ts: Array | None = None,
        gen_U: Callable[[dict[str, Any]], Qarray] | None = None,
        gen_Ht: Callable[[dict[str, Any]], Qarray] | None = None,
        gen_c_ops: Callable[[dict[str, Any]], Qarray] | None = None,
        gen_KM: Callable[[dict[str, Any]], list[Qarray]] | None = None,
        channel_apply: Callable[[Array, dict[str, Any]], Array] | None = None,
        lazy_kraus: bool = False,
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
            channel_apply: Optional direct density-matrix channel kernel.
            lazy_kraus: Generate Kraus operators only when ``KM`` is accessed.
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
        _Ht = gen_Ht(params) if gen_Ht is not None else None
        _c_ops = gen_c_ops(params) if gen_c_ops is not None else Qarray.from_list([])

        _KM = gen_KM(params) if gen_KM is not None and not lazy_kraus else None

        return Gate(
            dims=dims,
            _U=_U,
            _Ht=_Ht,
            _KM=_KM,
            _c_ops=_c_ops,
            _gen_KM=gen_KM if lazy_kraus else None,
            _channel_apply=channel_apply,
            _params=params if params is not None else {},
            _ts=ts if ts is not None else jnp.array([]),
            _name=name,
            num_modes=num_modes,
        )

    def __str__(self):
        return self._name

    def __repr__(self):
        return self._name

    @property
    def name(self):
        return self._name

    @property
    def U(self):
        return self._U

    @property
    def Ht(self):
        return self._Ht

    @property
    def KM(self):
        if self._KM is not None:
            return self._KM
        if self._gen_KM is not None:
            return self._gen_KM(self.params)
        if self._U is None:
            return Qarray.from_list([])
        impl = type(self._U._impl).from_data(self._U.data[None])
        return Qarray._from_impl(impl, self._U._qdims)

    @property
    def c_ops(self):
        return self._c_ops

    @property
    def channel_apply(self):
        """Direct density-matrix kernel, if defined."""
        return self._channel_apply

    @property
    def params(self):
        return self._params

    @property
    def ts(self):
        return self._ts

    def add_Ht(self, Ht: Callable[[float], Qarray]):
        """Add a Hamiltonian function to the gate."""

        def new_Ht(t):
            return Ht(t) + self.Ht(t) if self.Ht is not None else Ht(t)

        return Gate(
            dims=self.dims,
            _U=self.U,
            _Ht=new_Ht,
            _KM=self._KM,
            _c_ops=self.c_ops,
            _gen_KM=self._gen_KM,
            _channel_apply=self._channel_apply,
            _params=self.params,
            _ts=self.ts,
            _name=self.name,
            num_modes=self.num_modes,
        )

    def add_c_ops(self, c_ops: Qarray):
        """Add a c_ops to the gate."""
        return Gate(
            dims=self.dims,
            _U=self.U,
            _Ht=self.Ht,
            _KM=self._KM,
            _c_ops=concatenate([self.c_ops, c_ops]),
            _gen_KM=self._gen_KM,
            _channel_apply=self._channel_apply,
            _params=self.params,
            _ts=self.ts,
            _name=self.name,
            num_modes=self.num_modes,
        )

    def copy(self):
        """Return a copy of the gate."""
        return Gate(
            dims=deepcopy(self.dims),
            _U=self.U,
            _Ht=deepcopy(self.Ht),
            _KM=self._KM,
            _c_ops=self.c_ops,
            _gen_KM=self._gen_KM,
            _channel_apply=self._channel_apply,
            _params=deepcopy(self.params),
            _ts=self.ts,
            _name=self.name,
            num_modes=self.num_modes,
        )

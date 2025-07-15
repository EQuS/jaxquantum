"""Constants."""

from enum import Enum


class SimulateMode(str, Enum):
    UNITARY = "unitary"
    KRAUS = "kraus"
    HAMILTONIAN = "hamiltonian"
    DEFAULT = "default"

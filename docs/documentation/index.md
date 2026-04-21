# Documentation

Welcome to the jaxquantum documentation.

jaxquantum is a JAX-native quantum simulation toolkit for superconducting circuit devices and bosonic qubits. It serves as a drop-in QuTiP replacement while adding GPU/TPU acceleration, `jax.vmap` for parameter sweeps, `jax.grad` for differentiable physics, and `jax.jit` for compiled simulation loops.

## Getting Started

New to jaxquantum? Start here:

1. [**Installation**](getting_started/installation.md) — install via pip or from source
2. [**Qarray**](getting_started/qarray.ipynb) — the core quantum array type: kets, operators, batch dimensions
3. [**The Sharp Bits**](getting_started/sharp_bits.md) — common JAX pitfalls to avoid

## Tutorials

Hands-on notebooks spanning the library:

| Tutorial | What you'll learn |
|---|---|
| [Devices & Systems](tutorials/devices.ipynb) | Transmon spectroscopy, flux sweeps with `vmap`, parameter fitting with `grad` |
| [Bosonic Codes](tutorials/bosonic_codes.ipynb) | Cat, GKP, and Binomial qubit encodings; phase-space visualization; logical gates |
| [Circuits](tutorials/circuits.ipynb) | Gate-based circuits; unitary, Hamiltonian, and Kraus simulation modes; circuit optimization |
| [Sparse Backends](tutorials/sparse_backends.ipynb) | SparseDIA and BCOO formats; when and how to use sparse operators; performance comparison |

## Code Reference

Auto-generated API documentation for all modules: [Reference](../reference/)

> **Energy Number Restricted (ENR) basis** — documentation coming soon.

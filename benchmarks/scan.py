"""Benchmarks for compiled-loop paths."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

import jaxquantum as jqt
from jaxquantum.core.qp_distributions import (
    _qfunc_coherent_grid,
    _qfunc_iterative_single,
)


def _qfunc_runner(dimension, points, mode):
    coordinates = jnp.linspace(-4.0, 4.0, points)
    grid, prefactor = _qfunc_coherent_grid(coordinates, coordinates, 2.0)

    def run(rho):
        values, vectors = jnp.linalg.eigh(rho)
        vectors = vectors.T

        def component(index):
            return values[index] * _qfunc_iterative_single(
                vectors[index],
                grid,
                prefactor,
                2.0,
            )

        if mode == "loop":
            out = jax.lax.fori_loop(
                0,
                dimension,
                lambda index, total: total + component(index),
                jnp.zeros_like(grid.real),
            )
        elif mode == "chunked":
            chunk_size = min(8, dimension)
            chunks = (dimension + chunk_size - 1) // chunk_size
            padding = chunks * chunk_size - dimension
            padded_values = jnp.pad(values, (0, padding))
            padded_vectors = jnp.pad(vectors, ((0, padding), (0, 0)))

            def add_chunk(index, total):
                start = index * chunk_size
                chunk_values = jax.lax.dynamic_slice_in_dim(
                    padded_values, start, chunk_size
                )
                chunk_vectors = jax.lax.dynamic_slice_in_dim(
                    padded_vectors, start, chunk_size
                )
                contributions = jax.vmap(
                    lambda vector: _qfunc_iterative_single(vector, grid, prefactor, 2.0)
                )(chunk_vectors)
                return total + jnp.tensordot(
                    chunk_values,
                    contributions,
                    axes=1,
                )

            out = jax.lax.fori_loop(
                0,
                chunks,
                add_chunk,
                jnp.zeros_like(grid.real),
            )
        elif mode == "vectorized":
            contributions = jax.vmap(
                lambda vector: _qfunc_iterative_single(vector, grid, prefactor, 2.0)
            )(vectors)
            out = jnp.tensordot(values, contributions, axes=1)
        else:
            out = component(0)
            for index in range(1, dimension):
                out += component(index)
        return out / jnp.pi

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--points", type=int, default=81)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()
    ket = (
        jqt.coherent(args.dimension, 1.4) + 0.7j * jqt.coherent(args.dimension, -1.2)
    ).unit()
    rho = ket.to_dm().data

    reports = {
        f"qfunc_{mode}": jqt.benchmark_jax_function(
            _qfunc_runner(args.dimension, args.points, mode),
            rho,
            iterations=args.iterations,
        )
        for mode in ("unrolled", "loop", "chunked", "vectorized")
    }
    payload = {
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxquantum": jqt.__version__,
        "jaxquantum_source": str(Path(jqt.__file__).resolve()),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "parameters": {
            "dimension": args.dimension,
            "points": args.points,
            "iterations": args.iterations,
        },
        "results": reports,
    }
    output = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

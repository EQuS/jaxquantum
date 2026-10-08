"""Bose-Hubbard cuQuantum benchmark for one or more physical GPUs.

Run with one MPI rank per GPU, for example::

    mpirun -n 2 python -m benchmarks.cuquantum_multigpu --output two.json

The cuQuantum tutorial imports the model and timing function from this module.
"""

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path


def bose_hubbard_chain(
    impl, n_sites=4, d=4, J=1.0, U=0.5, mu=-0.2, gamma=0.05
):
    """Build a 1D Bose-Hubbard Hamiltonian and decay on site zero."""
    import jax.numpy as jnp
    import jaxquantum as jqt

    a_local = jqt.destroy(d, implementation=impl)
    n_local = jqt.num(d, implementation=impl)
    identity = jqt.identity(d, implementation=impl)

    def at(site, op):
        ops = [identity] * n_sites
        ops[site] = op
        return jqt.tensor(*ops)

    hamiltonian = 0.0 * at(0, identity)
    for site in range(n_sites - 1):
        hopping = at(site, a_local.dag()) @ at(site + 1, a_local)
        hamiltonian = hamiltonian - J * (hopping + hopping.dag())
    for site in range(n_sites):
        occupation = at(site, n_local)
        hamiltonian = hamiltonian + (
            (U / 2.0) * occupation @ (occupation - at(site, identity))
            + mu * occupation
        )
    collapse = jnp.sqrt(gamma) * at(0, a_local)
    return hamiltonian, collapse


def time_mesolve(
    hamiltonian, rho0, tlist, c_ops, *, n_runs=3, label="", comm=None,
    return_result=False,
):
    """Time the first JIT call and ``n_runs`` warmed calls, blocking on results.

    With ``comm``, synchronize ranks and report the slowest rank's wall time.
    """
    import jax
    import jaxquantum as jqt

    options = jqt.SolverOptions.create(
        progress_meter=False, solver="Euler", stepsize_controller="ConstantStepSize"
    )
    solve = jax.jit(partial(jqt.mesolve, solver_options=options))
    times = []
    result = None
    for _ in range(n_runs + 1):
        if comm is not None:
            comm.Barrier()
        start = time.perf_counter()
        result = solve(hamiltonian, rho0, tlist, c_ops=c_ops)
        result.data.block_until_ready()
        elapsed = time.perf_counter() - start
        if comm is not None:
            from mpi4py import MPI

            elapsed = comm.allreduce(elapsed, op=MPI.MAX)
        times.append(elapsed)
    if return_result:
        return times, result
    return times


def main():
    """Run the same tutorial workload with one MPI rank per visible GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import jax
    from mpi4py import MPI

    jax.config.update("jax_enable_x64", True)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ranks = comm.Get_size()
    if ranks > 1:
        jax.distributed.initialize(cluster_detection_method="mpi4py")

    import jax.numpy as jnp
    import jaxquantum as jqt

    if ranks > 1:
        from cuquantum.densitymat.jax import set_communicator

        set_communicator(comm, provider="MPI")

    if jax.local_device_count() != 1 or jax.device_count() != ranks:
        raise RuntimeError("Expected one visible GPU per MPI rank")

    report = {"ranks": ranks, "d": args.d, "jax": jax.__version__,
              "devices": [str(device) for device in jax.devices("gpu")], "cases": []}
    tlist = jnp.linspace(0.0, 4.0, 41)
    for n_sites in args.sites:
        hamiltonian, collapse = bose_hubbard_chain("cuquantum", n_sites, args.d)
        rho0 = jqt.basis_like(hamiltonian, [1] + [0] * (n_sites - 1)).to_dm()
        times, result = time_mesolve(
            hamiltonian, rho0, tlist, [collapse], n_runs=args.repeats,
            comm=comm, return_result=True,
        )
        final = result.data[-1]
        trace = float(jnp.real(jnp.trace(final)).block_until_ready())
        initial_index = args.d ** (n_sites - 1)
        initial_population = float(
            jnp.real(final[initial_index, initial_index]).block_until_ready()
        )
        if abs(trace - 1.0) > 2e-3:
            raise AssertionError(f"Unexpected trace at {n_sites} sites: {trace}")
        case = {
            "sites": n_sites,
            "dim": args.d ** n_sites,
            "first_s": times[0],
            "warm_s": times[1:],
            "warm_median_s": statistics.median(times[1:]),
            "trace": trace,
            "initial_population": initial_population,
        }
        if rank == 0:
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
        comm.Barrier()
    if rank == 0:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

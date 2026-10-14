"""Bose-Hubbard dense and cuQuantum benchmarks on physical GPUs.

cuQuantum uses one MPI rank per GPU, for example::

    mpirun -n 2 python -m benchmarks.cuquantum_multigpu --output two.json

The cuQuantum tutorial imports the model and timing function from this module.
"""

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path


def bose_hubbard_chain(impl, n_sites=4, d=4, J=1.0, U=0.5, mu=-0.2, gamma=0.05):
    """Build a 1D Bose-Hubbard Hamiltonian and decay on site zero."""
    import jax.numpy as jnp

    import jaxquantum as jqt

    a_local = jqt.destroy(d, implementation=impl)
    n_local = jqt.num(d, implementation=impl)
    identity = jqt.identity(d, implementation=impl)

    def at(changes):
        ops = [identity] * n_sites
        for site, operator in changes.items():
            ops[site] = operator
        return jqt.tensor(*ops)

    # Form products locally. Multiplying full embedded operators creates
    # unnecessarily large dense intermediates and cuQuantum expressions.
    onsite_local = (U / 2.0) * (n_local @ (n_local - identity)) + mu * n_local
    hamiltonian = None
    for site in range(n_sites - 1):
        hopping = at({site: a_local.dag(), site + 1: a_local})
        term = -J * (hopping + hopping.dag())
        hamiltonian = term if hamiltonian is None else hamiltonian + term
    for site in range(n_sites):
        hamiltonian = hamiltonian + at({site: onsite_local})
    collapse = jnp.sqrt(gamma) * at({0: a_local})
    return hamiltonian, collapse


def _time_solve(
    solve_fn,
    hamiltonian,
    state0,
    tlist,
    *,
    c_ops=None,
    n_runs=3,
    label="",
    comm=None,
    return_result=False,
    final_only=False,
    solver="Euler",
):
    """Time one compiling call and warmed calls, synchronizing MPI ranks."""
    import diffrax
    import jax
    import jax.numpy as jnp

    import jaxquantum as jqt

    solver_class = {"Euler": diffrax.Euler, "Tsit5": diffrax.Tsit5}[solver]
    options = jqt.SolverOptions(
        progress_meter=None,
        solver=solver_class(),
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    kwargs = {"solver_options": options}
    if final_only:
        kwargs["saveat_tlist"] = jnp.array([])
    solve = jax.jit(partial(solve_fn, **kwargs))
    times = []
    result = None
    for _ in range(n_runs + 1):
        if comm is not None:
            comm.Barrier()
        start = time.perf_counter()
        if c_ops is None:
            result = solve(hamiltonian, state0, tlist)
        else:
            result = solve(hamiltonian, state0, tlist, c_ops=c_ops)
        result.data.block_until_ready()
        elapsed = time.perf_counter() - start
        if comm is not None:
            from mpi4py import MPI

            elapsed = comm.allreduce(elapsed, op=MPI.MAX)
        times.append(elapsed)
    if return_result:
        return times, result
    return times


def time_mesolve(
    hamiltonian,
    rho0,
    tlist,
    c_ops,
    *,
    n_runs=3,
    label="",
    comm=None,
    return_result=False,
    final_only=False,
    solver="Euler",
):
    """Time JIT-compiled master-equation solves; first call includes compilation."""
    import jaxquantum as jqt

    return _time_solve(
        jqt.mesolve,
        hamiltonian,
        rho0,
        tlist,
        c_ops=c_ops,
        n_runs=n_runs,
        label=label,
        comm=comm,
        return_result=return_result,
        final_only=final_only,
        solver=solver,
    )


def time_sesolve(
    hamiltonian,
    psi0,
    tlist,
    *,
    n_runs=3,
    label="",
    comm=None,
    return_result=False,
    final_only=False,
    solver="Tsit5",
):
    """Time JIT-compiled statevector solves; first call includes compilation."""
    import jaxquantum as jqt

    return _time_solve(
        jqt.sesolve,
        hamiltonian,
        psi0,
        tlist,
        n_runs=n_runs,
        label=label,
        comm=comm,
        return_result=return_result,
        final_only=final_only,
        solver=solver,
    )


def main():
    """Run the tutorial workload with dense sharding or cuQuantum MPI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--solver", choices=("mesolve", "sesolve"), default="mesolve")
    parser.add_argument(
        "--backend", choices=("dense", "cuquantum"), default="cuquantum"
    )
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--tfinal", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import jax
    from mpi4py import MPI

    jax.config.update("jax_enable_x64", True)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ranks = comm.Get_size()
    if args.backend == "cuquantum" and ranks > 1:
        jax.distributed.initialize(cluster_detection_method="mpi4py")
    elif args.backend == "dense" and ranks != 1:
        raise RuntimeError("Dense sharding uses one process and a JAX device mesh")

    import jax.numpy as jnp

    import jaxquantum as jqt

    if args.backend == "cuquantum" and ranks > 1:
        from cuquantum.densitymat.jax import set_communicator

        set_communicator(comm, provider="MPI")

    jqt.clear_default_sharding()
    if args.backend == "dense":
        if jax.local_device_count() != jax.device_count():
            raise RuntimeError("Dense sharding requires one process with local GPUs")
        if jax.device_count() > 1:
            jqt.set_device_mesh(
                shape=(jax.device_count(),),
                axis_names=("mp",),
                devices=jax.devices("gpu"),
            )
    elif jax.local_device_count() != 1 or jax.device_count() != ranks:
        raise RuntimeError("Expected one visible GPU per MPI rank")

    report = {
        "ranks": ranks,
        "d": args.d,
        "solver": args.solver,
        "backend": args.backend,
        "gpus": jax.device_count(),
        "final_only": args.final_only,
        "steps": args.steps,
        "tfinal": args.tfinal,
        "jax": jax.__version__,
        "devices": [str(device) for device in jax.devices("gpu")],
        "cases": [],
    }
    if args.steps < 1:
        parser.error("--steps must be positive")
    tlist = jnp.linspace(0.0, args.tfinal, args.steps + 1)
    for n_sites in args.sites:
        hamiltonian, collapse = bose_hubbard_chain(args.backend, n_sites, args.d)
        psi0 = jqt.basis_like(hamiltonian, [1] + [0] * (n_sites - 1))
        initial_index = args.d ** (n_sites - 1)
        if args.solver == "mesolve":
            rho0 = psi0.to_dm()
            times, result = time_mesolve(
                hamiltonian,
                rho0,
                tlist,
                jqt.Qarray.from_list([collapse])
                if args.backend == "dense"
                else [collapse],
                n_runs=args.repeats,
                comm=comm,
                return_result=True,
                final_only=args.final_only,
            )
            final = result.data[-1]
            conserved = float(jnp.real(jnp.trace(final)).block_until_ready())
            population = float(
                jnp.real(final[initial_index, initial_index]).block_until_ready()
            )
            conserved_name = "trace"
            if abs(conserved - 1.0) > 2e-3:
                raise AssertionError(
                    f"Unexpected trace at {n_sites} sites: {conserved}"
                )
        else:
            times, result = time_sesolve(
                hamiltonian,
                psi0,
                tlist,
                n_runs=args.repeats,
                comm=comm,
                return_result=True,
                final_only=args.final_only,
            )
            final = result.data[-1]
            conserved = float(jnp.real(jnp.vdot(final, final)).block_until_ready())
            population = float(jnp.abs(final[initial_index]).block_until_ready() ** 2)
            conserved_name = "norm"
            if abs(conserved - 1.0) > 2e-3:
                raise AssertionError(f"Unexpected norm at {n_sites} sites: {conserved}")
        case = {
            "sites": n_sites,
            "dim": args.d**n_sites,
            "first_s": times[0],
            "warm_s": times[1:],
            "warm_median_s": statistics.median(times[1:]),
            conserved_name: conserved,
            "initial_population": population,
            "input_sharding": str(
                rho0.data.sharding if args.solver == "mesolve" else psi0.data.sharding
            ),
            "result_sharding": str(result.data.sharding),
            "result_local_shapes": [
                tuple(shard.data.shape) for shard in result.data.addressable_shards
            ],
        }
        if rank == 0:
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
        comm.Barrier()
    if rank == 0:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

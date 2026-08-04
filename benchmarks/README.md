# Performance benchmarks

The benchmark scripts synchronize device work before recording a sample.

```bash
python benchmarks/performance.py
python benchmarks/roofline.py
python benchmarks/channels.py
python benchmarks/hamiltonian.py
python benchmarks/construction.py
python benchmarks/devices.py
python benchmarks/scan.py
python benchmarks/cat_sbs.py path/to/cat_sbs.py
python benchmarks/sbs_device.py --code gkp
python benchmarks/sbs_budget_sweep.py
python benchmarks/gkp_sbs_compare.py path/to/sbs_accelerated_blocked.py
python benchmarks/cat_control_sweep.py --parameters measured
python benchmarks/gkp_error_budget.py --output-dir path/to/archive
```

Use `--help` for sizes, iteration counts, and JSON output. For a shared display
GPU, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` before starting Python.

`sbs_device.py` profiles cold/warm execution, compiler memory, HLO, accuracy,
and optionally an error budget. The sweep scripts record the cat-size budget
and cross-check control points against either colleague implementation.
`gkp_error_budget.py` saves exact inputs, full curves, fits, and an analysis
figure; named experimental inputs live in `experiments/circuit/sbs_parameters.py`.

See the [extended CPU/GPU results](results/performance_summary.md) for the
additional channel, loop, simulation, device, and cat-sBs measurements.

## Reference result

Measured on Windows 11 / WSL2 Ubuntu 24.04, RTX 4080 Super, driver 595.95,
JAX 0.11.0, and Python 3.12. Exact data is in
[`results/rtx4080_super_wsl_jax_0.11.0.json`](results/rtx4080_super_wsl_jax_0.11.0.json).
The CPU before/after data is in
[`results/windows_cpu_jax_0.11.0.json`](results/windows_cpu_jax_0.11.0.json).

The complex128 empirical roof is about 0.96 TFLOP/s median, while the float64
stream kernel sustains about 340 GB/s. A 6144-dimensional dense complex128
Qarray matrix-vector product reaches about 460 GB/s effective bandwidth. The
20-qubit, 64-gate circuit reaches only about 24 GB/s, indicating that small
gate launches and layout changes dominate rather than memory bandwidth.

On CPU, the optimized 8-qubit, 20-gate circuit reduced compiled HLO text from
1,165,204 to 215,649 characters, compile time from 0.576 s to 0.107 s, and
warmed-up time from 9.37 ms to 0.046 ms. Treat absolute times as
machine-specific; HLO size and the before/after ratios are more portable.

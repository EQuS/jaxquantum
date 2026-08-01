# Performance benchmarks

Both scripts synchronize device work before recording a sample.

```bash
python benchmarks/performance.py
python benchmarks/roofline.py
```

Use `--help` for sizes, iteration counts, and JSON output. For a shared display
GPU, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` before starting Python.

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

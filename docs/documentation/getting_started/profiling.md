# Profiling JAX code

JAX execution is asynchronous, so ordinary wall-clock timing can measure only
dispatch. Jaxquantum's profiling helpers synchronize every output leaf and
separate lowering, compilation, first execution, and warmed execution:

```python
import jax.numpy as jnp
import jaxquantum as jqt

state = jqt.coherent(32, 1.2)

def step(psi):
    return (jqt.displace(32, 0.05) @ psi).data

report = jqt.benchmark_jax_function(step, state, iterations=25)
print(report["timings_s"])
print(report["memory_bytes"])
```

The aggregate report includes:

- synchronized cold and warmed timing distributions;
- XLA argument, output, temporary, peak, and code-size estimates;
- device allocator snapshots when the backend exposes them;
- compiler cost analysis and StableHLO size.

Set `include_hlo=True` to include the full StableHLO text. For individual
queries, use `jax_hlo`, `lower_jax_function`, `jax_memory_stats`, or
`jax_device_memory_stats`.

## Precision comparison

`compare_precision=True` runs both float64/complex128 and float32/complex64,
then reports accuracy, speed, and memory changes:

```python
report = jqt.benchmark_jax_function(
    step,
    state,
    compare_precision=True,
    iterations=25,
)
print(report["accuracy"])
print(report["single_vs_double"])
```

Both modes are tested regardless of the current `jax_enable_x64` setting. The
original setting is restored even if profiling raises an exception. Because
this setting is process-global, do not change it concurrently from another
thread.

Compiler memory values describe the executable's buffers. Allocator snapshots
describe the process and can include inputs, caches, and unrelated live arrays.
Use the former for function-to-function comparisons and the latter to diagnose
device-level pressure.

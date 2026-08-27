# Extended performance results

Measured with JAX/JAXlib 0.11.0 in float64/complex128. CPU results use native
Windows; GPU results use WSL2 and an RTX 4080. Every timing synchronizes device
work. Ratios are reference time or memory divided by optimized time or memory,
so values above 1 are improvements.

## Direct channels

At dimension 32, direct local kernels avoid promoting each Kraus operator to
the full register:

| Channel | CPU warm | CPU compile | CPU temporary memory |
| --- | ---: | ---: | ---: |
| Amplitude damping | 17.61x | 1.04x | 30.1x |
| Amplitude gain | 7.84x | 0.95x | 30.1x |
| Thermal | 8.11x | 0.89x | 35.6x |
| Bosonic dephasing | 7.75x | 1.26x | 229.6x |
| Ancilla dephasing/reset | 22.82x | 2.45x | 20,643.8x |
| Z/X measurement | 4.17x / 6.85x | 1.39x / 2.07x | 393 KB to 0 |
| Reset/imperfect reset | 6.05x / 14.77x | 2.06x / 1.97x | 393/786 KB to 0 |
| Qubit thermal/dephasing | 21.61x / 7.04x | 1.46x / 1.74x | 786/393 KB to 8 B/0 |

On GPU, the oscillator kernels reduce temporary memory by 24.6x–124x and the
qubit kernels reduce 4.59–4.98 MB to zero or near-zero. Tiny-kernel warmed
times are dominated by WSL/display-GPU dispatch and range from modest wins to
regressions; the ancilla reset is 1.06x faster and compiles 1.54x faster.

Reproduce with `python benchmarks/channels.py --include-reference`.

Shifted bosonic channels now reduce Kraus branches sequentially on CPU and
with `vmap` on accelerators. At dimension 60 with four density matrices, the
RTX 4080 path was 1.11x faster warm for 9 branches (2.64x at the 10th
percentile) and 1.69x for 81 branches. Cold time improved 1.02x and 1.07x,
respectively, while HLO fell from 170 to 110 lines. Peak memory rose by 1.8%
for 9 branches and 14.4% for 81 branches; the largest numerical difference
was `3.6e-15`. The sequential path remains selected on CPU because
vectorization slowed its 9-branch warm time by 1.69x.

## Compiled loops

For 16 channel rounds, `simulate_repeated` reduces CPU compile time by
2.17x–2.96x and StableHLO size by 5.39x–7.73x. Warm performance is
backend- and channel-dependent: CPU ratios range from 0.72x to 2.14x, while
GPU ratios range from 0.17x to 3.68x. The loop is therefore primarily a
compile-size/scaling improvement; it is not forced on short circuits.

## Construction and local simulation

| Change | CPU cold | CPU warm | GPU cold | GPU warm | Main memory result |
| --- | ---: | ---: | ---: | ---: | --- |
| Conditional displacement | 1.14x | 1.69x | 1.81x | 1.99x | CPU temp 1.34x lower |
| Echoed conditional displacement | 1.01x | 2.71x | 1.23x | 1.97x | CPU temp 1.34x lower |
| Thermal Kraus construction | 1.25x | 11.00x | 0.96x | 0.96x | temp 3.87x CPU / 2.58x GPU lower |
| Local Schrödinger evolution | 1.19x | 3.84x | auto-selects promoted | 1.02x | CPU temp 4.82x lower |
| Local Lindblad evolution | 1.16x | 2.17x | 1.11x | 1.90x | temp 1.14x CPU / 1.03x GPU lower |

Lazy amplitude-damping gate construction is 1.41x faster on CPU and 2.89x on
GPU when the fallback Kraus map is not requested.

Reproduce with `construction.py` and `hamiltonian.py --include-reference`.

## Distributions, devices, and sweeps

| Change | CPU cold | CPU warm | GPU cold | GPU warm | Other |
| --- | ---: | ---: | ---: | ---: | --- |
| Chunked density-matrix Q function | 4.89x | 1.43x | 6.74x | 1.29x | peak memory 12.67x CPU / 6.06x GPU lower |
| Hoisted Transmon Hamiltonian ops | 1.28x | 0.99x | 2.77x | 1.36x | compile 1.32x CPU / 3.47x GPU faster |
| Hoisted KNO Hamiltonian ops | 1.31x | 1.71x | 2.63x | 0.99x | HLO lines 1.98x lower |
| Vectorized Transmon wavefunctions | 41.93x | 2.30x | 1,192.89x | 1.38x | HLO lines 110x lower |
| `vmap` parameter sweep | 2.92x | 2.55x | 1.21x | 40.97x | HLO lines 10.51x lower |

The vectorized GPU wavefunction path trades compiler memory for its very large
compile reduction: temporary memory rises from 0.025 MB to 4.01 MB. The GPU
sweep also raises peak memory by about 9%.

Reproduce with `scan.py` and `devices.py`.

## Cat-sBs end to end

The external 480-round, dimension-32 model prebuilds dense Kraus tensors and
applies them through custom `einsum` kernels inside its own `lax.scan`. An
adapter now preserves JAXQuantum channel objects and applies direct kernels to
the storage density matrix and the non-trailing ancilla axes. Custom ECD noise
retains its original Kraus contraction.

| Backend | Variant | First cold | Warm median | Warm ratio |
| --- | --- | ---: | ---: | ---: |
| CPU | Current raw Kraus | 6.176 s | 1.521 s | 1.00x |
| CPU | Cached raw Kraus | 6.017 s | 1.447 s | 1.05x |
| CPU | Direct ancilla | 5.420 s | 1.101 s | 1.38x |
| CPU | Direct ancilla + storage | 4.840 s | 0.857 s | **1.77x** |
| RTX 4080 | Current raw Kraus | 8.242 s | 0.604 s | 1.00x |
| RTX 4080 | Cached raw Kraus | 8.220 s | 0.541 s | 1.12x |
| RTX 4080 | Direct ancilla | 7.809 s | 0.505 s | **1.20x** |
| RTX 4080 | Direct ancilla + storage | 7.538 s | 0.554 s | 1.09x |

The all-direct CPU kernel is 1.86x faster, with 19% less compiler temporary
memory and 23% smaller arguments. On the GPU, direct ancilla channels are
1.25x faster at the kernel level; direct storage channels enlarge HLO and lose
that gain, so the ancilla-only variant is preferred. Maximum trace differences
from the raw Kraus reference are `1.11e-13` (CPU) and `3.63e-14` (GPU);
lifetime relative differences remain below `2e-12`.

The external-model benchmark and raw outputs are archived with the cat project
in `bosonic-sims`.

## Shared cat/GKP sBs simulation

The shared functional kernel represents noisy conditional displacements
blockwise, applies local channels directly, and uses `lax.scan` over rounds.
Cat and GKP protocols differ only in pulse geometry, rotations, timing, and
device parameters.

For the current GKP device at dimension 60, the shared model agrees with the
jump-operator-corrected colleague model to 0.0278% in fitted lifetime
(2245.95 us versus 2245.33 us); the full contrast differs by `3.43e-6` in
relative L2 norm. Under the legacy parameters it gives 70.276 us versus
69.926 us (0.501%), while the ideal traces agree to `3.74e-9` relative L2.
Compared with the colleague implementation, the shared model is 1.64x faster
including build and cold execution and 28.24x faster warm for the current
case; the legacy ratios are 1.29x and 11.45x.

Replacing dense noisy-CD tensors with the blockwise form at dimension 60 cuts
GPU argument memory 3.85x, compiler peak memory 2.59x, and temporary memory
1.24x while improving warm execution 1.23x. Using the exact zero-temperature
oscillator path improves the same dense reference by 4.40x warm, 1.42x cold,
4.14x in arguments, and 2.69x in peak memory with no measurable lifetime
change.

| GKP profile | Windows CPU | RTX 4080 | Ratio |
| --- | ---: | ---: | ---: |
| Cold kernel | 1.584 s | 2.757 s | CPU 1.74x faster |
| Warm kernel | 0.473 s | 0.0954 s | GPU 4.96x faster |
| Compiler temporary | 10.68 MiB | 6.63 MiB | GPU 1.61x lower |
| Compiler peak | 7.76 MiB | 15.22 MiB | CPU 1.96x lower |

CPU and GPU lifetimes agree to below `1e-12` relative. The current all-on GKP
lifetime is 2.255 ms at one microstep and 2.263 ms at two; one microstep is
0.405% below a four-microstep reference. Its all-on-context error ranking is
storage Tphi (171.41/s), CD-qubit T1 (169.19/s), storage T1 (139.76/s), qubit
Tphi (9.38/s), idle qubit T1 (1.85/s), then reset (1.24/s).

For nominal cat parameters at `Delta=0.60`, `ratio=3.125`, the fitted bit
lifetime rises from 0.668 ms at nbar 1 to 45.112 ms at nbar 4; the ranking is
stable: storage Tphi, storage T1, qubit Tphi, CD-qubit T1, idle qubit T1,
reset. At nbar 4, one microstep differs from four by 0.0097% and dimension 32
differs from dimension 72 by 0.0021%.

The exact measured-device configuration supplied for the independent
`cat_sbs.py` model gives 33.108 ms at `Delta=0.60`, `ratio=3.125`, dimension
52, and 480 alternating rounds; the shared result differs by 0.00097%. At
`ratio=2`, its local sweep instead peaks at
41.622 ms near `Delta=0.75`; the shared result is 41.613 ms (0.023% lower).
Thus the remembered approximately 30 ms optimum requires a different model
revision or an additional error channel not present in that configuration.

Reproduce the kernel profile with `sbs_device.py`. The experiment-specific
error-budget, comparison, and control-sweep runners are archived with their cat
and GKP projects in `bosonic-sims`.

## Final core review

The PR-polish pass retained small, backend-neutral changes with exact numerical
agreement and no measured memory regression:

| Change | CPU cold | CPU warm/eager | RTX 4080 cold | RTX 4080 warm/eager | Compiler result |
| --- | ---: | ---: | ---: | ---: | --- |
| Metadata-only `reshape_qdims` | 3.46x | 16.05x eager | 3.70x | 159x eager | StableHLO 6.38x smaller |
| Direct scalar-identity arithmetic | 1.23x | 1.80x eager | 1.27x | 1.87x eager | StableHLO 2.09x–2.64x smaller |
| Dense operator norm | 1.35x | 3.08x | neutral | neutral | CPU StableHLO 1.83x smaller |
| Single-branch Kraus map | 1.16x | 1.15x | 1.22x | 1.01x | lowering 1.73x–1.77x faster |
| Shared local-Kraus layout | 1.12x | 0.99x | 1.15x | 0.99x | compile 1.13x–1.18x faster |

`reshape_qdims` now changes only static quantum metadata, while
`reshape_bdims` reuses the existing backend implementation after reshaping.
Scalar operator arithmetic no longer recursively constructs and tidies a
Qarray, and is now JIT- and gradient-safe. SparseDIA identities remain sparse;
BCOO scalar batches use sparse-native identities and broadcasting. Warm JIT
execution and compiler memory were unchanged for these Qarray operations.

Dense operator norms now compute singular values directly on CPU and retain
the Gram/eigenvalue route on accelerators, where small-matrix SVD was much
slower. At batch 4 and dimension 64, CPU compiler temporary memory rose from
512 KiB to 770 KiB, while reported peak memory stayed unchanged. The relative
numerical difference was about `4e-16`.

For a dimension-52 sBs half-round with 12 reset branches, replacing the
prebuilt dense reset Kraus stack with its exact blockwise factors improved CPU
cold/warm time by 1.08x/1.89x and GPU cold/warm time by 1.89x/1.22x. Argument
memory fell 2.97x; compiler peak memory fell 2.70x on CPU and 1.34x on GPU.
GPU compile time improved 1.66x and StableHLO lines fell 1.76x. The maximum
density-matrix difference was `2.08e-17`.

The review also fixed sparse initial-state simulation, batched thermal-qubit
Kraus construction, batched device eigenvector slicing, SparseDIA batch
reshaping, and operator quantum-dimension reshaping. These are compatibility
fixes rather than benchmark claims.

## Diffrax solver interface

The native Diffrax interface adds solver, controller, `SaveAt`, adjoint,
event, progress-meter, and continuation-state controls without changing the
lowered computation. CPU and GPU compiler temporary/peak memory and StableHLO
line counts were identical before and after the interface update.

| Workload | CPU cold (old -> new) | CPU warm (old -> new) | RTX 4080 cold (old -> new) | RTX 4080 warm (old -> new) |
| --- | ---: | ---: | ---: | ---: |
| `sesolve` | 0.387 -> 0.373 s | 43.7 -> 45.0 us | 0.455 -> 0.444 s | 1.476 -> 1.542 ms |
| `mesolve` | 0.447 -> 0.454 s | 110.0 -> 107.6 us | 0.549 -> 0.533 s | 2.254 -> 2.041 ms |
| Circuit solve | 0.417 -> 0.409 s | 58.0 -> 60.2 us | 0.525 -> 0.488 s | 2.822 -> 2.716 ms |

These small mixed changes are benchmark noise rather than an execution-path
regression: output norms were identical, and numerical equivalence tests cover
the legacy and native interfaces. Default calls retain the existing progress
bar, `max_steps=100_000`, save-at-input-times behavior, and Qarray return type.
Legacy string/bool options remain available with `FutureWarning` guidance.

## Rejected experiments

- A composite `ptrace` lowered HLO from 96 to 55 lines, but cold time was
  unchanged, warm time improved only 1.04x, and temporary memory rose from
  zero to 333 KB. It was removed.
- Chunking shifted-channel branches improved large CPU thermal maps but
  regressed cold time and small-map runtime, so CPU retains the simpler
  sequential reduction.
- Direct storage channels improve the cat-sBs CPU path but slow its GPU kernel;
  the benchmark keeps storage and ancilla selection explicit.
- Moving the first noisy round inside `simulate_repeated` reduced StableHLO by
  about 1.8x and GPU cold time by 1.42x, but a first-round ket-to-density-matrix
  conversion changes the loop carry pytree. The general API keeps the
  behavior-preserving structure.
- Chunking generic high-branch Kraus maps improved a 64-branch GPU kernel by
  2.73x warm, but raised temporary/peak memory by 23%/18% and gave negligible
  gains at 16–32 branches. The sequential default remains simpler and leaner.

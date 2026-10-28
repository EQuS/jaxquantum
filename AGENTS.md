# Guidance for agents working on jaxquantum

## Ask before installing

If someone provides only a repository link, read this file at the requested
branch or commit. Ask only the questions they have not answered:

1. Do you want to use jaxquantum, or develop it and prepare pull requests?
2. Which OS will run Python (Windows, Linux, macOS, or WSL), and should JAX
   use a CPU, NVIDIA GPU, or TPU? How many devices are available?
3. Do you have an environment to reuse, or should I create one?

For NVIDIA GPUs on Windows, use WSL2 and the [Windows GPU guide](https://jaxquantum.org/documentation/getting_started/gpu_windows.html).
Verify that JAX sees the requested hardware, and use the same Python environment
for commands and notebook kernels.

### Use jaxquantum

Create or activate an isolated Python 3.11+ environment. Choose the command
that matches the hardware:

```bash
# CPU
python -m pip install "jaxquantum @ git+https://github.com/EQuS/jaxquantum.git"
# NVIDIA GPU with CUDA 13
python -m pip install "jaxquantum[gpu] @ git+https://github.com/EQuS/jaxquantum.git"
# TPU
python -m pip install "jaxquantum[tpu] @ git+https://github.com/EQuS/jaxquantum.git"
```

Append `@REF` to the Git URL for a specific branch or commit. Do not install
development extras for someone who only wants to use the package. Check
`jax.devices()`, import `jaxquantum`, and run a small operation. See the
[installation guide](https://jaxquantum.org/documentation/getting_started/installation.html)
for troubleshooting.

### Develop jaxquantum

Clone the requested branch or commit and install it editable with
`python -m pip install -e ".[dev,docs,tests]"`. Add `gpu` or `tpu` to the
extras when needed. Read the relevant API and tutorials before changing public
behavior, then run the PR checks below.

## Learn the API before changing it

- Read the relevant [API reference](https://jaxquantum.org/reference/jaxquantum/index.html),
  [tutorial notebooks](https://jaxquantum.org/documentation/tutorials/index.html),
  and nearby Markdown documentation. Inspect the matching source and tests;
  the published site may lag the branch.
- Keep examples runnable and short. Document user-facing behavior in the docs
  and contributor workflow here. Update docs and tests when changing an API.
- Preserve default dense behavior and avoid silently materializing structured
  sparse operators as dense arrays.

## Before a pull request

1. Install the development, docs, and test extras in the target environment.
2. Run `bash test.sh` from the repository root. Set `PYTHON=/path/to/python`
   if needed. It checks Ruff without changing files, runs ordinary pytest,
   isolated virtual-CPU sharding tests, documentation notebooks, and a docs
   build. Use `--no-docs` only for a quick code check.
3. On a machine with two physical GPUs, run
   `pytest -q test/manual_multi_gpu/test_two_gpu_sharding.py` explicitly.
   These tests are excluded from the default suite.
4. Review `git diff --check` and the diff against the PR base. Avoid broad
   formatting changes. Investigate skipped tests before reporting a pass.

For performance changes, use the benchmarking utilities, separate first-call
compilation from warm execution, synchronize timings with
`block_until_ready()`, and compare the same problem sizes and hardware.

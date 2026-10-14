# Guidance for agents working on jaxquantum

## Installation: ask first, then choose a track

If someone provides only a GitHub repository link, read this file at the same
branch or commit before choosing commands. Ask these questions in one short
message, omitting anything the user has already answered:

1. Do you want to **use** jaxquantum in your own code, or also **develop** it and
   prepare pull requests?
2. What OS will run Python (Windows, Linux, macOS, or WSL), and what hardware
   should JAX use (CPU, NVIDIA GPU, or TPU)? For a GPU, how many devices?
3. Do you want the optional cuQuantum backend? If yes, do you have access to
   Shantanu's private `cuquantum_python_jax_cu13-0.0.7.tar.gz` PID, and where
   is it stored?
4. Do you have an environment to reuse, or should I create an isolated one?

Use the answers to select one track below. For NVIDIA GPUs on Windows, use
WSL2 and the [Windows GPU guide](https://jaxquantum.org/documentation/getting_started/gpu_windows.html);
native Windows JAX is CPU-only. Verify the chosen GPU is visible inside WSL.
Use the selected environment's Python for both commands and notebook kernels.

### Track A: use jaxquantum

Create or activate an isolated Python 3.11+ environment. Install from the
requested GitHub branch or commit without cloning. Choose exactly one command:

```bash
# CPU
python -m pip install "jaxquantum @ git+https://github.com/EQuS/jaxquantum.git"
# NVIDIA GPU with CUDA 13
python -m pip install "jaxquantum[gpu] @ git+https://github.com/EQuS/jaxquantum.git"
# TPU
python -m pip install "jaxquantum[tpu] @ git+https://github.com/EQuS/jaxquantum.git"
```

For a specific branch or commit, append `@REF` to the Git URL (for example,
`...jaxquantum.git@feature/cqt-jax`). Do not install development, test, or docs
extras for a user who only wants to run the package. After installation, check
`jax.devices()`, import `jaxquantum`, and run a small dense operation on the
requested hardware. See the [installation guide](https://jaxquantum.org/documentation/getting_started/installation.html)
for OS-specific details and troubleshooting.

### Track B: develop jaxquantum

Clone the requested branch or commit, create or activate an isolated environment,
and install editable source with `python -m pip install -e ".[dev,docs,tests]"`.
Add `gpu` or `tpu` to the extras only when the selected hardware needs it.
Follow the PR checks below. Read the relevant API and tutorials before changing
public behavior.

### Optional cuQuantum setup (either track)

cuQuantum is not needed for dense or sparse simulation. It currently requires
the private PID above plus a compatible public cuQuantum package; there is no
public `jaxquantum[cuquantum]` extra yet. Only set it up when the user asks for
it and supplies access to the PID. Follow the
[cuQuantum installation notes](https://jaxquantum.org/documentation/tutorials/_cuquantum_installation.html).
Keep the PID and extracted files outside Git (the ignored `.local/` directory
is suitable); never commit or publish them. Verify ordinary jaxquantum usage
without cuQuantum as well.

## Learn the API before changing it

- Read the relevant [API reference](https://jaxquantum.org/reference/jaxquantum/index.html),
  [tutorial notebooks](https://jaxquantum.org/documentation/tutorials/index.html),
  and nearby Markdown documentation before changing public behavior. For a
  branch checkout, inspect its matching docs, source, and tests as well; the
  published site can lag the branch.
- Keep examples runnable and short. Explain user-facing behavior in the docs;
  reserve implementation and contributor workflow rules for this file. Update
  both docs and tests when changing an API or backend behavior.
- Preserve the default dense path and backend-agnostic behavior. Keep optional
  backend imports behind their selection boundary, and test the package with
  cuQuantum unavailable. Do not introduce silent dense materialization of
  multi-mode structured operators.

## Before a pull request (developer track)

1. Install the development, docs, and test extras in the target environment.
2. Run `bash test.sh` from the repository root, setting `PYTHON=/path/to/python`
   if needed. It performs a non-mutating Ruff correctness check, ordinary
   pytest, isolated eight-device CPU sharding tests, documentation notebooks,
   and an MkDocs build. Use `--no-docs` only for a quick code check.
3. If a CUDA GPU and the private PID are available, also run
   `bash test.sh --cuquantum` to include cuQuantum tests and its tutorial.
   Physical multi-GPU tests live in `test/manual_multi_gpu/` and require an
   explicitly allocated multi-GPU machine; they are not part of the default run.
4. Review `git diff --check` and the diff against the PR base. Avoid broad
   formatting changes and do not run Ruff with `--fix` or `format` on unrelated
   files. Investigate skipped tests rather than treating a partial run as a pass.

For performance changes, use the existing benchmarking utilities, distinguish
first-call compilation from warm execution, synchronize JAX timings with
`block_until_ready()`, and compare the same physics across backends and problem
sizes. State hardware and memory limits when reporting multi-GPU results.

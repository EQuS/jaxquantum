# Guidance for agents working on jaxquantum

## Start with the user's environment

- For installation work, establish the OS, Python environment, hardware (CPU,
  NVIDIA GPU, or TPU), and GPU count when they are not already known. Ask whether
  the user wants the optional cuQuantum backend. Do not make a GPU installation
  the default for CPU users.
- Use an isolated environment and verify `import jaxquantum` and `jax.devices()`
  in the same interpreter that will run tests or notebooks. For development,
  install `python -m pip install -e ".[dev,docs,tests]"`; add `gpu` or `tpu` only
  when appropriate. See [installation](docs/documentation/getting_started/installation.md).
- For NVIDIA GPUs on Windows, use WSL2 and follow the
  [Windows GPU guide](docs/documentation/getting_started/gpu_windows.md).
  Native Windows JAX installations are CPU-only. Select the matching WSL Python
  environment as the notebook kernel.
- cuQuantum is optional and currently requires access to Shantanu's private
  `cuquantum_python_jax_cu13-0.0.7.tar.gz` PID, alongside the compatible public
  cuQuantum package. Ask for the private archive's location before attempting
  that setup. Follow the [cuQuantum installation notes](docs/documentation/tutorials/_cuquantum_installation.md);
  keep the PID and extracted files outside Git (the ignored `.local/` directory
  is suitable). Never commit or publish them. Standard dense and sparse paths
  must work without cuQuantum installed.

## Learn the API before changing it

- Read the relevant [API reference](docs/reference/jaxquantum/index.md),
  [tutorial notebooks](docs/documentation/tutorials/index.md), and nearby
  Markdown documentation before changing public behavior. Check the source and
  tests as well; documentation can lag implementation.
- Keep examples runnable and short. Explain user-facing behavior in the docs;
  reserve implementation and contributor workflow rules for this file. Update
  both docs and tests when changing an API or backend behavior.
- Preserve the default dense path and backend-agnostic behavior. Keep optional
  backend imports behind their selection boundary, and test the package with
  cuQuantum unavailable. Do not introduce silent dense materialization of
  multi-mode structured operators.

## Before a pull request

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

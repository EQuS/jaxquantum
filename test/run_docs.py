"""Execute documentation notebooks without modifying source files."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    """Execute notebooks in a temporary directory with their required devices."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuquantum",
        action="store_true",
        help="also execute the optional CUDA/cuQuantum tutorial",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as output_dir:
        for notebook in sorted((root / "docs").rglob("*.ipynb")):
            is_cuquantum = notebook.name == "cuquantum.ipynb"
            if is_cuquantum and not args.cuquantum:
                print(f"Skipping optional {notebook.relative_to(root)}", flush=True)
                continue

            # The sharding tutorial creates virtual CPU devices; the cuQuantum
            # tutorial needs a CUDA device. Keep each kernel in its own process.
            env = os.environ.copy()
            env["JAX_PLATFORMS"] = "cuda,cpu" if is_cuquantum else "cpu"
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            print(f"Executing {notebook.relative_to(root)}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--execute",
                    "--to=notebook",
                    f"--output-dir={output_dir}",
                    "--ExecutePreprocessor.timeout=600",
                    str(notebook),
                ],
                check=True,
                cwd=root,
                env=env,
            )


if __name__ == "__main__":
    main()

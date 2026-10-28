"""Execute every documentation notebook without modifying source files."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    """Execute notebooks in separate CPU processes in a temporary directory."""
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as output_dir:
        for notebook in sorted((root / "docs").rglob("*.ipynb")):
            env = os.environ.copy()
            env["JAX_PLATFORMS"] = "cpu"
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

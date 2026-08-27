"""Execute every documentation notebook without modifying source files."""

import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]

with tempfile.TemporaryDirectory() as output_dir:
    for notebook in sorted((root / "docs").rglob("*.ipynb")):
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
        )

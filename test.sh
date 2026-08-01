#!/usr/bin/env bash
set -euo pipefail

python_cmd="${PYTHON:-python}"
"$python_cmd" -m pytest test --ignore=test/test_sharding.py "$@"
JAX_PLATFORMS=cpu "$python_cmd" -m pytest test/test_sharding.py "$@"

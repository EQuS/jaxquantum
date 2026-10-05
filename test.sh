#!/usr/bin/env bash
set -euo pipefail

python_cmd="${PYTHON:-python}"
run_docs=false
pytest_args=()
for argument in "$@"; do
    if [[ "$argument" == "--docs" ]]; then
        run_docs=true
    else
        pytest_args+=("$argument")
    fi
done

"$python_cmd" -m pytest test --ignore=test/test_sharding.py "${pytest_args[@]}"
JAX_PLATFORMS=cpu "$python_cmd" -m pytest test/test_sharding.py "${pytest_args[@]}"

if $run_docs; then
    "$python_cmd" test/run_docs.py
fi

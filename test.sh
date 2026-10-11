#!/usr/bin/env bash
set -euo pipefail

python_cmd="${PYTHON:-python}"
run_docs=true
run_cuquantum=false
pytest_args=()
for argument in "$@"; do
    case "$argument" in
        --docs) run_docs=true ;;
        --no-docs) run_docs=false ;;
        --cuquantum) run_cuquantum=true ;;
        *) pytest_args+=("$argument") ;;
    esac
done

# Check correctness without rewriting existing code or formatting.
"$python_cmd" -m ruff check --select E4,E7,E9,F \
    --ignore E402,E731,E741,F401,F403 jaxquantum test/run_docs.py \
    test/test_optional_cuquantum.py docs/scripts/publish_agents.py

"$python_cmd" -m pytest test --ignore=test/test_sharding.py "${pytest_args[@]}"
JAX_PLATFORMS=cpu "$python_cmd" -m pytest test/test_sharding.py "${pytest_args[@]}"

if $run_cuquantum; then
    "$python_cmd" -c 'import jax; jax.config.update("jax_enable_x64", True); import cuquantum.densitymat.jax; assert any(d.platform == "gpu" for d in jax.devices()), "cuQuantum checks require a GPU"'
    "$python_cmd" -m pytest -o addopts= test/test_cuquantum_backend.py "${pytest_args[@]}"
fi

if $run_docs; then
    if $run_cuquantum; then
        "$python_cmd" test/run_docs.py --cuquantum
    else
        "$python_cmd" test/run_docs.py
    fi
    "$python_cmd" -m mkdocs build --clean
    cmp AGENTS.md site/AGENTS.md
    cmp AGENTS.md site/agents.md
fi

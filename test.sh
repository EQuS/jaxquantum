#!/usr/bin/env bash
set -euo pipefail

python_cmd="${PYTHON:-python}"
run_docs=true
pytest_args=()
for argument in "$@"; do
    case "$argument" in
        --docs) run_docs=true ;;
        --no-docs) run_docs=false ;;
        *) pytest_args+=("$argument") ;;
    esac
done

# Check correctness without rewriting code or formatting.
"$python_cmd" -m ruff check --select E4,E7,E9,F \
    --ignore E402,E731,E741,F401,F403 jaxquantum \
    test/run_docs.py test/manual_multi_gpu docs/scripts/publish_agents.py

"$python_cmd" -m pytest test --ignore=test/test_sharding.py "${pytest_args[@]}"
JAX_PLATFORMS=cpu "$python_cmd" -m pytest test/test_sharding.py "${pytest_args[@]}"

if $run_docs; then
    "$python_cmd" test/run_docs.py
    "$python_cmd" -m mkdocs build --clean
    cmp AGENTS.md site/AGENTS.md
    cmp AGENTS.md site/agents.md
fi

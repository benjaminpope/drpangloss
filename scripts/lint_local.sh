#!/usr/bin/env bash

set -eo pipefail

show_help() {
  cat <<'EOF'
Low-noise local lint runner for drpangloss.

Usage:
  scripts/lint_local.sh [--fix] [--no-notebooks] [--changed]

Options:
  --fix           Apply Ruff fixes/formatting changes.
  --no-notebooks  Skip notebook lint pass.
  --changed       Lint only changed files vs merge-base.
  -h, --help      Show this help message.

Defaults:
  - Checks src/ and tests/
  - Includes notebooks/*.ipynb
  - Writes full logs to .lint-logs/
EOF
}

fix_mode=0
include_notebooks=1
changed_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fix)
      fix_mode=1
      ;;
    --no-notebooks)
      include_notebooks=0
      ;;
    --changed)
      changed_only=1
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help
      exit 2
      ;;
  esac
  shift
done

if command -v uv >/dev/null 2>&1 && [[ -x ".venv/bin/python" ]]; then
  ruff_cmd=(uv run --python .venv/bin/python ruff)
elif command -v uv >/dev/null 2>&1; then
  ruff_cmd=(uv run ruff)
elif command -v ruff >/dev/null 2>&1; then
  ruff_cmd=(ruff)
else
  echo "Could not find 'ruff'. Install with 'uv pip install --python .venv/bin/python ruff' or ensure 'ruff' is on PATH." >&2
  exit 127
fi

mkdir -p .lint-logs
timestamp="$(date +"%Y%m%d_%H%M%S")"
log_file=".lint-logs/lint_${timestamp}.log"

run_phase() {
  local label="$1"
  shift
  {
    echo
    echo "===== ${label} ====="
    echo "Command: $*"
    "$@"
  } >>"$log_file" 2>&1
}

count_python=0
count_notebooks=0
py_targets=()
ipynb_targets=()

if [[ "$changed_only" -eq 1 ]]; then
  if git rev-parse --verify origin/main >/dev/null 2>&1; then
    base_ref="origin/main"
  else
    base_ref="$(git rev-list --max-parents=0 HEAD | tail -n 1)"
  fi

  changed_py=()
  while IFS= read -r file; do
    [[ -n "$file" ]] && changed_py+=("$file")
  done < <(git diff --name-only --diff-filter=ACMRTUXB "${base_ref}"...HEAD -- "*.py" | sort)

  changed_ipynb=()
  while IFS= read -r file; do
    [[ -n "$file" ]] && changed_ipynb+=("$file")
  done < <(git diff --name-only --diff-filter=ACMRTUXB "${base_ref}"...HEAD -- "*.ipynb" | sort)

  for file in "${changed_py[@]}"; do
    if [[ "$file" == src/* || "$file" == tests/* ]]; then
      py_targets+=("$file")
    fi
  done

  for file in "${changed_ipynb[@]}"; do
    if [[ "$file" == notebooks/* ]]; then
      ipynb_targets+=("$file")
    fi
  done
else
  py_targets=(src tests)
  ipynb_targets=(notebooks/*.ipynb)
fi

if [[ "${#py_targets[@]}" -gt 0 ]]; then
  count_python="${#py_targets[@]}"
  if [[ "$fix_mode" -eq 1 ]]; then
    run_phase "ruff check (python, --fix)" "${ruff_cmd[@]}" check --fix "${py_targets[@]}"
    run_phase "ruff format (python)" "${ruff_cmd[@]}" format "${py_targets[@]}"
  else
    run_phase "ruff check (python)" "${ruff_cmd[@]}" check "${py_targets[@]}"
    run_phase "ruff format --check (python)" "${ruff_cmd[@]}" format --check "${py_targets[@]}"
  fi
fi

if [[ "$include_notebooks" -eq 1 && "${#ipynb_targets[@]}" -gt 0 ]]; then
  count_notebooks="${#ipynb_targets[@]}"
  if [[ "$fix_mode" -eq 1 ]]; then
    run_phase "ruff check (notebooks, --fix)" "${ruff_cmd[@]}" check --fix "${ipynb_targets[@]}"
    run_phase "ruff format (notebooks)" "${ruff_cmd[@]}" format "${ipynb_targets[@]}"
  else
    run_phase "ruff check (notebooks)" "${ruff_cmd[@]}" check "${ipynb_targets[@]}"
    run_phase "ruff format --check (notebooks)" "${ruff_cmd[@]}" format --check "${ipynb_targets[@]}"
  fi
fi

if [[ "$count_python" -eq 0 && ( "$include_notebooks" -eq 0 || "$count_notebooks" -eq 0 ) ]]; then
  echo "No lint targets found for requested scope."
  echo "Log: ${log_file}"
  exit 0
fi

echo "Lint completed successfully."
echo "Python targets: ${count_python}"
if [[ "$include_notebooks" -eq 1 ]]; then
  echo "Notebook targets: ${count_notebooks}"
else
  echo "Notebook targets: skipped"
fi
echo "Full log: ${log_file}"
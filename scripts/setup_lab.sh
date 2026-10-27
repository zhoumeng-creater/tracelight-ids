#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' 'Usage: bash scripts/setup_lab.sh --venv-dir PATH [--python python3.12] [--device cuda|cpu] [--skip-tests]'
}

PYTHON_BIN=python3.12
VENV_DIR=''
DEVICE=cuda
SKIP_TESTS=0
while (($#)); do
  case "$1" in
    --venv-dir) VENV_DIR="${2:?--venv-dir requires a path}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?--python requires an executable}"; shift 2 ;;
    --device) DEVICE="${2:?--device requires cuda or cpu}"; shift 2 ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$VENV_DIR" ]] || { usage >&2; exit 2; }
[[ "$DEVICE" == cuda || "$DEVICE" == cpu ]] || { printf '%s\n' '--device must be cuda or cpu' >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 for the laboratory recipe"'
VENV_DIR="$("$PYTHON_BIN" -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).expanduser().resolve())' "$VENV_DIR")"
if [[ -e "$VENV_DIR" ]]; then
  [[ -f "$VENV_DIR/pyvenv.cfg" && -x "$VENV_DIR/bin/python" ]] || { printf '%s\n' 'Venv directory exists but is not a POSIX virtual environment; choose a new directory.' >&2; exit 2; }
else
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing virtual environment is not Python 3.12"'
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r "$REPO_ROOT/requirements-lock.txt"
"$VENV_PYTHON" -m pip install -e "$REPO_ROOT" --no-deps
if [[ "$DEVICE" == cuda ]]; then
  TORCH_INDEX=https://download.pytorch.org/whl/cu126
  TORCH_VERSION='torch==2.6.0+cu126'
else
  TORCH_INDEX=https://download.pytorch.org/whl/cpu
  TORCH_VERSION='torch==2.6.0+cpu'
fi
"$VENV_PYTHON" -m pip install "$TORCH_VERSION" --index-url "$TORCH_INDEX"
"$VENV_PYTHON" -m pip check
export CUBLAS_WORKSPACE_CONFIG=:4096:8
DOCTOR_ARGUMENTS=(-m ids doctor)
if [[ "$DEVICE" == cuda ]]; then DOCTOR_ARGUMENTS+=(--require-cuda); fi
"$VENV_PYTHON" "${DOCTOR_ARGUMENTS[@]}"
if [[ "$SKIP_TESTS" -eq 0 ]]; then
  "$VENV_PYTHON" -m pytest "$REPO_ROOT/tests" -q
fi
printf 'Environment ready: %s\n' "$VENV_PYTHON"
printf '%s\n' 'No dataset was downloaded and no research experiment was started.'

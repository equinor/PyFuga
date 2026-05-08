#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
  echo "Usage: $0 <venv_dir> [packages...]"
  exit 1
fi

VENV_DIR="$1"
shift

uv venv "${VENV_DIR}"
# shellcheck disable=SC1090
. "${VENV_DIR}/bin/activate"

if [ "$#" -gt 0 ]; then
  uv pip install "$@"
fi

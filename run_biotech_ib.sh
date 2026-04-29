#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/Users/ultrarelativisticenergy/miniconda3/bin/python3"

if [[ ! -x "$PYTHON" ]]; then
  echo "Miniconda Python not found at: $PYTHON" >&2
  exit 1
fi

exec "$PYTHON" "$ROOT/main.py" "$@"

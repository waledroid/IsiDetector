#!/usr/bin/env bash
# Real compress.sh — invoked via the repo-root thin wrapper ./compress.sh.
# Lives under deploy/_impl/; the `compression/` package sits at the repo
# root. We cd to the repo root (not into compression/) so Python can see
# compression/ as a top-level package when invoked via `python -m`.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/isidet:${PYTHONPATH:-}"
exec python -m compression "$@"

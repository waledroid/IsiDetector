#!/usr/bin/env bash
# Thin wrapper: the real script lives at deploy/_impl/run_start.sh.
# Kept at repo root so muscle-memory `./run_start.sh` still works.
set -euo pipefail
exec "$(dirname "$0")/deploy/_impl/run_start.sh" "$@"

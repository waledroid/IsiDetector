#!/usr/bin/env bash
# Thin wrapper: the real script lives at deploy/_impl/up.sh.
# Kept at repo root so muscle-memory `./up.sh` still works.
set -euo pipefail
exec "$(dirname "$0")/deploy/_impl/up.sh" "$@"

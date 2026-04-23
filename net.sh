#!/usr/bin/env bash
# Thin wrapper: the real script lives at deploy/_impl/net.sh.
# Kept at repo root so muscle-memory `./net.sh` still works.
set -euo pipefail
exec "$(dirname "$0")/deploy/_impl/net.sh" "$@"

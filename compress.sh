#!/usr/bin/env bash
# Thin wrapper: the real script lives at deploy/_impl/compress.sh.
# Kept at repo root so muscle-memory `./compress.sh` still works.
set -euo pipefail
exec "$(dirname "$0")/deploy/_impl/compress.sh" "$@"

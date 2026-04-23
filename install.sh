#!/usr/bin/env bash
# Thin wrapper: the real script lives at deploy/_impl/install.sh.
# Kept at repo root so muscle-memory `./install.sh` still works.
set -euo pipefail
exec "$(dirname "$0")/deploy/_impl/install.sh" "$@"

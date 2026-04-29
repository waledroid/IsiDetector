#!/usr/bin/env bash
# Thin wrapper: the real script lives at deploy/_impl/autostart.sh.
# Kept at repo root so muscle-memory `./autostart.sh enable` works.
set -euo pipefail
exec "$(dirname "$0")/deploy/_impl/autostart.sh" "$@"

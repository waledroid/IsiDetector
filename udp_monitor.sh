#!/usr/bin/env bash
# Thin wrapper: the real script lives at deploy/_impl/udp_monitor.sh.
# Kept at repo root so muscle-memory `./udp_monitor.sh` still works.
set -euo pipefail
exec "$(dirname "$0")/deploy/_impl/udp_monitor.sh" "$@"

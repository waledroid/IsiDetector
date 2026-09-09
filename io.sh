#!/usr/bin/env bash
# Thin wrapper: the real script lives at deploy/_impl/io.sh.
set -euo pipefail
exec "$(dirname "$0")/deploy/_impl/io.sh" "$@"

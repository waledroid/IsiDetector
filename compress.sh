#!/usr/bin/env bash
# Thin wrapper around `python -m Compression` — symmetric with up.sh.
set -euo pipefail
cd "$(dirname "$0")"
exec python -m Compression "$@"

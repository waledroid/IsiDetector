#!/usr/bin/env bash
# Start the IsiDetector stack and open the UI once the web container is
# fully warm — waits for a readiness marker, then launches Chrome (or
# the default browser if Chrome isn't installed).
#
# Reads .deployment.env written by run_start.sh to pick the right
# compose profile. On a fresh machine run ./run_start.sh first; after
# that ./up.sh is the daily starter.
#
# Usage:
#   ./up.sh                 # auto (reads .deployment.env, else detects)
#   ./up.sh --force-cpu     # use CPU compose regardless of marker/GPU
#   ./up.sh --force-gpu     # use GPU compose even if nvidia-smi fails
#   ./up.sh -h | --help     # print this usage and exit
#
# Environment overrides:
#   URL=http://localhost:9501   destination to open
#   TIMEOUT_SEC=300             max seconds to wait for readiness
#   NO_BROWSER=1                start the stack but don't open a browser
#   FORCE_CPU=1                 same as --force-cpu (legacy)

set -euo pipefail
cd "$(dirname "$0")"

# ── Parse CLI flags ─────────────────────────────────────────────────────────
FORCE_MODE=""
for arg in "$@"; do
    case "$arg" in
        --force-cpu|--cpu)  FORCE_MODE="cpu" ;;
        --force-gpu|--gpu)  FORCE_MODE="gpu" ;;
        -h|--help)
            sed -n '2,21p' "$0" | sed 's/^# *//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Try: $0 --help" >&2
            exit 2
            ;;
    esac
done

URL="${URL:-http://localhost:9501}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"

# ── Detect compose profile ──────────────────────────────────────────────────
# Priority:
#   1. --force-cpu / --force-gpu CLI flag (or legacy FORCE_CPU=1 env var)
#   2. .deployment.env written by run_start.sh (COMPOSE_MODE=gpu|cpu)
#   3. autodetect via nvidia-smi
#
# Then a sanity check: if GPU mode is chosen but nvidia-smi can't reach
# the driver, auto-fall-back to CPU with a warning — avoids the cryptic
# 'could not select driver "nvidia"' daemon error when a machine was
# migrated from GPU→CPU but .deployment.env is stale.
COMPOSE_MODE="gpu"
SUDO_DOCKER=""

if [[ -f .deployment.env ]]; then
    # shellcheck disable=SC1091
    source .deployment.env
fi

# CLI flag beats env var beats file
if [[ -n "$FORCE_MODE" ]]; then
    COMPOSE_MODE="$FORCE_MODE"
elif [[ "${FORCE_CPU:-0}" == "1" ]]; then
    COMPOSE_MODE="cpu"
elif [[ ! -f .deployment.env ]]; then
    # No run_start.sh marker — autodetect
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        COMPOSE_MODE="gpu"
    else
        COMPOSE_MODE="cpu"
    fi
fi

# Sanity: if we think we're in GPU mode but the driver is unreachable,
# auto-fall-back. Skip this check if --force-gpu was explicitly passed.
if [[ "$COMPOSE_MODE" == "gpu" && "$FORCE_MODE" != "gpu" ]]; then
    if ! (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1); then
        echo "⚠ .deployment.env says GPU but nvidia-smi is unavailable on this host."
        echo "  Falling back to CPU compose. Re-run ./run_start.sh to refresh the marker,"
        echo "  or pass --force-gpu to override this safety check."
        COMPOSE_MODE="cpu"
    fi
fi

if [[ "$COMPOSE_MODE" == "cpu" ]]; then
    COMPOSE_CMD="$SUDO_DOCKER docker compose -f docker-compose.yml -f docker-compose.cpu.yml"
    READY_PATTERN="Running on http://"   # Flask startup banner — works on both CPU and GPU
    echo "▶ Using CPU compose profile (Dockerfile.cpu)"
else
    COMPOSE_CMD="$SUDO_DOCKER docker compose"
    READY_PATTERN="ONNX preload (CUDA kernels warm"
    echo "▶ Using GPU compose profile"
fi

# ── Start the stack ─────────────────────────────────────────────────────────
echo "▶ Starting IsiDetector stack (docker compose up -d --build)..."
$COMPOSE_CMD up -d --build

echo "▶ Waiting for web container to finish ONNX preload (timeout ${TIMEOUT_SEC}s)..."

# grep -m 1 exits after first match → SIGPIPE kills docker logs -f → its
# exit status becomes 141. With `set -o pipefail` that 141 would propagate
# and the `if` would misread grep's success as failure. Turn pipefail off
# just for this pipeline so the check reflects grep's real exit code.
set +o pipefail
if timeout "${TIMEOUT_SEC}s" $COMPOSE_CMD logs -f web 2>/dev/null \
        | grep --line-buffered -m 1 "$READY_PATTERN"; then
    echo "✓ Web container ready"
else
    echo "⚠ Readiness marker not seen within ${TIMEOUT_SEC}s — opening browser anyway"
fi
set -o pipefail

if [[ "${NO_BROWSER:-0}" == "1" ]]; then
    echo "ℹ NO_BROWSER=1 — skipping browser launch"
    exit 0
fi

# ── Open the UI ─────────────────────────────────────────────────────────────
open_url() {
    local url="$1"

    # 1. Chrome first (any packaging)
    for cmd in google-chrome google-chrome-stable chromium chromium-browser; do
        if command -v "$cmd" >/dev/null 2>&1; then
            "$cmd" "$url" >/dev/null 2>&1 &
            disown
            echo "✓ Opened $url in $cmd"
            return 0
        fi
    done

    # 2. macOS — try Chrome via `open -a`, else system default
    if [[ "$(uname)" == "Darwin" ]]; then
        if open -a "Google Chrome" "$url" >/dev/null 2>&1; then
            echo "✓ Opened $url in Google Chrome (macOS)"
        else
            open "$url"
            echo "✓ Opened $url via macOS default browser"
        fi
        return 0
    fi

    # 3. WSL — reach out to Windows-side Chrome or explorer
    if grep -qi microsoft /proc/version 2>/dev/null; then
        if command -v wslview >/dev/null 2>&1; then
            wslview "$url"
            echo "✓ Opened $url via wslview"
            return 0
        fi
        if command -v cmd.exe >/dev/null 2>&1; then
            if cmd.exe /c start chrome "$url" >/dev/null 2>&1; then
                echo "✓ Opened $url in Chrome (Windows via cmd.exe)"
            else
                cmd.exe /c start "" "$url" >/dev/null 2>&1
                echo "✓ Opened $url via Windows default browser"
            fi
            return 0
        fi
    fi

    # 4. Generic Linux — xdg-open hits the configured default
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url" >/dev/null 2>&1 &
        disown
        echo "✓ Opened $url via xdg-open"
        return 0
    fi

    echo "⚠ No browser launcher found. Open manually: $url"
    return 1
}

open_url "$URL"

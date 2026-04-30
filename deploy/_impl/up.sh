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
#   ./up.sh --no-build      # skip docker compose's --build flag (offline-safe)
#   ./up.sh --kiosk         # open browser fullscreen, no chrome UI
#   ./up.sh --open-only     # don't touch compose; wait for port + open browser
#                           # (use when systemd already brings the stack up)
#   ./up.sh -h | --help     # print this usage and exit
#
# Environment overrides:
#   URL=http://localhost:9501   destination to open
#   TIMEOUT_SEC=300             max seconds to wait for readiness
#   NO_BROWSER=1                start the stack but don't open a browser
#   FORCE_CPU=1                 same as --force-cpu (legacy)

set -euo pipefail
# Script lives at deploy/_impl/up.sh after the repo restructure. Remember
# the repo root so we can return to it at the end (hygiene: if anyone
# ever `source`s this script, their interactive shell stays at the repo
# root instead of leaking into deploy/).
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# cd to deploy/ so the compose files and .deployment.env are all
# reachable by their bare names below.
cd "$(dirname "$0")/.."

# ── Parse CLI flags ─────────────────────────────────────────────────────────
FORCE_MODE=""
for arg in "$@"; do
    case "$arg" in
        --force-cpu|--cpu)  FORCE_MODE="cpu" ;;
        --force-gpu|--gpu)  FORCE_MODE="gpu" ;;
        --no-build)         NO_BUILD=1 ;;     # skip docker compose's --build flag
        --kiosk)            KIOSK=1 ;;        # open browser fullscreen, no chrome
        --open-only)        OPEN_ONLY=1 ;;    # skip compose; just wait + open
        -h|--help)
            sed -n '2,25p' "$0" | sed 's/^# *//'
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
    # CPU: no ONNX preload is attempted (no CUDA to warm), so the Flask
    # startup banner is the only useful readiness signal.
    READY_PATTERN="Running on http://"
    echo "▶ Using CPU compose profile (Dockerfile.cpu, rfdetr sidecar skipped)"
else
    # docker-compose.gpu.yml adds the nvidia device reservation; --profile gpu
    # activates the rfdetr sidecar (gated in docker-compose.yml).
    COMPOSE_CMD="$SUDO_DOCKER docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile gpu"
    # GPU: match EITHER the ONNX preload marker (fires when settings.json
    # has a pre-configured .onnx weight → first inference is CUDA-warm)
    # OR the Flask startup banner (fires always → unblocks the browser
    # even when weight paths are empty and no preload was attempted).
    # Whichever hits first wins — grep -m 1 kills the pipe.
    READY_PATTERN="ONNX preload \(CUDA kernels warm|Running on http://"
    echo "▶ Using GPU compose profile (rfdetr sidecar enabled)"
fi

# ── --open-only short-circuit ──────────────────────────────────────────────
# When systemd brings the compose stack up at boot, the desktop autostart
# entry runs `up.sh --open-only` which skips the compose calls entirely
# and just waits for the web port to come alive, then opens the browser.
# Avoids racing with the systemd unit and double-launching the stack.
if [[ "${OPEN_ONLY:-0}" == "1" ]]; then
    PORT="${URL##*:}"
    PORT="${PORT%%/*}"
    : "${PORT:=9501}"
    echo "▶ --open-only: waiting for tcp/${PORT} (max ${TIMEOUT_SEC}s)..."
    deadline=$(( $(date +%s) + TIMEOUT_SEC ))
    while ! (echo > "/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; do
        if (( $(date +%s) >= deadline )); then
            echo "⚠ Port ${PORT} not reachable within ${TIMEOUT_SEC}s — opening browser anyway"
            break
        fi
        sleep 1
    done
    [[ "$(echo > /dev/tcp/127.0.0.1/${PORT} 2>/dev/null && echo ok)" == "ok" ]] && echo "✓ Port ${PORT} is up"

    if [[ "${NO_BROWSER:-0}" == "1" ]]; then
        echo "ℹ NO_BROWSER=1 — skipping browser launch"
        cd "$REPO_ROOT"
        exit 0
    fi
    # Fall through to the open_url block at the end of this script.
    SKIP_COMPOSE=1
fi

# ── Start the stack ─────────────────────────────────────────────────────────
# --no-build (or NO_BUILD=1) skips the rebuild step so an offline boot
# (e.g. autostart on a site PC with no internet) doesn't fail trying to
# pull base layers. The image must already exist locally — first install
# always runs run_start.sh which builds it.
if [[ "${SKIP_COMPOSE:-0}" != "1" ]]; then
if [[ "${NO_BUILD:-0}" == "1" ]]; then
    echo "▶ Starting IsiDetector stack (docker compose up -d, no build)..."
    $COMPOSE_CMD up -d
else
    echo "▶ Starting IsiDetector stack (docker compose up -d --build)..."
    $COMPOSE_CMD up -d --build
fi

echo "▶ Waiting for web container to finish ONNX preload (timeout ${TIMEOUT_SEC}s)..."

# grep -m 1 exits after first match → SIGPIPE kills docker logs -f → its
# exit status becomes 141. With `set -o pipefail` that 141 would propagate
# and the `if` would misread grep's success as failure. Turn pipefail off
# just for this pipeline so the check reflects grep's real exit code.
set +o pipefail
if timeout "${TIMEOUT_SEC}s" $COMPOSE_CMD logs -f web 2>/dev/null \
        | grep --line-buffered -E -m 1 "$READY_PATTERN"; then
    echo "✓ Web container ready"
else
    echo "⚠ Readiness marker not seen within ${TIMEOUT_SEC}s — opening browser anyway"
fi
set -o pipefail

if [[ "${NO_BROWSER:-0}" == "1" ]]; then
    echo "ℹ NO_BROWSER=1 — skipping browser launch"
    exit 0
fi
fi  # /SKIP_COMPOSE guard

# ── Open the UI ─────────────────────────────────────────────────────────────
open_url() {
    local url="$1"

    # Kiosk-mode flags — applied to Chrome/Chromium when KIOSK=1 is set
    # (or --kiosk was passed). Used by the autostart desktop entry so a
    # site-PC operator can't accidentally close the tab or navigate away.
    local kiosk_args=()
    if [[ "${KIOSK:-0}" == "1" ]]; then
        kiosk_args=(
            --kiosk
            --no-first-run
            --noerrdialogs
            --disable-translate
            --disable-features=TranslateUI
            --disable-pinch
            --overscroll-history-navigation=0
            --autoplay-policy=no-user-gesture-required
        )
    fi

    # 1. Chrome first (any packaging)
    for cmd in google-chrome google-chrome-stable chromium chromium-browser; do
        if command -v "$cmd" >/dev/null 2>&1; then
            "$cmd" "${kiosk_args[@]}" "$url" >/dev/null 2>&1 &
            disown
            local mode="${KIOSK:+kiosk }"
            echo "✓ Opened $url in $cmd (${mode:-windowed})"
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

# Return to the repo root so any code appended to this script later runs
# from a predictable CWD, and any `source ./up.sh` call leaves the
# interactive shell exactly where it started.
cd "$REPO_ROOT"

#!/usr/bin/env bash
# Start the IsiDetector stack and open the UI once the web container is
# fully warm — waits for the ONNX preload log line, then launches Chrome
# (or the default browser if Chrome isn't installed).
#
# Usage:
#   ./scripts/up.sh
#
# Environment overrides:
#   URL=http://localhost:9501   destination to open
#   TIMEOUT_SEC=300             max seconds to wait for readiness
#   NO_BROWSER=1                start the stack but don't open a browser

set -euo pipefail
cd "$(dirname "$0")/.."

URL="${URL:-http://localhost:9501}"
READY_PATTERN="ONNX preload (CUDA kernels warm"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"

echo "▶ Starting IsiDetector stack (docker compose up -d --build)..."
docker compose up -d --build

echo "▶ Waiting for web container to finish ONNX preload (timeout ${TIMEOUT_SEC}s)..."

# grep -m 1 exits after first match → SIGPIPE propagates up → docker logs -f
# dies cleanly. `timeout` bounds the wait so a broken preload path can't
# hang the script forever.
if timeout "${TIMEOUT_SEC}s" docker compose logs -f web 2>/dev/null \
        | grep --line-buffered -m 1 "$READY_PATTERN"; then
    echo "✓ Web container ready"
else
    echo "⚠ Readiness marker not seen within ${TIMEOUT_SEC}s — opening browser anyway"
fi

if [[ "${NO_BROWSER:-0}" == "1" ]]; then
    echo "ℹ NO_BROWSER=1 — skipping browser launch"
    exit 0
fi

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

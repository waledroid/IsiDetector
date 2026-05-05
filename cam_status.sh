#!/usr/bin/env bash
# ============================================================================
# cam_status.sh — probe the RTSP camera for live specs and supported streams
#
# Reads the saved camera URL from webapp/isitec_app/settings.json, then:
#   1. Checks network reachability (ping + TCP/554)
#   2. Reads the most recent "📹 Stream:" log line from the running container
#   3. Probes the saved URL via cv2 (inside the docker container) for
#      resolution, native fps, and codec
#   4. Tries common stream/sub-stream URL variants so you can see which
#      paths the camera actually answers on
#
# Usage:
#   ./cam_status.sh                      # use saved URL from settings.json
#   ./cam_status.sh -u rtsp://user:pass@ip:554/path
#                                        # explicit URL override
#   ./cam_status.sh --no-variants        # skip the alternative-path probing
#   ./cam_status.sh -h | --help
#
# Requires: bash, python3, the docker stack running (uses `docker compose
# exec web` to run cv2 probes — opencv is in the container, not on the host).
# ============================================================================

set -u

# ── Colour helpers (mirrors net.sh / autostart.sh) ──────────────────────────
if [ -t 1 ]; then
    GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
    RED='\033[0;31m';   BOLD='\033[1m';     NC='\033[0m'
else
    GREEN=''; YELLOW=''; CYAN=''; RED=''; BOLD=''; NC=''
fi

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[  OK]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $*" >&2; }
header()  {
    echo ""
    echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  $*${NC}"
    echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
    echo ""
}
section() {
    echo ""
    echo -e "${BOLD}─── $* ─${NC}"
}

# ── Locate repo + settings ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SETTINGS_PATH="webapp/isitec_app/settings.json"
[ -f "$SETTINGS_PATH" ] || SETTINGS_PATH="webapp/isitec_api/settings.json"

# ── CLI ─────────────────────────────────────────────────────────────────────
URL=""
DO_VARIANTS=1
PROBE_TIMEOUT=8

print_help() { sed -n '2,22p' "$0" | sed 's/^# *//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        -u|--url)        URL="${2:-}"; shift ;;
        --url=*)         URL="${1#*=}" ;;
        --no-variants)   DO_VARIANTS=0 ;;
        --timeout)       PROBE_TIMEOUT="${2:-8}"; shift ;;
        -h|--help)       print_help; exit 0 ;;
        *)               fail "Unknown argument: $1"; print_help; exit 2 ;;
    esac
    shift
done

# ── Resolve URL (saved → CLI override) ──────────────────────────────────────
if [ -z "$URL" ]; then
    if [ ! -f "$SETTINGS_PATH" ]; then
        fail "No settings.json found at $SETTINGS_PATH; pass --url manually."
        exit 3
    fi
    URL=$(python3 -c "import json,sys; d=json.load(open('$SETTINGS_PATH')); print(d.get('rtsp_url','').strip())")
fi

if [ -z "$URL" ]; then
    fail "No RTSP URL configured. Set one in Settings → Camera or pass --url."
    exit 3
fi

# ── Parse URL ───────────────────────────────────────────────────────────────
# Strip credentials for display + extract host / port / path. Done in python
# rather than awk so we get URL-aware parsing (handles ipv6, escaped chars).
read CAM_HOST CAM_PORT CAM_PATH SAFE_URL < <(python3 - "$URL" <<'PY'
import sys, urllib.parse
u = urllib.parse.urlparse(sys.argv[1])
host = u.hostname or '?'
port = u.port or 554
path = (u.path or '/') + (('?' + u.query) if u.query else '')
safe = u._replace(netloc=f"{u.hostname}:{port}" if u.hostname else '').geturl()
print(host, port, path, safe)
PY
)

header "Camera status — $CAM_HOST"

info "Saved URL:    ${SAFE_URL}"
info "Host:port:    ${CAM_HOST}:${CAM_PORT}"
info "Path:         ${CAM_PATH}"

# ── Network reachability ────────────────────────────────────────────────────
section "Network reachability"

if ping -c 2 -W 1 "$CAM_HOST" >/dev/null 2>&1; then
    PING_AVG=$(ping -c 4 -W 1 "$CAM_HOST" 2>/dev/null | awk -F'/' '/^rtt|^round-trip/ {print $5; exit}')
    success "Ping ${CAM_HOST}:    ${PING_AVG:-?} ms avg"
else
    fail "Ping ${CAM_HOST}:    no reply (camera off, wrong NIC, or ICMP blocked)"
fi

if (echo > "/dev/tcp/${CAM_HOST}/${CAM_PORT}") 2>/dev/null; then
    success "TCP ${CAM_PORT} (RTSP):   open"
else
    fail "TCP ${CAM_PORT} (RTSP):   closed (camera off, firewall, or wrong port)"
fi

# ── Most recent stream log line from the running container ──────────────────
section "Currently streaming (most recent connect)"

if command -v docker >/dev/null 2>&1 && docker compose ps --services --filter status=running 2>/dev/null | grep -q '^web$'; then
    LAST=$(docker compose logs --tail 200 web 2>&1 | grep -F '📹 Stream' | tail -1)
    if [ -n "$LAST" ]; then
        echo "  $LAST"
    else
        warn "No 📹 Stream log line found yet — start the stream in the dashboard once."
    fi
else
    warn "web container not running — skipping log lookup."
fi

# ── Probe one URL via cv2 inside the container ─────────────────────────────
# Returns: WxH@FPS codec=fourcc|<error>
probe_via_container() {
    local url="$1"
    if ! command -v docker >/dev/null 2>&1; then
        echo "no-docker"
        return 1
    fi
    if ! docker compose ps --services --filter status=running 2>/dev/null | grep -q '^web$'; then
        echo "container-down"
        return 1
    fi
    docker compose exec -T web python3 -c "
import sys, os, cv2
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
cap = cv2.VideoCapture(sys.argv[1], cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, ${PROBE_TIMEOUT}000])
if not cap.isOpened():
    print('OPEN_FAIL'); sys.exit(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
fps = cap.get(cv2.CAP_PROP_FPS) or 0
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
codec = bytes([(fourcc >> shift) & 0xFF for shift in (0,8,16,24)]).decode(errors='replace').strip('\x00') or '?'
ok = False
for _ in range(5):
    ok, _ = cap.read()
    if ok: break
cap.release()
print(f'{w}x{h}|{fps:.0f}|{codec}|{\"frames-ok\" if ok else \"no-frames\"}')
" "$url" 2>/dev/null | tail -1
}

# ── Saved URL probe ─────────────────────────────────────────────────────────
section "Saved URL probe (cv2 / FFmpeg over RTSP+TCP)"

RESULT=$(probe_via_container "$URL")
case "$RESULT" in
    no-docker)
        warn "docker not available; install docker or pass --url to a host with it." ;;
    container-down)
        warn "web container not running — start the stack first (./up.sh --force-cpu)." ;;
    OPEN_FAIL)
        fail "VideoCapture failed to open. Wrong URL, wrong credentials, or camera blocking RTSP." ;;
    "")
        warn "Empty result from probe — container may have errored silently." ;;
    *)
        IFS='|' read -r WH FPS CODEC FRAMES <<< "$RESULT"
        success "Resolution:   ${WH}"
        success "Native FPS:   ${FPS}"
        success "Codec:        ${CODEC}"
        success "First frame:  ${FRAMES}"
        ;;
esac

# ── Stream variant probing ──────────────────────────────────────────────────
if [ "$DO_VARIANTS" -eq 0 ]; then
    section "Skipping variant probes (--no-variants)"
else
    section "Stream variant probing — common camera URL patterns"
    info "Trying alternative paths on ${CAM_HOST}:${CAM_PORT} ..."
    info "(✅ = camera answered, ❌ = closed/refused/timeout, takes ~${PROBE_TIMEOUT}s per try)"
    echo ""

    # Extract user:pass if present in saved URL — otherwise reuse "anonymous"
    AUTH=$(python3 -c "
import urllib.parse, sys
u = urllib.parse.urlparse(sys.argv[1])
if u.username:
    pwd = u.password or ''
    print(f'{u.username}:{pwd}@')
else:
    print('')
" "$URL")

    BASE="rtsp://${AUTH}${CAM_HOST}:${CAM_PORT}"

    # Common URL conventions across IP-camera vendors. Most cameras only
    # respond to one family — the others time out cleanly.
    declare -a VARIANTS=(
        "/1                       (Generic OEM main-stream, ch.1)"
        "/11                      (Generic OEM sub-stream, ch.1)"
        "/12                      (Generic OEM 2nd sub-stream)"
        "/Streaming/Channels/101  (Hikvision main)"
        "/Streaming/Channels/102  (Hikvision sub)"
        "/cam/realmonitor?channel=1&subtype=0  (Dahua main)"
        "/cam/realmonitor?channel=1&subtype=1  (Dahua sub)"
        "/h264Preview_01_main     (Reolink main)"
        "/h264Preview_01_sub      (Reolink sub)"
        "/onvif1                  (ONVIF profile S, common)"
        "/live/main               (Foscam / Amcrest main)"
        "/live/sub                (Foscam / Amcrest sub)"
    )

    printf "%-50s  %-22s  %s\n" "Path" "Result" "Vendor hint"
    printf "%-50s  %-22s  %s\n" "----" "------" "-----------"

    for line in "${VARIANTS[@]}"; do
        # split path | label
        path=$(echo "$line" | awk '{print $1}')
        label=$(echo "$line" | sed 's/^[^ ]*  *//')

        full="${BASE}${path}"
        result=$(probe_via_container "$full")

        case "$result" in
            OPEN_FAIL|""|container-down|no-docker)
                printf "  %-48s  ${RED}%-22s${NC}  %s\n" "$path" "❌ no response" "$label"
                ;;
            *)
                IFS='|' read -r WH FPS CODEC FRAMES <<< "$result"
                printf "  %-48s  ${GREEN}%-22s${NC}  %s\n" "$path" "✅ ${WH} @ ${FPS}fps ${CODEC}" "$label"
                ;;
        esac
    done
fi

# ── Summary ─────────────────────────────────────────────────────────────────
section "Recommendations"

cat <<EOF
- The cv2/FFmpeg probe is what stream_handler does at runtime, so the
  numbers above match what the inference loop actually sees.
- If the saved URL probe succeeded but the FPS counter on the dashboard
  is meaningfully lower, the gap is downstream (inference, RTSP buffering)
  not the camera — see the FPS / latency analysis in the site visit report.
- If the camera reports something bigger than 25 fps native (e.g. 30 or 60),
  it's likely an NTSC-region camera or has been configured for higher-rate
  streaming — your inference floor (~10-15 ms on i7-10710U) means the AI
  could keep up to ~60 fps in principle, modulo PL1 thermal cap.
- If multiple variants succeeded above, you can pick the one that gives
  the best resolution/FPS tradeoff for your CPU. For sub-stream sites,
  prefer the smaller resolution variant with H.264 codec (not H.265).
- To switch streams: edit Settings → Camera → Default RTSP URL with the
  preferred variant, click Save, then Stop+Start the stream.

EOF

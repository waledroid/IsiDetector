#!/usr/bin/env bash
# ============================================================================
# IsiDetector — UDP Sort-Trigger Monitor
#
# Reads the current UDP target (from webapp/isitec_app/settings.json) and
# shows live tcpdump-captured datagrams as the inference stack emits them
# to the sorter PLC. Useful for:
#   - Pre-deploy verification that crossings actually emit UDP
#   - On-site debugging when the automate isn't receiving sort triggers
#   - Confirming retargeting (POST /api/udp) took effect
#
# Usage:
#   ./udp_monitor.sh                       # same as 'watch'
#   ./udp_monitor.sh status                # config readout only (no sudo)
#   ./udp_monitor.sh watch                 # live sniff (sudo auto-escalates)
#   ./udp_monitor.sh test                  # synthetic loopback test
#   ./udp_monitor.sh -h | --help           # this help
# ============================================================================

set -u

# ── Colour helpers (mirrors net.sh / remote.sh / cam_status.sh) ─────────────
if [ -t 1 ]; then
    GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BLUE='\033[0;34m'
    RED='\033[0;31m';   BOLD='\033[1m';     NC='\033[0m'
else
    GREEN=''; YELLOW=''; CYAN=''; BLUE=''; RED=''; BOLD=''; NC=''
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Argument parsing ────────────────────────────────────────────────────────
CMD=""
ORIG_ARGS=("$@")
while [ $# -gt 0 ]; do
    case "$1" in
        status|watch|test|help) CMD="$1" ;;
        -h|--help)              CMD="help" ;;
        *) fail "unknown argument: $1"; CMD="help" ;;
    esac
    shift
done
CMD="${CMD:-watch}"

# ── Sudo escalation (watch + test need root for tcpdump / raw sockets) ──────
case "$CMD" in
    watch|test)
        if [ "${EUID:-$(id -u)}" -ne 0 ]; then
            info "'$CMD' needs root for packet capture — re-executing with sudo..."
            exec sudo -E "$0" "${ORIG_ARGS[@]}"
        fi
        ;;
esac

# ── Read current UDP target ─────────────────────────────────────────────────
# Priority: settings.json (operator-tunable, persists across reboot)
#        → container env (UDP_HOST / UDP_PORT in docker-compose env)
#        → built-in default (10.0.0.2:9502 — the canonical Isitec site target).
read_udp_target() {
    local settings_file="${REPO_ROOT}/webapp/isitec_app/settings.json"
    local host="" port=""
    if [ -r "$settings_file" ]; then
        host=$(grep -oE '"udp_host"[[:space:]]*:[[:space:]]*"[^"]*"' "$settings_file" \
            | head -1 | sed -E 's/.*"udp_host"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/')
        port=$(grep -oE '"udp_port"[[:space:]]*:[[:space:]]*[0-9]+' "$settings_file" \
            | head -1 | sed -E 's/.*"udp_port"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/')
    fi
    if [ -z "$host" ] && command -v docker >/dev/null 2>&1; then
        host=$(docker exec deploy-web-1 sh -c 'printf "%s" "${UDP_HOST:-}"' 2>/dev/null)
    fi
    if [ -z "$port" ] && command -v docker >/dev/null 2>&1; then
        port=$(docker exec deploy-web-1 sh -c 'printf "%s" "${UDP_PORT:-}"' 2>/dev/null)
    fi
    [ -z "$host" ] && host="10.0.0.2"
    [ -z "$port" ] && port="9502"
    echo "${host}|${port}"
}

# ── Print full network picture for the configured target ────────────────────
# Sets TARGET_IPV4, TARGET_SRC, TARGET_DEV, TARGET_PORT globals as a side effect
# so cmd_watch can pass them into the pretty-printer.
TARGET_IPV4=""
TARGET_SRC=""
TARGET_DEV=""
TARGET_PORT=""

print_target_info() {
    local host="$1" port="$2"
    header "UDP target configuration"

    printf "  Target host:port:    %s:%s\n" "$host" "$port"
    printf "  Source:              %s\n" "${REPO_ROOT}/webapp/isitec_app/settings.json"

    # Literal IP vs hostname
    local ipv4=""
    if [[ "$host" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        ipv4="$host"
        printf "  Resolved IP:         %s (literal IPv4)\n" "$ipv4"
    else
        ipv4=$(getent ahosts "$host" 2>/dev/null \
            | awk '/STREAM/ && $1 ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print $1; exit}')
        if [ -n "$ipv4" ]; then
            printf "  Resolved IP:         %s (from DNS)\n" "$ipv4"
        else
            warn "Cannot resolve $host to an IPv4 address."
            printf "  Resolved IP:         (unknown)\n"
            return 1
        fi
    fi

    # Route to the target
    local route_line src dev gateway
    route_line=$(ip route get "$ipv4" 2>/dev/null | head -1)
    if [ -z "$route_line" ]; then
        warn "No route to $ipv4."
        return 1
    fi
    src=$(echo "$route_line" | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
    dev=$(echo "$route_line" | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')
    gateway=$(echo "$route_line" | awk '{for(i=1;i<=NF;i++) if($i=="via"){print $(i+1); exit}}')

    echo ""
    echo "  Local outbound:"
    printf "    Interface:         %s\n" "${dev:-?}"
    printf "    Source IP:         %s\n" "${src:-?}"
    if [ -n "$dev" ]; then
        local cidr mac
        cidr=$(ip -4 -o addr show dev "$dev" 2>/dev/null | awk '{print $4}' | head -1)
        mac=$(ip -o link show "$dev" 2>/dev/null | awk -F'link/ether ' 'NF>1{print $2}' | awk '{print $1}')
        printf "    CIDR:              %s\n" "${cidr:-?}"
        printf "    MAC:               %s\n" "${mac:-?}"
    fi
    if [ -n "$gateway" ]; then
        printf "    Gateway:           %s\n" "$gateway"
    else
        printf "    Gateway:           (direct connection — same subnet)\n"
    fi

    # Reachability
    echo ""
    if ping -c 1 -W 2 "$ipv4" >/dev/null 2>&1; then
        success "Target $ipv4 reachable (ICMP ping ok)"
    else
        warn "Target $ipv4 did NOT respond to ICMP — could be firewall or PLC offline."
        warn "  UDP may still work; ICMP and UDP are separate paths."
    fi
    echo ""

    TARGET_IPV4="$ipv4"
    TARGET_SRC="${src:-}"
    TARGET_DEV="${dev:-}"
    TARGET_PORT="$port"
}

# ── cmd: status ─────────────────────────────────────────────────────────────
cmd_status() {
    local target host port
    target=$(read_udp_target)
    host="${target%|*}"; port="${target#*|}"
    print_target_info "$host" "$port"
}

# ── cmd: watch ──────────────────────────────────────────────────────────────
# tcpdump filtered on `udp port <N>` → pretty-printed via inline python.
# Captures both directions (OUT from us, IN to us, OTHER for unrelated chatter).
cmd_watch() {
    local target host port
    target=$(read_udp_target)
    host="${target%|*}"; port="${target#*|}"
    print_target_info "$host" "$port" || exit 1

    if ! command -v tcpdump >/dev/null 2>&1; then
        fail "tcpdump not installed.  sudo apt install -y tcpdump"
        exit 2
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        fail "python3 not installed (parser needs it).  sudo apt install -y python3"
        exit 2
    fi

    header "Live UDP datagrams (port $port)"
    echo -e "  Direction key:  ${GREEN}OUT${NC} (leaving us) | ${BLUE}IN${NC} (arriving) | ${YELLOW}OTHER${NC}"
    echo "  Press Ctrl+C to stop. Final summary prints on exit."
    echo ""

    # `-i any` listens on every interface (incl. loopback); `-l` = line-buffered;
    # `-tttt` = full timestamp; `-A` = ASCII payload after the header.
    tcpdump -i any -l -n -tttt -A "udp port $port" 2>/dev/null \
        | _pretty_print_loop
}

# Pretty-printer: tcpdump emits one "header" line per packet followed by 0+
# payload lines (with mixed binary header bytes + ASCII JSON payload). We
# accumulate per-packet, extract the JSON, and emit one coloured line.
_pretty_print_loop() {
    python3 - "$TARGET_SRC" "$TARGET_IPV4" "$TARGET_PORT" \
              "$GREEN" "$BLUE" "$YELLOW" "$NC" <<'PYEOF'
import sys, re, time

our_ip   = sys.argv[1]
peer_ip  = sys.argv[2]
port     = sys.argv[3]
GREEN, BLUE, YELLOW, NC = sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]

# Header line shape (tcpdump -tttt -A):
#   2026-05-13 14:23:45.312847 IP 10.0.0.5.48372 > 10.0.0.2.9502: UDP, length 60
HDR_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+IP\s+'
    r'(?P<src>[\d\.]+)\.(?P<sport>\d+)\s+>\s+'
    r'(?P<dst>[\d\.]+)\.(?P<dport>\d+):\s+UDP'
)

count = out_count = in_count = other_count = 0
start_t = time.time()
last_seen_t = None
current = None

def flush(pkt):
    global count, out_count, in_count, other_count, last_seen_t
    if not pkt:
        return
    hdr = pkt['hdr']
    body = ''.join(pkt['body'])
    # Pull just the JSON object out of the body (tcpdump prefixes ~28 bytes
    # of IP+UDP header rendered as ASCII garbage).
    m = re.search(r'\{[^}]*\}', body)
    payload = m.group(0) if m else body.strip()[-80:].replace('\n', ' ')

    if hdr['src'] == our_ip:    direction, color = 'OUT  ', GREEN
    elif hdr['dst'] == our_ip:  direction, color = 'IN   ', BLUE
    else:                       direction, color = 'OTHER', YELLOW

    short_ts = hdr['ts'].split(' ')[1][:12]  # HH:MM:SS.mmm
    line = (f"[{short_ts}] {color}{direction}{NC}  "
            f"{hdr['src']}:{hdr['sport']} → {hdr['dst']}:{hdr['dport']}  {payload}")
    # Clear the inline stats footer before printing the packet line
    sys.stdout.write('\r\033[K')
    print(line)
    sys.stdout.flush()

    count += 1
    if   direction == 'OUT  ':  out_count   += 1
    elif direction == 'IN   ':  in_count    += 1
    else:                       other_count += 1
    last_seen_t = time.time()
    _print_stats()

def _print_stats():
    elapsed = max(0.001, time.time() - start_t)
    rate = count / elapsed
    last_ago = "—" if last_seen_t is None else f"{time.time()-last_seen_t:.0f}s ago"
    footer = (f"  → total: {count} "
              f"(OUT {out_count} | IN {in_count} | OTHER {other_count})  "
              f"| rate: {rate:.2f}/s  | last: {last_ago}")
    sys.stdout.write('\r\033[K' + footer)
    sys.stdout.flush()

try:
    for raw in sys.stdin:
        line = raw.rstrip('\n')
        m = HDR_RE.match(line)
        if m:
            flush(current)
            current = {'hdr': m.groupdict(), 'body': []}
        elif current is not None:
            current['body'].append(line)
    flush(current)
except KeyboardInterrupt:
    pass
finally:
    sys.stdout.write('\r\033[K')
    elapsed = max(0.001, time.time() - start_t)
    rate = count / elapsed
    print(f"\n──────────────────────────────────────────────────")
    print(f"  Summary: {count} datagram(s) in {elapsed:.1f}s ({rate:.2f}/s avg)")
    print(f"    OUT (from us):   {out_count}")
    print(f"    IN  (to us):     {in_count}")
    print(f"    OTHER:           {other_count}")
PYEOF
}

# ── cmd: test ───────────────────────────────────────────────────────────────
# Binds a temporary listener on 127.0.0.1:54321, sends one synthetic datagram
# via nc -u, verifies receipt. Doesn't touch the real UDP target — strictly a
# host-level UDP sanity check. Useful when watch shows zero traffic and you
# need to rule out "is UDP even working on this machine".
cmd_test() {
    local target host port
    target=$(read_udp_target)
    host="${target%|*}"; port="${target#*|}"
    print_target_info "$host" "$port" || true

    header "Synthetic loopback test"

    if ! command -v python3 >/dev/null 2>&1; then
        fail "python3 required for this test (binds + receives)."
        exit 2
    fi
    if ! command -v nc >/dev/null 2>&1 && ! command -v ncat >/dev/null 2>&1; then
        fail "nc / ncat required to send the test datagram.  sudo apt install -y netcat-openbsd"
        exit 2
    fi
    local NC_BIN; NC_BIN=$(command -v nc || command -v ncat)

    info "Binding listener on 127.0.0.1:54321 (3 s timeout)..."
    local listener_out; listener_out=$(mktemp)
    python3 - <<'PYEOF' >"$listener_out" 2>&1 &
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(3.0)
s.bind(('127.0.0.1', 54321))
try:
    data, addr = s.recvfrom(2048)
    print(f"RECEIVED {len(data)} bytes from {addr[0]}:{addr[1]}:")
    print(f"  {data.decode(errors='replace')}")
except socket.timeout:
    print("TIMEOUT — no datagram received")
finally:
    s.close()
PYEOF
    local listener_pid=$!
    sleep 0.5

    info "Sending synthetic datagram via $NC_BIN..."
    local payload='{"class":"test","id":0,"ts":"'"$(date -Iseconds)"'"}'
    echo -n "$payload" | "$NC_BIN" -u -w 1 127.0.0.1 54321 2>/dev/null || true

    wait $listener_pid 2>/dev/null || true
    cat "$listener_out" | sed 's/^/    /'
    rm -f "$listener_out"

    echo ""
    if grep -q '^RECEIVED' "$listener_out" 2>/dev/null; then
        : # success path already printed by the python listener
    elif grep -q "RECEIVED" /dev/null 2>&1; then
        :
    fi

    info "If RECEIVED above → host-level UDP loopback works end to end."
    info "For a real-app test: ./udp_monitor.sh watch + trigger a parcel crossing."
}

# ── cmd: help ───────────────────────────────────────────────────────────────
cmd_help() {
    sed -n '2,17p' "$0" | sed 's/^# \?//'
}

# ── Dispatch ────────────────────────────────────────────────────────────────
case "$CMD" in
    status) cmd_status ;;
    watch)  cmd_watch ;;
    test)   cmd_test ;;
    help|*) cmd_help ;;
esac

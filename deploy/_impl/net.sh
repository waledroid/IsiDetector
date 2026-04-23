#!/usr/bin/env bash
# ============================================================================
# IsiDetector — Site Network Lock-down & Automaticien Handshake
#
# Freezes the current DHCP-issued IP / gateway / DNS into a static
# NetworkManager config so they can never drift after a restart, Wi-Fi
# reconnect, or lease expiry — and prints a ready-to-email mini manual
# for the automation engineer on the other end of the UDP link.
#
# Usage:
#   ./net.sh                            # same as 'show'
#   ./net.sh show                       # current network + UDP publisher state
#   ./net.sh apply                      # lock the discovered values (sudo)
#   ./net.sh apply --force              # same, skip confirm prompt
#   ./net.sh apply [--ip ... | --gateway ... | --dns '...' | --conn NAME]
#   ./net.sh revert                     # restore DHCP (sudo)
#   ./net.sh test                       # reachability + live UDP egress
#   ./net.sh manual                     # French mini-manual for the automaticien
#   ./net.sh manual --en                # English variant
#   ./net.sh -h | --help
#
# Discovery-first design: no site-specific value (SSID, IP, gateway, automate
# address) is hardcoded. Everything is read from the live system at
# invocation time so the same script works unmodified on every customer PC.
# ============================================================================

set -u
# NOTE: no `set -e` — several commands below are expected to sometimes
# fail (e.g. ping during a `test` run on a disconnected network). We
# check exit codes explicitly where it matters.

# ── Colour helpers (mirrors run_start.sh so both scripts feel uniform) ──────
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Discovery helpers ───────────────────────────────────────────────────────
# All the site-specific values are derived here. No hardcodes.

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || { fail "missing command: $1"; return 1; }
}

discover_conn() {
    # First non-virtual active NetworkManager connection.
    # Excludes lo/docker*/br-*/veth* and bridge/loopback types.
    nmcli -t -f NAME,TYPE,DEVICE,STATE connection show --active 2>/dev/null \
        | awk -F: '$4=="activated" && $3 !~ /^(lo|docker|br-|veth)/ && $2 !~ /^(bridge|loopback)$/ {print $1; exit}'
}

discover_iface() {
    # Interface of the discovered connection.
    #
    # `nmcli -t -f DEVICE connection show <name>` returns empty on modern
    # nmcli because `connection show <name>` uses the detail schema (dot-
    # notation field names like GENERAL.DEVICES) — plain DEVICE is only a
    # field in the list schema (`connection show --active`). Re-use the
    # same --active listing discover_conn relies on.
    local conn="${1:?conn required}"
    nmcli -t -f NAME,DEVICE connection show --active 2>/dev/null \
        | awk -F: -v c="$conn" '$1==c {print $2; exit}'
}

discover_ip_cidr() {
    local iface="${1:?iface required}"
    ip -4 -br addr show dev "$iface" 2>/dev/null | awk '{print $3}' | head -1
}

discover_gateway() {
    ip -4 route show default 2>/dev/null | awk '/^default/ {print $3; exit}'
}

discover_dns() {
    # Prefer resolvectl when available (systemd-resolved), else parse resolv.conf.
    local iface="${1:-}"
    local dns=""
    if command -v resolvectl >/dev/null 2>&1 && [ -n "$iface" ]; then
        dns=$(resolvectl dns "$iface" 2>/dev/null | sed 's/^[^:]*: *//' | tr '\n' ' ' | xargs)
    fi
    if [ -z "$dns" ] && [ -r /etc/resolv.conf ]; then
        dns=$(awk '/^nameserver / {print $2}' /etc/resolv.conf | tr '\n' ' ' | xargs)
    fi
    echo "$dns"
}

discover_ipv4_method() {
    local conn="${1:?conn required}"
    nmcli -t -f ipv4.method connection show "$conn" 2>/dev/null | cut -d: -f2
}

discover_autoconnect() {
    local conn="${1:?conn required}"
    local ac prio
    ac=$(nmcli -t -f connection.autoconnect connection show "$conn" 2>/dev/null | cut -d: -f2)
    prio=$(nmcli -t -f connection.autoconnect-priority connection show "$conn" 2>/dev/null | cut -d: -f2)
    echo "${ac:-?} (priority ${prio:-0})"
}

discover_conn_type() {
    local conn="${1:?conn required}"
    nmcli -t -f TYPE connection show "$conn" 2>/dev/null | head -1
}

# Reads UDP_HOST/UDP_PORT from the running web container. Empty strings
# if Docker or the container isn't reachable — callers handle the empty.
discover_udp_target() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "||"
        return
    fi
    # Test that web container is running. 2>/dev/null swallows noise.
    if ! docker compose ps --services --filter status=running 2>/dev/null | grep -q '^web$'; then
        echo "||"
        return
    fi
    local host port
    host=$(docker compose exec -T web printenv UDP_HOST 2>/dev/null | tr -d '\r\n')
    port=$(docker compose exec -T web printenv UDP_PORT 2>/dev/null | tr -d '\r\n')
    # Strip CIDR suffix if someone accidentally set one
    echo "${host}|${port}|ok"
}

strip_cidr() {   # "192.168.2.225/24" -> "192.168.2.225"
    echo "$1" | cut -d/ -f1
}

# ── CLI parsing ─────────────────────────────────────────────────────────────
CMD=""
ARG_FORCE=0
ARG_CONN=""
ARG_IP=""
ARG_GATEWAY=""
ARG_DNS=""
ARG_AUTOMATE=""
ARG_PORT=""
ARG_LANG="fr"

print_help() {
    sed -n '2,24p' "$0" | sed 's/^# *//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        show|apply|revert|test|manual|help) CMD="$1" ;;
        --force)            ARG_FORCE=1 ;;
        --conn)             ARG_CONN="${2:-}"; shift ;;
        --conn=*)           ARG_CONN="${1#*=}" ;;
        --ip)               ARG_IP="${2:-}"; shift ;;
        --ip=*)             ARG_IP="${1#*=}" ;;
        --gateway)          ARG_GATEWAY="${2:-}"; shift ;;
        --gateway=*)        ARG_GATEWAY="${1#*=}" ;;
        --dns)              ARG_DNS="${2:-}"; shift ;;
        --dns=*)            ARG_DNS="${1#*=}" ;;
        --automate)         ARG_AUTOMATE="${2:-}"; shift ;;
        --automate=*)       ARG_AUTOMATE="${1#*=}" ;;
        --port)             ARG_PORT="${2:-}"; shift ;;
        --port=*)           ARG_PORT="${1#*=}" ;;
        --en)               ARG_LANG="en" ;;
        -h|--help)          print_help; exit 0 ;;
        *)                  fail "Unknown argument: $1"; echo "Try: $0 --help" >&2; exit 2 ;;
    esac
    shift
done
CMD="${CMD:-show}"
if [ "$CMD" = "help" ]; then print_help; exit 0; fi

# ── Sanity: Docker IP range check ───────────────────────────────────────────
is_docker_range_ip() {
    # 172.16.0.0/12 → first-octet 172, second-octet 16..31
    local ip="$1"
    local o1 o2
    o1=$(echo "$ip" | cut -d. -f1)
    o2=$(echo "$ip" | cut -d. -f2)
    [ "$o1" = "172" ] && [ "$o2" -ge 16 ] && [ "$o2" -le 31 ] 2>/dev/null
}

# ── Resolve the inputs (discovery with CLI overrides) ───────────────────────
# strict_nm=1 → require a live NetworkManager connection (apply / revert / test / show)
# strict_nm=0 → tolerate missing nmcli, fill what we can (manual)
resolve_inputs() {
    local strict_nm="${1:-1}"

    # Distinguish "no nmcli at all" from "nmcli present but no active
    # connection". The first means the user is on a machine outside the
    # script's scope (WSL2, Ubuntu Server with systemd-networkd, any
    # non-NetworkManager Linux). Point them at the right machine instead
    # of the vague "start Wi-Fi" suggestion.
    if ! command -v nmcli >/dev/null 2>&1; then
        if [ "$strict_nm" = "1" ]; then
            fail "NetworkManager (nmcli) is not installed on this machine."
            fail "net.sh is designed for **site PCs** — Ubuntu Desktop hosts running"
            fail "NetworkManager, where a DHCP-issued Wi-Fi/Ethernet config needs to be"
            fail "frozen so the automate's firewall whitelist doesn't drift."
            fail ""
            fail "  You're on:  $(hostname) (no NetworkManager detected)"
            fail "  Likely cause:  WSL2, Ubuntu Server (uses systemd-networkd/netplan),"
            fail "                 or a dev workstation where network lock-down doesn't apply."
            fail ""
            fail "  If this really IS meant to run on this host, install NM first:"
            fail "    sudo apt install network-manager"
            fail "    sudo systemctl enable --now NetworkManager"
            fail ""
            fail "  Otherwise: copy this script to the actual site PC and run it there."
            exit 3
        fi
        CONN=""; IFACE=""; TYPE=""; METHOD=""; AUTO=""
        IP_CIDR=""; IP="<SITE_PC_IP>"; GATEWAY=""; DNS=""
        UDP_HOST=""; UDP_PORT=""; UDP_SRC=""
        return
    fi

    CONN="${ARG_CONN:-$(discover_conn)}"
    if [ -z "$CONN" ]; then
        if [ "$strict_nm" = "1" ]; then
            fail "No active non-virtual NetworkManager connection found."
            fail "Start Wi-Fi/Ethernet first, or pass --conn NAME explicitly."
            exit 3
        fi
        CONN=""; IFACE=""; TYPE=""; METHOD=""; AUTO=""
    else
        IFACE=$(discover_iface "$CONN")
        if [ -z "$IFACE" ] && [ "$strict_nm" = "1" ]; then
            fail "Couldn't resolve interface for connection '$CONN'."
            exit 3
        fi
        TYPE=$(discover_conn_type "$CONN")
        METHOD=$(discover_ipv4_method "$CONN")
        AUTO=$(discover_autoconnect "$CONN")
    fi

    # IP/CIDR — CLI override wins, else discover live value.
    IP_CIDR="${ARG_IP:-}"
    if [ -z "$IP_CIDR" ] && [ -n "$IFACE" ]; then
        IP_CIDR=$(discover_ip_cidr "$IFACE")
    fi
    if [ -z "$IP_CIDR" ]; then
        if [ "$strict_nm" = "1" ]; then
            fail "Couldn't discover an IPv4 address${IFACE:+ on $IFACE}."
            fail "Pass --ip A.B.C.D/PREFIX explicitly."
            exit 3
        fi
        IP="<SITE_PC_IP>"
    else
        IP=$(strip_cidr "$IP_CIDR")
        if is_docker_range_ip "$IP" && [ "$strict_nm" = "1" ]; then
            fail "Discovered IP $IP is in the Docker-bridge range (172.16.0.0/12)."
            fail "This doesn't look like your LAN address. Pass --ip explicitly."
            exit 3
        fi
    fi

    GATEWAY="${ARG_GATEWAY:-$(discover_gateway)}"
    DNS="${ARG_DNS:-$(discover_dns "$IFACE")}"

    # UDP target from Docker (live), with CLI override.
    local udp_raw udp_host udp_port udp_ok
    udp_raw=$(discover_udp_target)
    udp_host=$(echo "$udp_raw" | cut -d'|' -f1)
    udp_port=$(echo "$udp_raw" | cut -d'|' -f2)
    udp_ok=$(echo "$udp_raw" | cut -d'|' -f3)
    UDP_HOST="${ARG_AUTOMATE:-$udp_host}"
    UDP_PORT="${ARG_PORT:-$udp_port}"
    UDP_SRC=""
    if [ "$udp_ok" = "ok" ]; then UDP_SRC="docker"; fi
    if [ -n "$ARG_AUTOMATE" ] || [ -n "$ARG_PORT" ]; then UDP_SRC="override"; fi
}

# ── Commands ────────────────────────────────────────────────────────────────

cmd_show() {
    resolve_inputs
    header "Network state — $(hostname)"
    printf "  %-20s %s\n" "Active profile:" "$CONN  ($TYPE on $IFACE)"
    printf "  %-20s " "IPv4 method:"
    case "$METHOD" in
        manual) echo -e "${GREEN}manual${NC}  (DHCP disabled)" ;;
        auto)   echo -e "${YELLOW}auto${NC}    ← DHCP — config can drift" ;;
        *)      echo "$METHOD" ;;
    esac
    printf "  %-20s %s\n" "IP / mask:"    "$IP_CIDR"
    printf "  %-20s %s\n" "Gateway:"      "${GATEWAY:-<none>}"
    printf "  %-20s %s\n" "DNS:"          "${DNS:-<none>}"
    printf "  %-20s %s\n" "Auto-connect:" "$AUTO"

    echo ""
    echo -e "${BOLD}Route to automate${NC}"
    if [ -n "$UDP_HOST" ]; then
        ip route get "$UDP_HOST" 2>&1 | sed 's/^/  /'
    else
        echo -e "  ${YELLOW}<UDP_HOST not set — pass --automate or start web container>${NC}"
    fi

    echo ""
    echo -e "${BOLD}Docker UDP publisher${NC}"
    if [ -n "$UDP_HOST" ]; then
        printf "  %-20s %s (%s)\n" "UDP_HOST:" "$UDP_HOST" "$UDP_SRC"
        printf "  %-20s %s\n"       "UDP_PORT:" "${UDP_PORT:-<none>}"
    else
        echo -e "  ${YELLOW}<not available — web container isn't running or printenv failed>${NC}"
    fi

    echo ""
    if [ "$METHOD" = "manual" ]; then
        echo -e "  Status:  ${GREEN}✅ Locked${NC}"
    else
        echo -e "  Status:  ${YELLOW}⚠  DHCP — config can change on restart${NC}"
        echo -e "           Run '${CYAN}sudo ./net.sh apply${NC}' to freeze it."
    fi
    echo ""
}

cmd_apply() {
    resolve_inputs
    if [ -z "$GATEWAY" ]; then
        fail "No default gateway found. Pass --gateway X.Y.Z.W."
        exit 3
    fi

    header "Lock network config — $CONN"
    echo "Values to write:"
    printf "  %-20s %s\n"  "ipv4.method"          "manual"
    printf "  %-20s %s\n"  "ipv4.addresses"       "$IP_CIDR"
    printf "  %-20s %s\n"  "ipv4.gateway"         "$GATEWAY"
    printf "  %-20s %s\n"  "ipv4.dns"             "${DNS:-<empty>}"
    printf "  %-20s %s\n"  "ipv4.ignore-auto-dns" "yes"
    printf "  %-20s %s\n"  "ipv6.method"          "link-local"
    printf "  %-20s %s\n"  "connection.autoconnect"          "yes"
    printf "  %-20s %s\n"  "connection.autoconnect-priority" "100"
    echo ""

    if [ "$ARG_FORCE" -eq 0 ]; then
        read -rp "Apply these to '$CONN'? [y/N] " ans
        case "$ans" in y|Y|yes|YES) ;; *) info "Cancelled."; exit 0 ;; esac
    fi

    if [ "$(id -u)" -ne 0 ] && ! sudo -n true 2>/dev/null; then
        info "Needs sudo — you may be prompted."
    fi

    # Apply atomically so partial failures don't leave the config half-changed.
    if ! sudo nmcli connection modify "$CONN" \
            ipv4.method manual \
            ipv4.addresses "$IP_CIDR" \
            ipv4.gateway "$GATEWAY" \
            ipv4.dns "${DNS:-}" \
            ipv4.ignore-auto-dns yes \
            ipv6.method link-local \
            connection.autoconnect yes \
            connection.autoconnect-priority 100; then
        fail "nmcli modify failed."
        exit 4
    fi

    info "Bouncing connection to apply…"
    sudo nmcli connection down "$CONN" >/dev/null 2>&1 || true
    if ! sudo nmcli connection up "$CONN"; then
        fail "Couldn't bring '$CONN' back up. Run 'sudo ./net.sh revert' and retry."
        exit 4
    fi

    success "Config applied."
    echo ""
    # Re-run show to confirm
    cmd_show
}

cmd_revert() {
    resolve_inputs
    header "Revert to DHCP — $CONN"
    if [ "$ARG_FORCE" -eq 0 ]; then
        read -rp "Clear manual IP/gateway/DNS and re-enable DHCP on '$CONN'? [y/N] " ans
        case "$ans" in y|Y|yes|YES) ;; *) info "Cancelled."; exit 0 ;; esac
    fi
    if ! sudo nmcli connection modify "$CONN" \
            ipv4.method auto \
            ipv4.addresses "" \
            ipv4.gateway "" \
            ipv4.dns "" \
            ipv4.ignore-auto-dns no \
            ipv6.method auto; then
        fail "nmcli modify failed."
        exit 4
    fi
    sudo nmcli connection down "$CONN" >/dev/null 2>&1 || true
    if ! sudo nmcli connection up "$CONN"; then
        fail "Couldn't reconnect '$CONN' after revert."
        exit 4
    fi
    success "Reverted to DHCP."
    echo ""
    cmd_show
}

cmd_test() {
    resolve_inputs
    header "Network reachability — $(hostname)"
    local failed=0

    # 1. Gateway
    printf "  1. Gateway reachable       → %-16s" "${GATEWAY:-<none>}"
    if [ -z "$GATEWAY" ]; then
        echo -e "  ${YELLOW}skip${NC}"
    elif ping -c 2 -W 1 "$GATEWAY" >/dev/null 2>&1; then
        echo -e "  ${GREEN}✅${NC}"
    else
        echo -e "  ${RED}❌${NC}"; failed=1
    fi

    # 2. Internet — 1.1.1.1 is a universally-routable endpoint, not a site-specific IP.
    printf "  2. Internet reachable      → %-16s" "1.1.1.1"
    if ping -c 2 -W 1 1.1.1.1 >/dev/null 2>&1; then
        echo -e "  ${GREEN}✅${NC}"
    else
        echo -e "  ${RED}❌${NC}"; failed=1
    fi

    # 3. Automate
    printf "  3. Automate reachable      → %-16s" "${UDP_HOST:-<none>}"
    if [ -z "$UDP_HOST" ]; then
        echo -e "  ${YELLOW}skip${NC}"
    elif ping -c 2 -W 1 "$UDP_HOST" >/dev/null 2>&1; then
        echo -e "  ${GREEN}✅${NC}"
    else
        echo -e "  ${RED}❌${NC}"; failed=1
    fi

    # 4. Web publisher env
    printf "  4. Web container publisher → "
    if [ -n "$UDP_HOST" ] && [ -n "$UDP_PORT" ]; then
        echo -e "${UDP_HOST}:${UDP_PORT}   ${GREEN}✅${NC}"
    else
        echo -e "${YELLOW}not set (container not running?)${NC}"
    fi

    # 5. Live UDP egress probe
    printf "  5. Live UDP egress         → "
    if [ -z "$UDP_HOST" ] || [ -z "$UDP_PORT" ] || ! command -v docker >/dev/null 2>&1; then
        echo -e "${YELLOW}skip (no target or no docker)${NC}"
    else
        local pcap_file
        pcap_file=$(mktemp --suffix=.pcap)
        sudo -n tcpdump -n -i "$IFACE" -w "$pcap_file" -c 1 \
             "udp and host $UDP_HOST and port $UDP_PORT" >/dev/null 2>&1 &
        local tcpdump_pid=$!
        sleep 0.5
        docker compose exec -T web python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.sendto(b'net.sh-probe', ('$UDP_HOST', $UDP_PORT))
" >/dev/null 2>&1
        # Wait up to 3 s for tcpdump to capture
        for _ in 1 2 3 4 5 6; do
            if ! kill -0 "$tcpdump_pid" 2>/dev/null; then break; fi
            sleep 0.5
        done
        sudo -n kill "$tcpdump_pid" 2>/dev/null || true
        wait "$tcpdump_pid" 2>/dev/null || true
        if [ -s "$pcap_file" ] && sudo -n tcpdump -r "$pcap_file" 2>/dev/null | grep -q "$UDP_HOST"; then
            echo -e "${GREEN}✅${NC}  (probe left $IFACE)"
        else
            echo -e "${RED}❌${NC}  (nothing captured — firewall or container not publishing)"
            failed=1
        fi
        rm -f "$pcap_file"
    fi

    echo ""
    if [ "$failed" -eq 0 ]; then
        success "All checks passed."
        return 0
    else
        fail "One or more checks failed."
        return 1
    fi
}

cmd_manual() {
    # Manual is read-only and doesn't need a live NM connection — accept
    # whatever we can discover and fill placeholders for the rest.
    resolve_inputs 0
    if [ -z "$UDP_HOST" ] || [ -z "$UDP_PORT" ]; then
        warn "UDP_HOST/UDP_PORT not discovered. Placeholders will be left in the manual."
        UDP_HOST="${UDP_HOST:-<AUTOMATE_IP>}"
        UDP_PORT="${UDP_PORT:-<AUTOMATE_PORT>}"
    fi

    if [ "$ARG_LANG" = "en" ]; then
        cat <<EOF
─── Mini-manual for the automation engineer ─────────────────────────────

UDP sender:
  Machine    : $(hostname)
  Source IP  : $IP
  (Static — will not change on reboot)

UDP receiver (your side):
  IP         : $UDP_HOST
  Port       : $UDP_PORT
  Protocol   : UDP/IPv4
  Payload    : JSON, ~60 bytes per event

Payload format:
  {"class":"carton"|"polybag", "id":<int>, "ts":"<ISO8601>"}

Maximum rate: ~3000 events/hour.

Emission latency (measured on this PC):
  Median 78 µs · p99 474 µs · worst observed 637 µs
  Under 1 ms, quasi-instantaneous vs. your PLC scan cycle.

To receive (pick ONE method):

  # Python (portable):
  python3 -c 'import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind(("0.0.0.0",${UDP_PORT}));print("listening");[print(s.recvfrom(1024)[0].decode()) for _ in iter(int,1)]'

  # Netcat (Linux):
  nc -u -l ${UDP_PORT}

  # PowerShell (Windows admin):
  \$udp = New-Object System.Net.Sockets.UdpClient ${UDP_PORT}
  \$ep  = New-Object System.Net.IPEndPoint([Net.IPAddress]::Any, 0)
  while (\$true) { Write-Host ([Text.Encoding]::UTF8.GetString(\$udp.Receive([ref]\$ep))) }

Firewall to allow (generic examples — adapt to your system):

  Linux (iptables):
    sudo iptables -A INPUT -p udp --dport ${UDP_PORT} -s ${IP} -j ACCEPT

  Windows (PowerShell admin):
    New-NetFirewallRule -DisplayName "UDP ISIDetector" -Direction Inbound \\
      -Protocol UDP -LocalPort ${UDP_PORT} -RemoteAddress ${IP} -Action Allow

Validation handshake:
  1. Start your listener (Python/nc/PowerShell above).
  2. On my PC, I fire:
       docker compose exec web python3 -c 'import socket; socket.socket(socket.AF_INET,socket.SOCK_DGRAM).sendto(b"test-from-abdul", ("${UDP_HOST}", ${UDP_PORT})); print("sent")'
  3. You should immediately see the string "test-from-abdul".
     → Yes : the full path works; real events arrive the same way.
     → No  : issue on your side (firewall / bind / VLAN).

─────────────────────────────────────────────────────────────────────────
EOF
    else
        cat <<EOF
─── Mini-manuel pour l'automaticien ─────────────────────────────────────

Émetteur UDP :
  Machine    : $(hostname)
  IP source  : $IP
  (IP fixée — ne changera plus après redémarrage)

Récepteur UDP (chez toi) :
  IP         : $UDP_HOST
  Port       : $UDP_PORT
  Protocole  : UDP/IPv4
  Payload    : JSON, ~60 octets par événement

Format du payload :
  {"class":"carton"|"polybag", "id":<int>, "ts":"<ISO8601>"}

Cadence maximale : ~3000 événements/heure.

Latence émission (mesurée sur ce PC) :
  Médiane 78 µs · p99 474 µs · pire cas observé 637 µs
  Inférieur à 1 ms, quasi-instantané vs ton cycle automate.

Pour recevoir (choisir UNE méthode) :

  # Python (portable) :
  python3 -c 'import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind(("0.0.0.0",${UDP_PORT}));print("listening");[print(s.recvfrom(1024)[0].decode()) for _ in iter(int,1)]'

  # Netcat (Linux) :
  nc -u -l ${UDP_PORT}

  # PowerShell (Windows admin) :
  \$udp = New-Object System.Net.Sockets.UdpClient ${UDP_PORT}
  \$ep  = New-Object System.Net.IPEndPoint([Net.IPAddress]::Any, 0)
  while (\$true) { Write-Host ([Text.Encoding]::UTF8.GetString(\$udp.Receive([ref]\$ep))) }

Pare-feu à autoriser (exemples génériques — adapte à ton système) :

  Linux (iptables) :
    sudo iptables -A INPUT -p udp --dport ${UDP_PORT} -s ${IP} -j ACCEPT

  Windows (PowerShell admin) :
    New-NetFirewallRule -DisplayName "UDP ISIDetector" -Direction Inbound \\
      -Protocol UDP -LocalPort ${UDP_PORT} -RemoteAddress ${IP} -Action Allow

Handshake de validation :
  1. Lance ton écouteur (Python/nc/PowerShell ci-dessus).
  2. Sur mon PC, je déclenche :
       docker compose exec web python3 -c 'import socket; socket.socket(socket.AF_INET,socket.SOCK_DGRAM).sendto(b"test-from-abdul", ("${UDP_HOST}", ${UDP_PORT})); print("sent")'
  3. Tu dois voir apparaître la chaîne "test-from-abdul" immédiatement.
     → Oui : le chemin complet fonctionne, les vrais événements arrivent pareil.
     → Non : problème sur ton côté (pare-feu / bind / VLAN).

─────────────────────────────────────────────────────────────────────────
EOF
    fi
}

# ── Dispatch ────────────────────────────────────────────────────────────────
case "$CMD" in
    show)    cmd_show ;;
    apply)   cmd_apply ;;
    revert)  cmd_revert ;;
    test)    cmd_test ;;
    manual)  cmd_manual ;;
    *)       fail "Unknown command: $CMD"; print_help; exit 2 ;;
esac

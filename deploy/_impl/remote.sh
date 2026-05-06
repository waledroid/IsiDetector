#!/usr/bin/env bash
# ============================================================================
# IsiDetector — Site PC Remote Access Setup
#
# Installs and configures the two pieces that together give an admin
# unattended remote access to a site PC: Tailscale (zero-config VPN, free
# SSO sign-in via Gmail) and RustDesk (remote desktop GUI for the kiosk
# Chrome dashboard).
#
# Why both:
#   - Tailscale = the network. Site PC and admin laptop end up on the
#     same private 100.x.x.x mesh. Works through NAT, no port forwards,
#     no public IP required on either side.
#   - RustDesk  = the GUI. Once the network is up, RustDesk gives you a
#     full desktop view of the kiosk screen — see what the operator sees.
#
# Usage:
#   ./remote.sh                            # same as 'status'
#   ./remote.sh setup                      # install + start both services (sudo)
#   ./remote.sh setup --ts-key tskey-...   # tailscale unattended via auth key
#   ./remote.sh setup --rd-password PW     # set permanent RustDesk password
#   ./remote.sh status                     # show Tailscale IP + RustDesk ID
#   ./remote.sh test                       # connectivity probes (offline-tolerant)
#   ./remote.sh remove                     # uninstall both, leave system clean
#   ./remote.sh -h | --help
#
# Defaults (no flags): Tailscale uses interactive SSO (script prints the
# URL to visit; operator clicks once on the kiosk's Chrome and signs in
# with Gmail). RustDesk runs as a systemd service with a permanent
# password printed at the end of setup.
#
# Idempotent: re-running setup detects existing installs and skips them.
# Safe to run on every site PC update.
# ============================================================================

set -u

# ── Colour helpers (mirrors net.sh / run_start.sh so all scripts feel uniform)
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

# State file the admin can read on the next visit. Persists across reboots.
STATE_DIR="/var/log/isidetector"
STATE_FILE="${STATE_DIR}/remote-state.json"

# ── Argument parsing ────────────────────────────────────────────────────────
CMD=""
ARG_TS_KEY=""
ARG_RD_PASSWORD=""
ARG_RD_SERVER=""
ORIG_ARGS=("$@")

while [ $# -gt 0 ]; do
    case "$1" in
        setup|status|test|remove|help) CMD="$1" ;;
        --ts-key)        ARG_TS_KEY="${2:-}"; shift ;;
        --ts-key=*)      ARG_TS_KEY="${1#*=}" ;;
        --rd-password)   ARG_RD_PASSWORD="${2:-}"; shift ;;
        --rd-password=*) ARG_RD_PASSWORD="${1#*=}" ;;
        --rd-server)     ARG_RD_SERVER="${2:-}"; shift ;;
        --rd-server=*)   ARG_RD_SERVER="${1#*=}" ;;
        -h|--help)       CMD="help" ;;
        *) fail "unknown argument: $1"; CMD="help" ;;
    esac
    shift
done

CMD="${CMD:-status}"

# ── Sudo escalation ─────────────────────────────────────────────────────────
# setup / remove need root for apt/systemctl. status / test are read-only.
case "$CMD" in
    setup|remove)
        if [ "${EUID:-$(id -u)}" -ne 0 ]; then
            info "elevating with sudo..."
            exec sudo -E "$0" "${ORIG_ARGS[@]}"
        fi
        ;;
esac

# ── Helpers ─────────────────────────────────────────────────────────────────
require_cmd() {
    command -v "$1" >/dev/null 2>&1
}

# The "owner" user — the regular operator account that runs the kiosk.
# When the script runs under sudo we get this from $SUDO_USER; otherwise
# the current $USER. Used so the RustDesk GUI launches as the desktop
# user rather than as root (root has no DISPLAY in most cases).
desktop_user() {
    local u="${SUDO_USER:-${USER:-}}"
    if [ -z "$u" ] || [ "$u" = "root" ]; then
        # Last resort: pick the first /home/* with a real UID.
        u=$(awk -F: '$3 >= 1000 && $3 < 65000 {print $1; exit}' /etc/passwd)
    fi
    echo "$u"
}

is_apt_distro() {
    [ -r /etc/os-release ] && grep -qE '^ID(_LIKE)?=.*(debian|ubuntu)' /etc/os-release
}

internet_ok() {
    # 4-second budget; either tailscale's CDN or 1.1.1.1 must respond.
    curl -sS -m 4 -o /dev/null -w "%{http_code}" https://pkgs.tailscale.com/ 2>/dev/null | grep -qE '^(2|3)..' \
      || ping -c 1 -W 2 1.1.1.1 >/dev/null 2>&1
}

write_state() {
    # JSON state file: Tailscale IP, RustDesk ID, install timestamp, services.
    # chmod 0640 so the password isn't world-readable (we only stored the
    # SHA-256, but defensive anyway).
    mkdir -p "$STATE_DIR"
    chmod 0750 "$STATE_DIR"

    local ts_ip="${1:-}"
    local rd_id="${2:-}"
    local rd_pw_hash="${3:-}"
    local ts_status="${4:-}"
    local rd_status="${5:-}"

    cat > "$STATE_FILE" <<EOF
{
  "install_date": "$(date -Iseconds)",
  "tailscale": {
    "ip": "${ts_ip}",
    "service_status": "${ts_status}"
  },
  "rustdesk": {
    "id": "${rd_id}",
    "password_sha256": "${rd_pw_hash}",
    "service_status": "${rd_status}"
  }
}
EOF
    chmod 0640 "$STATE_FILE" || true
    success "state written → ${STATE_FILE}"
}

# ── Tailscale ───────────────────────────────────────────────────────────────
ts_install() {
    if require_cmd tailscale; then
        success "tailscale already installed ($(tailscale version | head -1))"
        return 0
    fi

    info "installing Tailscale via official script..."
    if ! curl -fsSL https://tailscale.com/install.sh | sh; then
        fail "Tailscale install failed"
        return 1
    fi
    success "Tailscale installed ($(tailscale version | head -1))"
}

ts_up() {
    # Already authenticated? Skip.
    if tailscale status >/dev/null 2>&1; then
        local cur_ip
        cur_ip=$(tailscale ip -4 2>/dev/null | head -1)
        if [ -n "$cur_ip" ]; then
            success "tailscale already up — IP ${cur_ip}"
            return 0
        fi
    fi

    if [ -n "$ARG_TS_KEY" ]; then
        info "bringing tailscale up via auth key..."
        if ! tailscale up --auth-key="$ARG_TS_KEY" --ssh --accept-routes; then
            fail "tailscale up (auth-key path) failed"
            return 1
        fi
    else
        info "bringing tailscale up via interactive SSO..."
        warn "the script will print a URL — open it in the kiosk's Chrome and"
        warn "sign in with the Gmail/email account that owns this Tailscale"
        warn "tailnet. The browser must run on this machine, not your laptop."
        # --ssh enables Tailscale SSH server (admin can `tailscale ssh user@site-pc`)
        # --accept-routes lets the site PC reach other tailnet subnets if any
        if ! tailscale up --ssh --accept-routes; then
            fail "tailscale up (interactive path) failed"
            return 1
        fi
    fi

    sleep 1
    local ip
    ip=$(tailscale ip -4 2>/dev/null | head -1)
    if [ -z "$ip" ]; then
        fail "tailscale up succeeded but no IPv4 was assigned — check the dashboard"
        return 1
    fi
    success "tailscale connected — site PC IP on the tailnet: ${ip}"
    echo "$ip"
}

ts_status() {
    if ! require_cmd tailscale; then
        echo "not-installed"
        return
    fi
    if tailscale status >/dev/null 2>&1; then
        echo "connected"
    else
        echo "logged-out"
    fi
}

# ── RustDesk ────────────────────────────────────────────────────────────────
RD_RELEASE_API="https://api.github.com/repos/rustdesk/rustdesk/releases/latest"

rd_latest_deb_url() {
    # Pick the Linux x86_64 .deb from the latest GitHub release.
    # Falls back to a known-good version if the API isn't reachable.
    local arch
    arch=$(dpkg --print-architecture 2>/dev/null || echo amd64)
    case "$arch" in
        amd64|x86_64) arch=x86_64 ;;
        arm64|aarch64) arch=aarch64 ;;
        *) fail "unsupported architecture: $arch"; return 1 ;;
    esac

    # `_$arch.deb` not `-$arch.deb` — RustDesk artefact name uses underscore.
    local url
    url=$(curl -sS -m 8 "$RD_RELEASE_API" 2>/dev/null \
            | grep -oE '"browser_download_url":[^"]*"[^"]*"' \
            | grep -E "_${arch}\.deb\"$" \
            | head -1 \
            | sed -E 's/.*"(https:[^"]*)"/\1/')
    if [ -z "$url" ]; then
        warn "could not query GitHub API — using known-good fallback"
        url="https://github.com/rustdesk/rustdesk/releases/download/1.3.8/rustdesk-1.3.8-${arch}.deb"
    fi
    echo "$url"
}

rd_install() {
    if require_cmd rustdesk; then
        success "rustdesk already installed ($(rustdesk --version 2>/dev/null | head -1 || echo unknown))"
        return 0
    fi

    info "fetching RustDesk runtime dependencies..."
    apt-get update -y
    apt-get install -y --no-install-recommends \
        libxdo3 libxcb-shape0 libxcb-xfixes0 libgtk-3-0 libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 libpulse0 libxkbcommon-x11-0 \
        libayatana-appindicator3-1 libdbus-1-3 || warn "some deps optional — continuing"

    local url
    url=$(rd_latest_deb_url)
    info "downloading RustDesk: $url"
    local deb="/tmp/rustdesk.deb"
    if ! curl -fsSL --retry 3 -o "$deb" "$url"; then
        fail "RustDesk .deb download failed"
        return 1
    fi
    info "installing RustDesk..."
    apt-get install -y "$deb"
    rm -f "$deb"
    success "RustDesk installed ($(rustdesk --version 2>/dev/null | head -1 || echo ok))"
}

rd_configure() {
    # Enable + start the systemd service so RustDesk runs at boot, before
    # any user logs in. Lets the admin connect even if the kiosk hasn't
    # auto-logged-in yet.
    if systemctl list-unit-files | grep -q '^rustdesk\.service'; then
        systemctl enable --now rustdesk.service 2>/dev/null \
            && success "rustdesk.service enabled + started" \
            || warn "could not enable rustdesk.service — RustDesk will only run when GUI launches it"
    else
        warn "rustdesk.service unit not found — package may not ship one on this distro"
    fi

    # Optional: point at a self-hosted relay server.
    if [ -n "$ARG_RD_SERVER" ]; then
        info "configuring RustDesk to use custom server: $ARG_RD_SERVER"
        # RustDesk reads server config from ~/.config/rustdesk/RustDesk2.toml.
        # This is per-user. We write it for the desktop user.
        local du; du=$(desktop_user)
        local cfg_dir="/home/${du}/.config/rustdesk"
        mkdir -p "$cfg_dir"
        cat > "${cfg_dir}/RustDesk2.toml" <<EOF
rendezvous_server = '${ARG_RD_SERVER}'
nat_type = 1
serial = 0

[options]
custom-rendezvous-server = '${ARG_RD_SERVER}'
EOF
        chown -R "$du:$du" "$cfg_dir"
        success "rustdesk server config written for ${du}"
    fi

    # Permanent password — required for unattended access. Either provided
    # via --rd-password or randomly generated.
    local pw="$ARG_RD_PASSWORD"
    if [ -z "$pw" ]; then
        pw=$(head -c 16 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 12)
        info "generated random permanent password (12 chars)"
    fi

    # Set it via the rustdesk CLI. This must run as the desktop user
    # because RustDesk stores the password in the user's config dir.
    local du; du=$(desktop_user)
    if su - "$du" -c "command -v rustdesk" >/dev/null 2>&1; then
        # The --password CLI option exists in 1.2+. Run via the desktop
        # user's environment so DISPLAY and config dir resolve correctly.
        if su - "$du" -c "rustdesk --password '$pw'" 2>/dev/null; then
            success "rustdesk permanent password set"
        else
            warn "could not set RustDesk password automatically — run 'rustdesk --password YOUR_PW' as ${du} after first GUI launch"
        fi
    else
        warn "rustdesk CLI not in user's PATH — set password manually after first GUI launch"
    fi

    # Echo the password for the admin (printed once, in plaintext, in the summary).
    # Also stash a SHA-256 in the state file so the admin can verify which password was set.
    REMOTE_RD_PW="$pw"
}

rd_get_id() {
    # The 9-digit RustDesk ID is generated at first start and stored in the
    # user's config. Read it after the service has had a moment to come up.
    local du; du=$(desktop_user)
    local cfg="/home/${du}/.config/rustdesk/RustDesk2.toml"
    if [ -r "$cfg" ]; then
        grep -E "^id\s*=\s*'" "$cfg" | head -1 | sed -E "s/^id\s*=\s*'([^']+)'.*/\1/"
        return
    fi
    # Fallback: ask the binary directly.
    su - "$du" -c "rustdesk --get-id" 2>/dev/null | head -1
}

rd_status() {
    if ! require_cmd rustdesk; then
        echo "not-installed"
        return
    fi
    if systemctl is-active --quiet rustdesk.service; then
        echo "running"
    else
        echo "installed-not-running"
    fi
}

# ── Subcommands ─────────────────────────────────────────────────────────────
cmd_setup() {
    header "IsiDetector remote-access setup"

    # 1. Pre-flight
    info "pre-flight checks..."
    if ! is_apt_distro; then
        fail "this script targets Debian/Ubuntu (apt). Detected:"
        cat /etc/os-release 2>/dev/null | head -3
        exit 2
    fi
    if ! internet_ok; then
        fail "no internet — Tailscale + RustDesk install need outbound HTTPS."
        warn "connect a network cable / hotspot, then re-run: ./remote.sh setup"
        exit 3
    fi
    success "internet reachable, apt-based distro detected"

    # 2. Tailscale
    header "Tailscale"
    ts_install || exit 4
    local ts_ip
    ts_ip=$(ts_up | tail -1)

    # 3. RustDesk
    header "RustDesk"
    rd_install || exit 5
    rd_configure
    sleep 2  # let the service initialise so RustDesk2.toml exists with an ID
    local rd_id
    rd_id=$(rd_get_id)
    if [ -z "$rd_id" ]; then
        warn "could not read RustDesk ID yet — will appear after first GUI launch"
        rd_id="(pending — launch RustDesk GUI once)"
    fi

    # 4. State recording
    local pw_hash="(unknown)"
    if [ -n "${REMOTE_RD_PW:-}" ]; then
        pw_hash=$(printf '%s' "$REMOTE_RD_PW" | sha256sum | awk '{print $1}')
    fi
    write_state "$ts_ip" "$rd_id" "$pw_hash" "$(ts_status)" "$(rd_status)"

    # 5. Summary
    header "Setup complete"
    echo "Tailscale IP (private mesh):  ${BOLD}${ts_ip}${NC}"
    echo "RustDesk ID:                  ${BOLD}${rd_id}${NC}"
    echo "RustDesk password:            ${BOLD}${REMOTE_RD_PW:-(unchanged)}${NC}"
    echo ""
    echo "From your laptop, after joining the same Tailscale tailnet:"
    echo "  • SSH:        tailscale ssh ${USER}@${ts_ip}"
    echo "  • Desktop:    open RustDesk → enter ID ${rd_id} + password above"
    echo ""
    echo "State for next visit: ${STATE_FILE}"
    echo ""
}

cmd_status() {
    header "Remote-access status"

    local ts_state rd_state ts_ip rd_id
    ts_state=$(ts_status)
    rd_state=$(rd_status)

    echo "Tailscale: ${ts_state}"
    if [ "$ts_state" = "connected" ]; then
        ts_ip=$(tailscale ip -4 2>/dev/null | head -1)
        echo "  IP:      ${ts_ip}"
        echo "  Peers:"
        tailscale status 2>/dev/null | head -10 | sed 's/^/    /'
    fi

    echo ""
    echo "RustDesk:  ${rd_state}"
    if [ "$rd_state" = "running" ] || [ "$rd_state" = "installed-not-running" ]; then
        rd_id=$(rd_get_id)
        echo "  ID:      ${rd_id:-(unknown — try GUI launch)}"
    fi

    if [ -r "$STATE_FILE" ]; then
        echo ""
        echo "Last setup state: ${STATE_FILE}"
        cat "$STATE_FILE" | sed 's/^/  /'
    fi
}

cmd_test() {
    header "Remote-access connectivity test"

    if internet_ok; then success "internet reachable"
    else                  warn   "no internet — Tailscale/RustDesk install would fail"
    fi

    if require_cmd tailscale; then
        if tailscale status >/dev/null 2>&1; then
            success "tailscale: connected — IP $(tailscale ip -4 2>/dev/null | head -1)"
            # Try to reach a tailnet peer (any) on TCP 22 / 80 — non-fatal probe.
            local peer
            peer=$(tailscale status 2>/dev/null | awk 'NR>1 && $2!="" {print $1; exit}')
            if [ -n "$peer" ]; then
                if timeout 3 bash -c "</dev/tcp/${peer}/22" 2>/dev/null; then
                    success "tailnet peer ${peer} reachable on TCP 22"
                else
                    warn "tailnet peer ${peer} not reachable on TCP 22 (non-fatal — peer may not run sshd)"
                fi
            else
                info "no tailnet peers visible yet"
            fi
        else
            warn "tailscale installed but not logged in — run 'sudo ./remote.sh setup'"
        fi
    else
        info "tailscale not installed"
    fi

    if require_cmd rustdesk; then
        case "$(rd_status)" in
            running)               success "rustdesk service running" ;;
            installed-not-running) warn   "rustdesk installed but service not active" ;;
            not-installed)         info   "rustdesk not installed" ;;
        esac
    else
        info "rustdesk not installed"
    fi
}

cmd_remove() {
    header "Removing Tailscale + RustDesk"
    warn "this will uninstall both packages and revoke the Tailscale device."
    read -p "Type 'remove' to confirm: " confirm
    if [ "$confirm" != "remove" ]; then
        info "cancelled."
        exit 0
    fi

    if require_cmd tailscale; then
        info "logging out of Tailscale + uninstalling..."
        tailscale logout 2>/dev/null || true
        apt-get remove -y tailscale 2>/dev/null || true
        rm -f /etc/apt/sources.list.d/tailscale.list /etc/apt/keyrings/tailscale*.gpg 2>/dev/null
        success "tailscale removed"
    fi

    if require_cmd rustdesk; then
        info "stopping + uninstalling RustDesk..."
        systemctl disable --now rustdesk.service 2>/dev/null || true
        apt-get remove -y rustdesk 2>/dev/null || true
        success "rustdesk removed"
    fi

    rm -f "$STATE_FILE" 2>/dev/null
    success "state file cleared"
}

cmd_help() {
    sed -nE '/^# Usage:/,/^# =+$/p' "$0" | sed 's/^# \?//'
}

# ── Dispatch ────────────────────────────────────────────────────────────────
case "$CMD" in
    setup)   cmd_setup ;;
    status)  cmd_status ;;
    test)    cmd_test ;;
    remove)  cmd_remove ;;
    help|*)  cmd_help ;;
esac

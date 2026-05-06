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

# Fleet-wide default RustDesk permanent password. One password to remember
# across every Isitec site PC — the admin team types it from muscle memory
# rather than fishing the per-host random one out of the state file. Tailscale
# is the access perimeter; the RustDesk password is the second factor at the
# session level.
#
# Override per-host with:  sudo ./remote.sh setup --rd-password 'OTHER_PW'
#
# CHANGING THIS VALUE: the new default only applies to fresh installs. To
# rotate on already-deployed site PCs, run there:
#   su - <desktop_user> -c "rustdesk --password 'NEW_PW'"
#   sudo systemctl restart rustdesk
RD_DEFAULT_PASSWORD="Isitec69+"

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
    # JSON state file: Tailscale IP, RustDesk ID + plaintext password,
    # direct-IP connection details, install timestamp, service states.
    # The plaintext password lets the admin read it directly on the next
    # visit without re-running setup; mode 0640 + a root-owned dir keeps
    # it off world view. Site PCs are single-admin boxes — full plaintext
    # recovery is the right tradeoff.
    mkdir -p "$STATE_DIR"
    chmod 0750 "$STATE_DIR"

    local ts_ip="${1:-}"
    local rd_id="${2:-}"
    local rd_pw="${3:-}"
    local ts_status="${4:-}"
    local rd_status="${5:-}"
    local lan_ip="${6:-}"

    # Build the direct-IP connect strings. Empty if either component
    # is missing so the JSON stays valid.
    local direct_ts="" direct_lan=""
    [ -n "$ts_ip" ]  && direct_ts="${ts_ip}:${RD_DIRECT_PORT:-21118}"
    [ -n "$lan_ip" ] && direct_lan="${lan_ip}:${RD_DIRECT_PORT:-21118}"

    cat > "$STATE_FILE" <<EOF
{
  "install_date": "$(date -Iseconds)",
  "tailscale": {
    "ip": "${ts_ip}",
    "service_status": "${ts_status}"
  },
  "rustdesk": {
    "id": "${rd_id}",
    "password": "${rd_pw}",
    "service_status": "${rd_status}",
    "direct_ip_via_tailscale": "${direct_ts}",
    "direct_ip_via_lan": "${direct_lan}",
    "verification_method": "use-permanent-password"
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

# TS_IP is the global the rest of the script reads after ts_up succeeds.
# Using a global avoids the stdout-capture-vs-log-output mess of
# `ts_ip=$(ts_up | tail -1)` — info/warn lines from inside ts_up
# can't pollute the captured value.
TS_IP=""

ts_up() {
    # Already authenticated? Skip the whole interactive flow.
    local cur_ip
    cur_ip=$(tailscale ip -4 2>/dev/null | head -1)
    if [ -n "$cur_ip" ]; then
        success "tailscale already up — IP ${cur_ip}"
        TS_IP="$cur_ip"
        return 0
    fi

    if [ -n "$ARG_TS_KEY" ]; then
        # Auth-key path: synchronous, no human in the loop. Doesn't hang.
        info "bringing tailscale up via auth key (unattended)..."
        if ! tailscale up --auth-key="$ARG_TS_KEY"; then
            fail "tailscale up (auth-key path) failed"
            return 1
        fi
    else
        # Interactive SSO path with explicit operator confirmation.
        # Earlier we tried polling `tailscale ip -4` to detect auth
        # completion automatically. That fails too often:
        #   - tailnet device-approval is async (admin clicks Approve at
        #     login.tailscale.com/admin/machines — no fixed timeout)
        #   - the daemon sometimes lags 10–30 s after browser auth before
        #     assigning an IP
        #   - ACL / subnet-route policy can defer the IP grant indefinitely
        # The robust UX is to let the operator tell us when they're done.
        info "bringing tailscale up via interactive SSO..."
        echo ""
        warn "ACTION REQUIRED — please follow these steps:"
        warn "  1. The script will print a URL below in a few seconds."
        warn "  2. Click it on the kiosk's Chrome (or copy + paste it)."
        warn "  3. Sign in with the Gmail/email that owns this tailnet."
        warn "  4. If your tailnet requires device approval, click 'Approve'"
        warn "     at https://login.tailscale.com/admin/machines"
        warn "  5. Come back to this terminal and press Enter."
        echo ""

        # CRITICAL: don't pass --ssh or --accept-routes to `tailscale up`.
        # Both can cause the CLI to hang indefinitely waiting for ACL or
        # subnet-route approval. We apply them separately via `tailscale
        # set` after auth completes, with error tolerance.
        #
        # `tailscale up --reset` clears any prior partial state from a
        # previous failed run, so re-running setup after a hang doesn't
        # confuse the daemon.
        tailscale up --reset &
        local up_pid=$!

        # Give the URL a couple of seconds to print to the terminal before
        # we obscure it with our own prompt.
        sleep 3
        echo ""

        # Retry loop: if first attempt didn't yield an IP, give the operator
        # another chance (e.g., they noticed approval is pending and went
        # to fix it). Cap at 3 tries so a misconfigured tailnet doesn't
        # trap the script forever.
        local attempt=0
        local poll_ip=""
        while [ $attempt -lt 3 ]; do
            attempt=$((attempt + 1))
            local prompt_msg
            if [ $attempt -eq 1 ]; then
                prompt_msg="Press Enter once you have completed sign-in (or Ctrl+C to abort): "
            else
                prompt_msg="Press Enter to re-check tailscale status (or Ctrl+C to abort): "
            fi
            # Read from /dev/tty so the prompt works even if the script's
            # stdin is being piped in (e.g., curl | bash invocations).
            read -r -p "$prompt_msg" _ < /dev/tty || true
            echo ""

            # Kill the backgrounded `tailscale up` if still running — its
            # job (printing the URL + handling the auth callback) is done.
            if kill -0 "$up_pid" 2>/dev/null; then
                kill "$up_pid" 2>/dev/null
                wait "$up_pid" 2>/dev/null
            fi

            sleep 1
            poll_ip=$(tailscale ip -4 2>/dev/null | head -1)
            if [ -n "$poll_ip" ]; then
                break
            fi

            # No IP yet — show diagnostic + offer retry.
            warn "no Tailscale IPv4 assigned yet (attempt ${attempt}/3)."
            warn "Current 'tailscale status':"
            tailscale status 2>&1 | head -6 | sed 's/^/    /' || true
            echo ""

            if [ $attempt -lt 3 ]; then
                warn "Common reasons:"
                warn "  • Browser auth not completed → finish sign-in, then press Enter."
                warn "  • Device awaiting admin approval → approve at"
                warn "    https://login.tailscale.com/admin/machines, then press Enter."
                warn "  • Wrong Gmail account → 'tailscale logout' first, then re-run setup."
                echo ""
                # Re-launch tailscale up so a fresh URL appears if the
                # daemon let go of the previous auth handle.
                tailscale up --reset &
                up_pid=$!
                sleep 2
            fi
        done

        if [ -z "$poll_ip" ]; then
            fail "tailscale never received an IPv4 after 3 attempts."
            warn "The Tailscale coordination flow appears blocked."
            warn "Final 'tailscale status':"
            tailscale status 2>&1 | head -10 | sed 's/^/    /' || true
            warn "Recover later with: sudo tailscale up   (after fixing the cause)."
            return 1
        fi
    fi

    # Auth confirmed. Apply optional extras with error tolerance — none of
    # these are essential to remote access, so failures degrade gracefully.
    if tailscale set --ssh=true 2>/dev/null; then
        success "Tailscale SSH server enabled (run 'tailscale ssh ${SUDO_USER:-$USER}@<this-host>' from your laptop)"
    else
        warn "Tailscale SSH not enabled — your tailnet ACL must allow it on this user."
        warn "Not fatal; RustDesk handles remote desktop access independently."
    fi

    if tailscale set --accept-routes=true 2>/dev/null; then
        info "subnet route acceptance enabled"
    fi

    local ip
    ip=$(tailscale ip -4 2>/dev/null | head -1)
    if [ -z "$ip" ]; then
        fail "tailscale daemon ran but no IPv4 was assigned"
        return 1
    fi
    success "tailscale connected — site PC IP on the tailnet: ${ip}"
    TS_IP="$ip"
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

# ── Display server: force X11 (RustDesk requirement) ───────────────────────
# RustDesk's screen-capture pipeline works on X11 but is broken on Wayland —
# the Wayland security model blocks the keystroke / framebuffer access RustDesk
# needs for unattended remote control. Ubuntu 22.04+ ships GDM3 with Wayland
# as the default session, so on a fresh install the operator would see "no
# screen" or "input rejected" errors after RustDesk connects.
#
# Fix: write `WaylandEnable=false` into /etc/gdm3/custom.conf. Takes effect
# on next reboot — we DO NOT restart GDM here, that would kick the operator
# out of the desktop session mid-setup. The script flags the reboot need
# in the final summary instead.
#
# LightDM / SDDM aren't Wayland by default, so we skip the rewrite there
# (still warn if the *current* session happens to be Wayland for any reason).

current_session_type() {
    # `loginctl show-session ... -p Type` is most reliable, but needs a session
    # ID. $XDG_SESSION_TYPE is set in interactive shells but not always under sudo.
    local stype="${XDG_SESSION_TYPE:-}"
    if [ -z "$stype" ] && command -v loginctl >/dev/null 2>&1; then
        local sid
        sid=$(loginctl show-user "$(desktop_user)" -p Display 2>/dev/null | cut -d= -f2)
        if [ -n "$sid" ]; then
            stype=$(loginctl show-session "$sid" -p Type --value 2>/dev/null)
        fi
    fi
    if [ -z "$stype" ]; then
        # Last resort: presence of the Xwayland or Xorg process.
        if pgrep -x Xwayland >/dev/null 2>&1; then stype=wayland
        elif pgrep -x Xorg     >/dev/null 2>&1; then stype=x11
        else                                        stype=unknown
        fi
    fi
    echo "$stype"
}

active_dm() {
    # Echo gdm3 / lightdm / sddm / unknown.
    local dm=""
    for d in gdm3 gdm lightdm sddm; do
        if systemctl is-active --quiet "${d}.service" 2>/dev/null; then
            dm="$d"; break
        fi
    done
    if [ -z "$dm" ] && [ -r /etc/X11/default-display-manager ]; then
        dm=$(basename "$(cat /etc/X11/default-display-manager 2>/dev/null)" 2>/dev/null)
    fi
    echo "${dm:-unknown}"
}

ensure_x11_session() {
    local dm sess
    dm=$(active_dm)
    sess=$(current_session_type)

    info "display manager: ${dm} | current session: ${sess}"

    # Already X11 in config and runtime? Nothing to do.
    case "$dm" in
        gdm3|gdm)
            local conf="/etc/gdm3/custom.conf"
            if [ ! -f "$conf" ]; then
                # GDM running but no custom.conf — write a fresh one.
                info "creating ${conf}"
                cat > "$conf" <<'EOF'
[daemon]
WaylandEnable=false
EOF
                REMOTE_NEED_REBOOT=1
                success "Wayland disabled (fresh ${conf} written)"
                return
            fi

            # Existing custom.conf — three states:
            #   1. Has commented-out `#WaylandEnable=false` (Ubuntu default)
            #   2. Has uncommented `WaylandEnable=false` already (we ran before)
            #   3. Has `WaylandEnable=true` or no entry at all
            if grep -qE '^\s*WaylandEnable\s*=\s*false' "$conf"; then
                success "Wayland already disabled in ${conf}"
                # If the current session is still Wayland, the config change
                # hasn't taken effect yet — needs a reboot.
                if [ "$sess" = "wayland" ]; then
                    warn "current session is still Wayland — reboot to apply"
                    REMOTE_NEED_REBOOT=1
                fi
                return
            fi

            # Backup once so the operator can restore if anything goes sideways.
            local backup="${conf}.pre-remote-$(date +%Y%m%d-%H%M%S)"
            cp -a "$conf" "$backup" 2>/dev/null && info "backup: ${backup}"

            # If the line exists commented-out, uncomment it. Otherwise add it
            # under the [daemon] section (creating that section if missing).
            if grep -qE '^\s*#\s*WaylandEnable\s*=' "$conf"; then
                sed -i -E 's/^\s*#\s*WaylandEnable\s*=.*/WaylandEnable=false/' "$conf"
                success "uncommented WaylandEnable=false in ${conf}"
            elif grep -qE '^\s*\[daemon\]' "$conf"; then
                sed -i -E '/^\s*\[daemon\]/a WaylandEnable=false' "$conf"
                success "added WaylandEnable=false under [daemon] in ${conf}"
            else
                printf '\n[daemon]\nWaylandEnable=false\n' >> "$conf"
                success "appended [daemon] WaylandEnable=false to ${conf}"
            fi
            REMOTE_NEED_REBOOT=1
            warn "REBOOT REQUIRED for the X11 switch to take effect"
            ;;

        lightdm|sddm)
            # These DMs default to X11; only warn if Wayland is somehow active.
            if [ "$sess" = "wayland" ]; then
                warn "${dm} is the active DM but the current session is Wayland."
                warn "This is unusual — check the user's session selection at login."
                warn "RustDesk capture/input may not work until logged into an X11 session."
            else
                success "${dm} runs X11 by default — no display-manager change needed"
            fi
            ;;

        *)
            warn "could not detect a known display manager — skipping Wayland-disable step."
            warn "If RustDesk shows a black screen or rejects input after connect, manually"
            warn "switch the desktop session to X11 (logout → gear icon → 'Ubuntu on Xorg')."
            ;;
    esac
}

# ── RustDesk ────────────────────────────────────────────────────────────────
RD_RELEASE_API="https://api.github.com/repos/rustdesk/rustdesk/releases/latest"

rd_latest_deb_url() {
    # Pick the Linux x86_64 / aarch64 .deb URL via a multi-strategy lookup,
    # HEAD-checking each candidate so we never commit to a download that
    # won't resolve on the site PC's network.
    local arch
    arch=$(dpkg --print-architecture 2>/dev/null || echo amd64)
    case "$arch" in
        amd64|x86_64)   arch=x86_64 ;;
        arm64|aarch64)  arch=aarch64 ;;
        *) fail "unsupported architecture: $arch"; return 1 ;;
    esac

    # Strategy 1: query GitHub API for latest release. Tolerate either
    # hyphen or underscore separator (RustDesk has used both across
    # versions); only require that the asset filename ends with the
    # arch + `.deb`.
    local url
    url=$(curl -sS -m 8 "$RD_RELEASE_API" 2>/dev/null \
            | grep -oE '"browser_download_url":[^"]*"[^"]*"' \
            | grep -E "[-_]${arch}\.deb\"$" \
            | head -1 \
            | sed -E 's/.*"(https:[^"]*)"/\1/')
    if [ -n "$url" ] && curl -fsSL -I -o /dev/null -m 6 "$url"; then
        echo "$url"
        return 0
    fi

    # Strategy 2: walk a list of known-good versions, newest first. HEAD-
    # check each so we don't pick a URL the site PC's network can't fetch
    # (e.g., a corporate proxy that allows api.github.com but blocks
    # objects.githubusercontent.com — we'd rather find one that works).
    warn "GitHub API didn't yield a usable URL — trying known-good versions"
    for v in 1.4.6 1.4.2 1.4.1 1.4.0 1.3.9 1.3.8; do
        local fb="https://github.com/rustdesk/rustdesk/releases/download/${v}/rustdesk-${v}-${arch}.deb"
        if curl -fsSL -I -o /dev/null -m 6 "$fb"; then
            info "using fallback: rustdesk ${v}"
            echo "$fb"
            return 0
        fi
    done

    fail "no RustDesk .deb URL is reachable — site PC may block github.com asset CDN"
    return 1
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
    url=$(rd_latest_deb_url) || return 1
    info "downloading RustDesk: $url"
    local deb="/tmp/rustdesk.deb"
    local http_code
    http_code=$(curl -fsSL --retry 3 -m 60 -o "$deb" -w "%{http_code}" "$url" 2>&1) || {
        fail "RustDesk .deb download failed (curl exit / http: ${http_code})"
        warn "URL was: $url"
        warn "Try manually: curl -fsSL -o /tmp/rustdesk.deb '$url'"
        return 1
    }
    info "installing RustDesk..."
    apt-get install -y "$deb"
    rm -f "$deb"
    success "RustDesk installed ($(rustdesk --version 2>/dev/null | head -1 || echo ok))"
}

#: Default port RustDesk listens on for direct IP connections (when
#: `direct-server = Y`). Standard across all RustDesk versions ≥1.2.
RD_DIRECT_PORT="21118"

# Determine which user the rustdesk systemd service runs as, and where its
# config dir lives. CRITICAL: on most distros the service runs as root and
# reads /root/.config/rustdesk/, NOT the desktop user's $HOME/.config/rustdesk/.
# Writing options to the wrong dir = settings never reach the running service.
rd_service_user_and_home() {
    local svc_user svc_home
    svc_user=$(systemctl show -p User --value rustdesk.service 2>/dev/null)
    [ -z "$svc_user" ] && svc_user=root
    svc_home=$(getent passwd "$svc_user" 2>/dev/null | cut -d: -f6)
    [ -z "$svc_home" ] && svc_home="/root"
    echo "${svc_user}|${svc_home}"
}

# Set a key=value pair under the [options] section of a RustDesk TOML config,
# creating/updating the section as needed. Idempotent. RustDesk uses single-
# quoted strings in its TOML, so we match that format.
rd_set_option() {
    local file="$1" key="$2" val="$3"
    mkdir -p "$(dirname "$file")"
    if [ ! -f "$file" ]; then
        cat > "$file" <<EOF
[options]
${key} = '${val}'
EOF
        return
    fi
    if grep -qE "^[[:space:]]*${key}[[:space:]]*=" "$file"; then
        # Replace existing line (any value, any whitespace).
        sed -i -E "s|^[[:space:]]*${key}[[:space:]]*=.*|${key} = '${val}'|" "$file"
    elif grep -qE "^[[:space:]]*\[options\]" "$file"; then
        # Insert under the existing [options] section header.
        sed -i -E "/^[[:space:]]*\[options\]/a ${key} = '${val}'" "$file"
    else
        # No [options] section yet — append one.
        printf '\n[options]\n%s = %s\n' "$key" "'${val}'" >> "$file"
    fi
}

rd_configure() {
    # 1. Enable + start the systemd service. RustDesk writes its initial
    # config (with the unique device ID) on first start. We need that ID
    # before we can layer our options on top.
    if systemctl list-unit-files 2>/dev/null | grep -q '^rustdesk\.service'; then
        systemctl enable --now rustdesk.service 2>/dev/null \
            && success "rustdesk.service enabled + started" \
            || warn "could not enable rustdesk.service"
    else
        warn "rustdesk.service unit not found — package may not ship one on this distro"
    fi

    # 2. Discover the service's identity. This is the bug from the prior
    # version: rustdesk-server runs as root by default, so its config lives
    # in /root/.config/rustdesk/, NOT in the desktop user's home. The CLI
    # commands `rustdesk --option / --password` invoked as the desktop
    # user wrote to /home/$user/.config/rustdesk/ — a directory the running
    # service never reads. Result: settings appeared to succeed (CLI exit
    # 0) but had zero effect on the actual service behaviour.
    local svc_info svc_user svc_home svc_cfg_dir
    svc_info=$(rd_service_user_and_home)
    svc_user="${svc_info%|*}"
    svc_home="${svc_info#*|}"
    svc_cfg_dir="${svc_home}/.config/rustdesk"
    info "rustdesk service runs as: ${svc_user} (config dir: ${svc_cfg_dir})"

    # 3. Wait for the service to write its initial config + ID. Without
    # this we'd race with config creation and our edits could be clobbered
    # when the daemon writes its first generated config.
    local main_cfg="${svc_cfg_dir}/RustDesk2.toml"
    local local_cfg="${svc_cfg_dir}/RustDesk_local.toml"
    local waited=0
    while [ ! -r "$main_cfg" ] && [ $waited -lt 15 ]; do
        sleep 1
        waited=$((waited + 1))
    done
    if [ ! -r "$main_cfg" ]; then
        warn "RustDesk hasn't written ${main_cfg} after 15 s — proceeding anyway"
    fi

    # 4. Stop the service so our edits land cleanly. The daemon caches the
    # config in memory and rewrites it on shutdown; editing while it runs
    # races with that flush.
    systemctl stop rustdesk.service 2>/dev/null
    sleep 2

    # 5. Resolve the password — fleet default unless --rd-password override.
    local pw="$ARG_RD_PASSWORD"
    if [ -z "$pw" ]; then
        pw="$RD_DEFAULT_PASSWORD"
        info "using fleet default permanent password (override with --rd-password)"
    fi

    # 6. Write the options DIRECTLY to RustDesk_local.toml in the service's
    # config dir. Bypasses the rustdesk CLI entirely — we control the
    # exact bytes. The local TOML is where RustDesk persists per-machine
    # options like verification-method, direct-server, etc.
    rd_set_option "$local_cfg" "verification-method" "use-permanent-password"
    rd_set_option "$local_cfg" "direct-server"        "Y"
    rd_set_option "$local_cfg" "direct-access-port"   "${RD_DIRECT_PORT}"
    if [ -n "$ARG_RD_SERVER" ]; then
        rd_set_option "$local_cfg" "custom-rendezvous-server" "$ARG_RD_SERVER"
    fi

    # Ensure the directory + files are owned by the service user so the
    # running service can read/write them.
    if [ "$svc_user" != "root" ]; then
        chown -R "${svc_user}:${svc_user}" "$svc_cfg_dir" 2>/dev/null || true
    else
        chown -R root:root "$svc_cfg_dir" 2>/dev/null || true
    fi

    # 7. Set the password via rustdesk CLI AS THE SERVICE USER. RustDesk
    # hashes the password before storing it (we can't write the hash
    # ourselves without replicating their algorithm), so the CLI is still
    # the right tool — we just have to invoke it with the same identity
    # the service uses.
    local pw_set=0
    if [ "$svc_user" = "root" ]; then
        if rustdesk --password "$pw" >/dev/null 2>&1; then
            pw_set=1
        fi
    else
        if su - "$svc_user" -c "rustdesk --password '$pw'" >/dev/null 2>&1; then
            pw_set=1
        fi
    fi
    if [ $pw_set -eq 1 ]; then
        success "rustdesk permanent password set"
    else
        warn "rustdesk --password CLI failed — falling back to direct TOML write"
        # Fallback: many RustDesk versions also accept a plaintext
        # 'password' field under [options]. Not all do; this is best-effort.
        rd_set_option "$local_cfg" "password" "$pw"
    fi

    # 8. Restart the service so it picks up the new config from disk.
    if systemctl start rustdesk.service 2>/dev/null; then
        sleep 2
        success "rustdesk.service restarted — new config is now active"
    else
        warn "could not restart rustdesk.service"
    fi

    # 9. Read back the local config and confirm our settings actually
    # landed. If they didn't, the operator gets specific instructions to
    # fix it manually rather than a silent success.
    if grep -qE "^[[:space:]]*verification-method[[:space:]]*=[[:space:]]*'use-permanent-password'" "$local_cfg" 2>/dev/null; then
        success "verification mode confirmed: use-permanent-password (in ${local_cfg})"
    else
        warn "verification-method NOT confirmed in ${local_cfg}"
        warn "Manual fix: open the RustDesk GUI → Settings → Security → set"
        warn "  'Use permanent password' and re-enter the password."
    fi
    if grep -qE "^[[:space:]]*direct-server[[:space:]]*=[[:space:]]*'Y'" "$local_cfg" 2>/dev/null; then
        success "direct IP access confirmed (TCP port ${RD_DIRECT_PORT}) in ${local_cfg}"
    else
        warn "direct-server NOT confirmed in ${local_cfg}"
        warn "Manual fix: open the RustDesk GUI → Settings → Network → tick"
        warn "  'Enable direct IP access'."
    fi

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

    # 2. Tailscale (non-fatal — RustDesk still useful on local LAN if this fails)
    header "Tailscale"
    local ts_ip=""
    if ts_install; then
        if ts_up; then
            ts_ip="$TS_IP"
        else
            warn "tailscale step failed — continuing with RustDesk install."
            warn "Remote-via-Tailscale won't work, but RustDesk on the local"
            warn "LAN will still let you connect when on-site."
        fi
    else
        warn "tailscale install failed — continuing with RustDesk install."
    fi

    # 3. Display server (RustDesk requires X11)
    header "Display server (X11 for RustDesk)"
    ensure_x11_session

    # 4. RustDesk
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

    # 5. Discover the LAN IP (first non-loopback, non-tailscale, non-docker IPv4).
    # Used for direct on-site connections that don't need Tailscale at all.
    local lan_ip
    lan_ip=$(ip -4 -o addr show 2>/dev/null \
        | awk '{print $2, $4}' \
        | awk -F'[ /]' '$1 !~ /^(lo|docker|br-|veth|tailscale)/ {print $2; exit}')

    # 6. State recording
    write_state "$ts_ip" "$rd_id" "${REMOTE_RD_PW:-}" "$(ts_status)" "$(rd_status)" "$lan_ip"

    # 7. Summary — plain text, no terminal escapes. Shows BOTH connection
    # methods (ID via public relay, and direct IP+port over Tailscale/LAN)
    # so the admin can pick the best path from their laptop.
    header "Setup complete"
    echo ""
    echo "  Tailscale IP (private mesh):  ${ts_ip:-(not connected — see warnings above)}"
    echo "  LAN IP (on-site only):        ${lan_ip:-(not detected)}"
    echo ""
    echo "  RustDesk ID:                  ${rd_id:-(pending — launch RustDesk GUI once)}"
    echo "  RustDesk password:            ${REMOTE_RD_PW:-(unchanged)}"
    echo "  Verification mode:            permanent password (script-set, not auto-rotating)"
    echo ""
    echo "  Connection options from your laptop's RustDesk client:"
    echo "    A) ID + password (works anywhere with internet):"
    echo "       Enter:   ${rd_id:-<id>}"
    echo "       Then password:  ${REMOTE_RD_PW:-<password>}"
    if [ -n "$ts_ip" ]; then
        echo "    B) Direct IP via Tailscale (preferred — fast, no public relay):"
        echo "       Enter:   ${ts_ip}:${RD_DIRECT_PORT}"
        echo "       Then password:  ${REMOTE_RD_PW:-<password>}"
    fi
    if [ -n "$lan_ip" ]; then
        echo "    C) Direct IP via LAN (when on-site, no Tailscale needed):"
        echo "       Enter:   ${lan_ip}:${RD_DIRECT_PORT}"
        echo "       Then password:  ${REMOTE_RD_PW:-<password>}"
    fi
    if [ -n "$ts_ip" ]; then
        echo "    D) SSH over Tailscale:  tailscale ssh ${SUDO_USER:-$USER}@${ts_ip}"
    fi
    echo ""
    echo "  These details are persisted in:  ${STATE_FILE}"
    echo "  (root-readable; on next visit run:  sudo cat ${STATE_FILE})"
    if [ "${REMOTE_NEED_REBOOT:-0}" = "1" ]; then
        echo ""
        warn "REBOOT REQUIRED: Wayland was disabled in /etc/gdm3/custom.conf."
        warn "Reboot the site PC now so RustDesk lands in an X11 session."
        warn "Run:  sudo reboot"
    fi
    echo ""
}

cmd_status() {
    header "Remote-access status"

    local ts_state rd_state ts_ip rd_id sess dm
    ts_state=$(ts_status)
    rd_state=$(rd_status)
    sess=$(current_session_type)
    dm=$(active_dm)

    echo "Display:   ${dm} | session: ${sess}"
    if [ "$sess" = "wayland" ]; then
        warn "Wayland active — RustDesk capture/input will be broken until session is X11"
    fi
    echo ""

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
        echo "Last setup state (from ${STATE_FILE}):"
        cat "$STATE_FILE" | sed 's/^/  /'
    elif [ -e "$STATE_FILE" ]; then
        echo ""
        warn "State file exists but is not readable by current user."
        warn "Run:  sudo cat ${STATE_FILE}"
    fi
}

cmd_test() {
    header "Remote-access connectivity test"

    if internet_ok; then success "internet reachable"
    else                  warn   "no internet — Tailscale/RustDesk install would fail"
    fi

    local sess; sess=$(current_session_type)
    case "$sess" in
        x11)     success "session: X11 (RustDesk capture/input will work)" ;;
        wayland) warn    "session: Wayland — RustDesk will see a black screen / reject input. Run setup or reboot to apply X11." ;;
        *)       info    "session type: ${sess}" ;;
    esac

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

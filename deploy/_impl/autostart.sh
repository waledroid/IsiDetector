#!/usr/bin/env bash
# ============================================================================
# IsiDetector — boot-time stack installer (headless standalone mode)
#
# One systemd unit + one settings flip make a site PC survive power cuts,
# Docker restarts and closed browsers with detection running:
#
#   power on → systemd runs `docker compose up -d` (after docker.service)
#            → container's auto_start replays the last successful Start
#            → detection + UDP + event CSV running — no browser, no clicks.
#
# The browser is a viewer only — open Chrome manually to watch; closing it
# changes nothing. This script NEVER touches display-manager config
# (auto-login, Wayland/X11) — the old kiosk layers are gone. remote.sh owns
# the Wayland/X question for RustDesk.
#
# Usage:
#   ./autostart.sh enable [USER]   install the systemd unit, set
#                                  auto_start=true, clean legacy kiosk
#                                  leftovers (sudo, auto-escalates).
#                                  USER only affects file-ownership fallback.
#   ./autostart.sh disable         remove the unit + legacy cleanup (sudo)
#   ./autostart.sh status          read-only — unit + auto_start state,
#                                  legacy-artifact warning, site diagnostics
#   ./autostart.sh -h | --help     this help
#
# One-time prerequisite for hands-free boots: click Start once on the
# dashboard so camera + model are recorded (last_model_type/last_weights);
# auto_start replays them on every boot from then on.
#
# Legacy: earlier versions installed OS auto-login (GDM3 custom.conf edits,
# WaylandEnable=false) and a kiosk-Chrome .desktop at login. enable/disable
# both remove those leftovers. AutomaticLogin* lines are stripped only when
# our .pre-autostart-* backups prove this script wrote them; WaylandEnable
# is never touched here (co-owned by remote.sh — RustDesk needs X11).
# ============================================================================

set -euo pipefail

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$SCRIPT_DIR}"
SYSTEMD_UNIT="/etc/systemd/system/isidetector.service"

# ── Privilege helper ────────────────────────────────────────────────────────
need_root() {
    if [[ $(id -u) -ne 0 ]]; then
        echo "▶ '$1' needs root — re-executing with sudo…"
        exec sudo -E env "INSTALL_DIR=$INSTALL_DIR" "ORIG_USER=${SUDO_USER:-$USER}" "$0" "$@"
    fi
}

# ── User resolution ─────────────────────────────────────────────────────────
# The install-dir owner is who settings.json must keep belonging to, and who
# the systemd unit runs compose as. Falls back to the invoking user when the
# install dir is root-owned.
_svc_user() {
    local u
    u="$(stat -c '%U' "$INSTALL_DIR")"
    if [[ -z "$u" || "$u" == "root" ]]; then
        u="${ORIG_USER:-${SUDO_USER:-root}}"
    fi
    echo "$u"
}

resolve_target_user() {
    local u="${1:-}"
    if [[ -z "$u" ]]; then
        u="${SUDO_USER:-${ORIG_USER:-}}"
    fi
    if [[ -z "$u" || "$u" = "root" ]]; then
        u=$(awk -F: '$3 >= 1000 && $3 < 65000 {print $1; exit}' /etc/passwd)
    fi
    echo "${u:-root}"
}

# ── Display-manager detection (diagnostics only — never written to) ─────────
detect_dm() {
    if [[ -f /etc/gdm3/custom.conf ]] || systemctl is-active --quiet gdm3 2>/dev/null; then
        echo "gdm3"; return
    fi
    if [[ -d /etc/lightdm ]] || systemctl is-active --quiet lightdm 2>/dev/null; then
        echo "lightdm"; return
    fi
    if [[ -f /etc/sddm.conf || -d /etc/sddm.conf.d ]] || systemctl is-active --quiet sddm 2>/dev/null; then
        echo "sddm"; return
    fi
    echo ""
}

# ── Legacy kiosk cleanup ────────────────────────────────────────────────────
# Removes what the old three-layer installer left behind. Idempotent; silent
# when there's nothing to do. Runs from both enable and disable.
#
# GDM3 rule: strip AutomaticLogin* ONLY when a .pre-autostart-* backup
# exists — the backup is the proof this script (not a sysadmin) wrote those
# lines. We deliberately do NOT restore the backup wholesale: it may predate
# a WaylandEnable=false that remote.sh wrote later, and reverting that would
# break RustDesk capture on the next reboot. WaylandEnable is never touched.
_legacy_cleanup() {
    local removed=0 f
    for f in /home/*/.config/autostart/isidetector.desktop \
             /root/.config/autostart/isidetector.desktop; do
        if [[ -f "$f" ]]; then
            rm -f "$f"
            echo "  ✓ legacy: removed $f"
            removed=1
        fi
    done

    local conf="/etc/gdm3/custom.conf"
    if [[ -f "$conf" ]] \
       && compgen -G "${conf}.pre-autostart-*" >/dev/null \
       && grep -qE '^[[:space:]]*AutomaticLogin' "$conf" 2>/dev/null; then
        sed -i -E \
            -e '/^[[:space:]]*AutomaticLoginEnable[[:space:]]*=/d' \
            -e '/^[[:space:]]*AutomaticLogin[[:space:]]*=/d' \
            "$conf"
        echo "  ✓ legacy: stripped AutomaticLogin* from $conf"
        echo "      (WaylandEnable left as-is — owned by remote.sh; backups kept on disk)"
        removed=1
    fi

    for f in /etc/lightdm/lightdm.conf.d/50-isidetector-autologin.conf \
             /etc/sddm.conf.d/50-isidetector-autologin.conf; do
        if [[ -f "$f" ]]; then
            rm -f "$f"
            echo "  ✓ legacy: removed $f"
            removed=1
        fi
    done

    if [[ "$removed" -eq 1 ]]; then
        echo "  ⓘ auto-login/kiosk removal takes effect on next reboot"
    else
        echo "  ✓ no legacy kiosk artifacts found"
    fi
}

# Read-only probe used by status. Mirrors _legacy_cleanup's detection rules.
_legacy_artifacts_present() {
    local f
    for f in /home/*/.config/autostart/isidetector.desktop \
             /root/.config/autostart/isidetector.desktop; do
        [[ -f "$f" ]] && return 0
    done
    if [[ -f /etc/gdm3/custom.conf ]] \
       && compgen -G "/etc/gdm3/custom.conf.pre-autostart-*" >/dev/null \
       && grep -qE '^[[:space:]]*AutomaticLogin' /etc/gdm3/custom.conf 2>/dev/null; then
        return 0
    fi
    [[ -f /etc/lightdm/lightdm.conf.d/50-isidetector-autologin.conf ]] && return 0
    [[ -f /etc/sddm.conf.d/50-isidetector-autologin.conf ]] && return 0
    return 1
}

# ── Systemd unit ────────────────────────────────────────────────────────────
_install_unit() {
    if [[ ! -f "$INSTALL_DIR/compose.yaml" && ! -f "$INSTALL_DIR/deploy/docker-compose.yml" ]]; then
        echo "✗ No compose file found under $INSTALL_DIR." >&2
        echo "  Set INSTALL_DIR=/path/to/your/clone and re-run." >&2
        exit 1
    fi

    local svc_user
    svc_user="$(_svc_user)"

    cat > "$SYSTEMD_UNIT" <<EOF
[Unit]
Description=IsiDetector inference stack (docker compose)
Documentation=https://github.com/waledroid/IsiDetector
After=docker.service network-online.target
Wants=network-online.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$INSTALL_DIR
User=$svc_user
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=300
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
EOF

    chmod 644 "$SYSTEMD_UNIT"
    systemctl daemon-reload
    systemctl enable isidetector.service
    systemctl start isidetector.service || true

    echo "✓ Systemd unit installed + enabled at $SYSTEMD_UNIT"
    echo "  WorkingDirectory: $INSTALL_DIR"
    echo "  User:             $svc_user"
    echo "  Will run:         docker compose up -d  (at boot, after docker.service)"
}

_remove_unit() {
    if [[ ! -f "$SYSTEMD_UNIT" ]]; then
        echo "ℹ No systemd unit at $SYSTEMD_UNIT (already removed)."
        return 0
    fi
    systemctl stop isidetector.service || true
    systemctl disable isidetector.service || true
    rm -f "$SYSTEMD_UNIT"
    systemctl daemon-reload
    echo "✓ Removed systemd unit + disabled."
}

# ── auto_start flip ─────────────────────────────────────────────────────────
# Sets auto_start=true in both webapps' settings.json. Direct file edit on
# purpose: POST /api/settings needs a dev-auth session, and auto_start is
# only read once at container boot — a live POST buys nothing. Failure here
# is a warning, never fatal (the unit install must stand regardless).
_set_auto_start() {
    local owner="${1:?owner required}"
    local sf found=0
    for sf in "$INSTALL_DIR/webapp/isitec_app/settings.json" \
              "$INSTALL_DIR/webapp/isitec_api/settings.json"; do
        [[ -f "$sf" ]] || continue
        found=1
        if python3 - "$sf" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
if data.get("auto_start") is not True:
    data["auto_start"] = True
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
PY
        then
            chown "$owner:$owner" "$sf" 2>/dev/null || true
            echo "  ✓ auto_start=true in $sf"
        else
            echo "  ⚠ could not update $sf — enable it in Settings → Camera instead"
        fi
    done
    if [[ "$found" -eq 0 ]]; then
        echo "  ⚠ no settings.json found under $INSTALL_DIR/webapp —"
        echo "    enable 'Auto-start stream on boot' in the Settings UI instead"
    fi
}

# Read-only auto_start probe for status: prints true / false / (unreadable).
_auto_start_state() {
    local sf
    for sf in "$INSTALL_DIR/webapp/isitec_app/settings.json" \
              "$INSTALL_DIR/webapp/isitec_api/settings.json"; do
        [[ -f "$sf" ]] || continue
        python3 - "$sf" <<'PY' 2>/dev/null && return 0
import json, sys
with open(sys.argv[1]) as f:
    print(str(bool(json.load(f).get("auto_start"))).lower())
PY
    done
    echo "(unreadable)"
}

# ── enable ──────────────────────────────────────────────────────────────────
cmd_enable() {
    local target_user
    target_user=$(resolve_target_user "${1:-}")
    need_root enable "$target_user"

    echo "─── IsiDetector headless boot mode — enable ───────────────────"
    echo ""
    echo "▶ Legacy kiosk cleanup"
    _legacy_cleanup
    echo ""
    echo "▶ Systemd boot-time compose"
    _install_unit
    echo ""
    echo "▶ Detection auto-start (settings.json)"
    _set_auto_start "$(_svc_user)"
    echo ""
    echo "─── Enabled ───────────────────────────────────────────────────"
    echo "After every boot, expect:"
    echo "  • inference stack up via systemd (no login needed)"
    echo "  • detection auto-resumes the last successful Start"
    echo "  • no auto-login, no kiosk Chrome — open a browser manually to watch"
    echo ""
    echo "⚠ One-time prerequisite: click Start once on the dashboard so the"
    echo "  camera + model get recorded — auto_start replays them from then on."
}

# ── disable ─────────────────────────────────────────────────────────────────
cmd_disable() {
    need_root disable

    echo "─── IsiDetector headless boot mode — disable ──────────────────"
    echo ""
    echo "▶ Remove systemd unit"
    _remove_unit
    echo ""
    echo "▶ Legacy kiosk cleanup"
    _legacy_cleanup
    echo ""
    echo "─── Disabled ──────────────────────────────────────────────────"
    echo "The stack no longer starts at boot. auto_start in settings.json is"
    echo "left as-is (harmless without the boot unit; operator-owned)."
}

# ── status ──────────────────────────────────────────────────────────────────
cmd_status() {
    echo "─── IsiDetector headless boot mode status ───────────────────────"
    echo ""

    local systemd_state
    if [[ -f "$SYSTEMD_UNIT" ]]; then
        if systemctl is-enabled --quiet isidetector.service 2>/dev/null; then
            if systemctl is-active --quiet isidetector.service 2>/dev/null; then
                systemd_state="ENABLED + ACTIVE"
            else
                systemd_state="ENABLED (not active right now)"
            fi
        else
            systemd_state="installed but disabled"
        fi
    else
        systemd_state="not installed"
    fi
    printf "  Boot unit (systemd):   %s\n" "$systemd_state"
    printf "  auto_start (settings): %s\n" "$(_auto_start_state)"

    if _legacy_artifacts_present; then
        echo ""
        echo "  ⚠ legacy kiosk artifacts found (old auto-login / kiosk Chrome)."
        echo "    Run 'sudo ./autostart.sh enable' or 'disable' to clean them up."
    fi

    echo ""
    _run_diagnostics
}

# ── Diagnostics helpers ────────────────────────────────────────────────────
_safe() {
    local out
    if out="$("$@" 2>/dev/null)" && [[ -n "$out" ]]; then
        echo "$out"
    else
        echo "(unavailable)"
    fi
}

_session_type() {
    if [[ -n "${XDG_SESSION_TYPE:-}" ]]; then
        echo "$XDG_SESSION_TYPE"; return
    fi
    if command -v loginctl >/dev/null 2>&1; then
        local user="${SUDO_USER:-$USER}"
        local sid
        sid="$(loginctl show-user "$user" -p Display --value 2>/dev/null)"
        if [[ -n "$sid" ]]; then
            local t
            t="$(loginctl show-session "$sid" -p Type --value 2>/dev/null)"
            [[ -n "$t" ]] && { echo "$t"; return; }
        fi
    fi
    if pgrep -x Xwayland >/dev/null 2>&1; then echo "wayland"
    elif pgrep -x Xorg >/dev/null 2>&1; then    echo "x11"
    else                                         echo "unknown"
    fi
}

_distro_pretty() {
    [[ -r /etc/os-release ]] || { echo "unknown"; return; }
    awk -F= '/^PRETTY_NAME=/{gsub(/"/,"",$2); print $2; exit}' /etc/os-release
}

_run_diagnostics() {
    echo "─── System diagnostics ────────────────────────────────────────────"
    echo ""

    echo "System:"
    printf "  Distro:        %s\n" "$(_distro_pretty)"
    printf "  Kernel:        %s\n" "$(_safe uname -srm)"
    printf "  Uptime:        %s\n" "$(_safe uptime -p)"
    printf "  Boot time:     %s\n" "$(_safe who -b | awk '{print $3, $4}')"
    printf "  Time now:      %s\n" "$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo ""

    echo "User & session:"
    local cur_user target_user_resolved sess dm_detected
    cur_user="$(id -un 2>/dev/null) (uid $(id -u 2>/dev/null))"
    target_user_resolved="$(resolve_target_user "" 2>/dev/null || echo '(not detectable)')"
    sess="$(_session_type)"
    dm_detected="$(detect_dm)"
    printf "  Current user:  %s\n" "$cur_user"
    printf "  Target user:   %s\n" "$target_user_resolved"
    printf "  Session type:  %s\n" "$sess"
    if [[ "$sess" = "wayland" ]]; then
        printf "                 ⓘ Wayland active — RustDesk capture won't work.\n"
        printf "                   './remote.sh setup' switches GDM3 to X11 (reboot to apply).\n"
    fi
    printf "  Display mgr:   %s\n" "${dm_detected:-not detected}"
    echo ""

    echo "Resources:"
    if command -v df >/dev/null 2>&1; then
        df -h / 2>/dev/null | awk 'NR==2 {printf "  Disk /:        %s used / %s total (%s)\n", $3, $2, $5}'
    fi
    if command -v free >/dev/null 2>&1; then
        free -h 2>/dev/null | awk '/^Mem:/ {printf "  RAM:           %s used / %s total\n", $3, $2}'
    fi
    if [[ -r /proc/loadavg ]]; then
        printf "  Load (1/5/15): %s\n" "$(awk '{print $1, $2, $3}' /proc/loadavg)"
    fi
    echo ""

    echo "Network:"
    printf "  Hostname:      %s\n" "$(_safe hostname)"
    if command -v ip >/dev/null 2>&1; then
        local nic_list
        nic_list="$(ip -4 -o addr show 2>/dev/null \
            | awk '{print $2, $4}' \
            | awk '$1 !~ /^(lo|docker|br-|veth|tailscale)/ {printf "                 %s → %s\n", $1, $2}')"
        if [[ -n "$nic_list" ]]; then
            echo "  IPv4 NICs:"
            echo "$nic_list"
        else
            echo "  IPv4 NICs:     (none non-virtual found)"
        fi
    fi
    local internet="(unknown)"
    if curl -sS -m 4 -o /dev/null -w "%{http_code}" https://1.1.1.1/ 2>/dev/null | grep -qE '^(2|3)..'; then
        internet="reachable (HTTPS)"
    elif ping -c 1 -W 2 1.1.1.1 >/dev/null 2>&1; then
        internet="reachable (ICMP)"
    else
        internet="UNREACHABLE"
    fi
    printf "  Internet:      %s\n" "$internet"
    if getent hosts github.com >/dev/null 2>&1; then
        printf "  DNS:           working (github.com resolves)\n"
    else
        printf "  DNS:           NOT RESOLVING github.com\n"
    fi
    echo ""

    echo "Docker stack:"
    if command -v docker >/dev/null 2>&1; then
        if systemctl is-active --quiet docker 2>/dev/null; then
            printf "  Daemon:        active\n"
        else
            printf "  Daemon:        NOT ACTIVE\n"
        fi
        local img_line
        img_line="$(docker images --format '{{.Repository}}:{{.Tag}} ({{.Size}}, {{.CreatedSince}})' isitec-visionai 2>/dev/null | head -1)"
        if [[ -n "$img_line" ]]; then
            printf "  Image:         %s\n" "$img_line"
        else
            printf "  Image:         isitec-visionai not built yet (run ./up.sh)\n"
        fi
        local web_state
        web_state="$(docker ps --filter 'name=deploy-web-1' --format '{{.Names}} → {{.Status}}' 2>/dev/null | head -1)"
        if [[ -n "$web_state" ]]; then
            printf "  Web container: %s\n" "$web_state"
        else
            printf "  Web container: not running\n"
        fi
        if [[ -n "$web_state" ]]; then
            local backend
            backend="$(docker exec deploy-web-1 sh -c 'echo ${WEB_BACKEND:-flask}' 2>/dev/null)"
            printf "  Backend:       %s\n" "${backend:-unknown}"
            local mode_line
            mode_line="$(docker exec deploy-web-1 sh -c 'echo ${COMPOSE_MODE:-cpu}' 2>/dev/null)"
            printf "  Compose mode:  %s\n" "${mode_line:-unknown}"
        fi
        if command -v ss >/dev/null 2>&1; then
            if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ':9501$'; then
                printf "  Port 9501:     listening\n"
            else
                printf "  Port 9501:     NOT listening\n"
            fi
        fi
    else
        printf "  Docker not installed\n"
    fi
    echo ""

    echo "Remote access:"
    if command -v tailscale >/dev/null 2>&1; then
        local ts_ip
        ts_ip="$(tailscale ip -4 2>/dev/null | head -1)"
        if [[ -n "$ts_ip" ]]; then
            printf "  Tailscale:     connected — IP %s\n" "$ts_ip"
        elif tailscale status >/dev/null 2>&1; then
            printf "  Tailscale:     installed but no IPv4 (check admin/machines for pending approval)\n"
        else
            printf "  Tailscale:     installed but logged out\n"
        fi
    else
        printf "  Tailscale:     not installed\n"
    fi
    if systemctl is-active --quiet rustdesk.service 2>/dev/null; then
        printf "  RustDesk:      service active\n"
        if [[ -r /var/log/isidetector/remote-state.json ]]; then
            local rd_id
            rd_id="$(grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]*"' /var/log/isidetector/remote-state.json | head -1 | sed -E 's/.*"id"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/')"
            [[ -n "$rd_id" ]] && printf "  RustDesk ID:   %s (from /var/log/isidetector/remote-state.json)\n" "$rd_id"
        fi
    elif command -v rustdesk >/dev/null 2>&1; then
        printf "  RustDesk:      installed but service not active\n"
    else
        printf "  RustDesk:      not installed\n"
    fi
    echo ""

    echo "─── End diagnostics ───────────────────────────────────────────────"
    echo ""
}

# ── Subcommand dispatch ────────────────────────────────────────────────────
cmd="${1:-status}"
shift || true

case "$cmd" in
    enable)   cmd_enable "$@" ;;
    disable)  cmd_disable "$@" ;;
    status)   cmd_status ;;
    -h|--help)
        # The header docstring is the single source of truth — print it back.
        sed -n '2,36p' "$0" | sed 's/^# *//'
        ;;
    *)
        echo "Unknown subcommand: $cmd" >&2
        echo "Available:  enable [USER] | disable | status | --help" >&2
        echo "Try: $0 --help" >&2
        exit 2
        ;;
esac

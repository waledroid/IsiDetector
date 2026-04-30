#!/usr/bin/env bash
# ============================================================================
# IsiDetector — Standalone-mode helpers for site PCs
#
# Three independent layers turn a fresh site PC into a hands-free kiosk:
#
#   1. enable-autologin    OS-level auto-login → no operator at the login screen
#   2. enable-systemd      docker compose up at boot via systemd → stack runs
#                          before the desktop session even loads
#   3. enable              desktop autostart → opens kiosk Chrome on the dashboard
#
# Layer 3 alone (the original behaviour) waits for the desktop to come up,
# then runs `up.sh` which both starts compose AND opens Chrome. Adding
# layer 2 cuts ~30 s off cold boot because the stack is warming up while
# the desktop is still rendering. Adding layer 1 removes the only manual
# step (sitting at the login screen).
#
# Recommended on a real site PC: enable all three.
#
# Subcommands:
#   ./autostart.sh enable                  install desktop autostart (kiosk Chrome)
#   ./autostart.sh disable                 remove desktop autostart + systemd unit
#   ./autostart.sh status                  print state of all three layers
#
#   ./autostart.sh enable-systemd          install + start /etc/systemd/system/isidetector.service
#   ./autostart.sh disable-systemd         stop + remove the systemd unit
#
#   ./autostart.sh enable-autologin USER   write AutomaticLogin= for GDM3 / LightDM (sudo)
#   ./autostart.sh disable-autologin       remove AutomaticLogin= (sudo)
#
#   ./autostart.sh -h | --help             this help
#
# Notes:
#   - enable-systemd auto-rewrites the .desktop file (if present) to use
#     `up.sh --open-only` so the desktop layer doesn't race with systemd
#     to bring up compose.
#   - enable-autologin takes effect on next reboot. We don't restart the
#     display manager — that would log the operator out mid-setup.
#   - `disable` removes layers 2 and 3 but leaves auto-login alone (it's
#     OS-wide and the operator may want it for non-IsiDetector reasons).
# ============================================================================

set -euo pipefail

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$SCRIPT_DIR}"

AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_FILE="$AUTOSTART_DIR/isidetector.desktop"

SYSTEMD_UNIT="/etc/systemd/system/isidetector.service"

# Flags passed to up.sh. --no-build means no internet needed at boot;
# --kiosk forces fullscreen Chrome with no UI affordances. --force-cpu
# is conservative (a CPU-mode site PC where this script is most useful).
UP_FLAGS_FULL="--no-build --kiosk --force-cpu"
UP_FLAGS_OPEN_ONLY="--no-build --kiosk --force-cpu --open-only"

# Pick the up-flags based on whether systemd is currently the source of truth.
desktop_up_flags() {
    if [[ -f "$SYSTEMD_UNIT" ]]; then
        echo "$UP_FLAGS_OPEN_ONLY"
    else
        echo "$UP_FLAGS_FULL"
    fi
}

# ── Privilege helper ────────────────────────────────────────────────────────
# enable-systemd / disable-systemd / enable-autologin / disable-autologin
# need root. Re-exec via sudo -E if we aren't already.
need_root() {
    if [[ $(id -u) -ne 0 ]]; then
        echo "▶ '$1' needs root — re-executing with sudo…"
        # Preserve INSTALL_DIR / HOME so the re-exec sees the right paths
        # (sudo strips most env by default).
        exec sudo -E env "INSTALL_DIR=$INSTALL_DIR" "ORIG_USER=${SUDO_USER:-$USER}" "$0" "$@"
    fi
}

# ── Display-manager detection ──────────────────────────────────────────────
detect_dm() {
    # Returns: gdm3 | lightdm | sddm | "" (unknown)
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

# ── enable: desktop autostart ──────────────────────────────────────────────
cmd_enable() {
    if [[ ! -x "$INSTALL_DIR/up.sh" ]]; then
        echo "✗ Could not find executable up.sh at $INSTALL_DIR/up.sh" >&2
        echo "  Set INSTALL_DIR=/path/to/your/clone and re-run." >&2
        exit 1
    fi
    mkdir -p "$AUTOSTART_DIR"
    local flags
    flags="$(desktop_up_flags)"
    cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=IsiDetector
Comment=Bring up the inference stack and open the dashboard in kiosk mode
Exec=/usr/bin/env bash -c 'cd "$INSTALL_DIR" && ./up.sh $flags'
Path=$INSTALL_DIR
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=10
EOF
    chmod 644 "$DESKTOP_FILE"
    echo "✓ Desktop autostart enabled."
    echo "  File:  $DESKTOP_FILE"
    echo "  Runs: cd $INSTALL_DIR && ./up.sh $flags"
    if [[ "$flags" == *--open-only* ]]; then
        echo "  Mode: --open-only (systemd unit handles the compose stack)."
    else
        echo "  Mode: full up.sh (compose + browser). Run 'enable-systemd' for"
        echo "        faster cold boot."
    fi
    echo ""
    echo "  ⓘ For a fully-hands-off boot, also enable:"
    echo "    sudo ./autostart.sh enable-autologin $USER"
    echo "    sudo ./autostart.sh enable-systemd"
}

cmd_disable() {
    local removed_any=0
    if [[ -f "$DESKTOP_FILE" ]]; then
        rm -f "$DESKTOP_FILE"
        echo "✓ Removed $DESKTOP_FILE"
        removed_any=1
    fi
    if [[ -f "$SYSTEMD_UNIT" ]]; then
        echo "▶ Removing systemd unit too (use 'disable-systemd' alone if you"
        echo "  want to keep the desktop autostart)…"
        cmd_disable_systemd
        removed_any=1
    fi
    if [[ $removed_any -eq 0 ]]; then
        echo "ℹ Nothing to remove (no desktop autostart, no systemd unit)."
    fi
}

# ── enable-systemd: docker compose up at boot ──────────────────────────────
cmd_enable_systemd() {
    need_root "enable-systemd" "$@"

    if [[ ! -f "$INSTALL_DIR/compose.yaml" && ! -f "$INSTALL_DIR/deploy/docker-compose.yml" ]]; then
        echo "✗ No compose file found under $INSTALL_DIR." >&2
        echo "  Set INSTALL_DIR=/path/to/your/clone and re-run." >&2
        exit 1
    fi

    # Prefer the install dir's owner so settings.json file ownership stays
    # consistent across systemd-managed and operator-managed runs.
    local svc_user
    svc_user="$(stat -c '%U' "$INSTALL_DIR")"
    if [[ -z "$svc_user" || "$svc_user" == "root" ]]; then
        svc_user="${ORIG_USER:-${SUDO_USER:-root}}"
    fi

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
    echo ""

    # If desktop autostart is also installed, rewrite it to use --open-only
    # so it doesn't race with systemd. The operator's path stays the same;
    # only the Exec= line changes.
    if [[ -f "$DESKTOP_FILE" ]]; then
        local owner_home
        owner_home="$(getent passwd "$svc_user" | cut -d: -f6)"
        local user_desktop="$owner_home/.config/autostart/isidetector.desktop"
        if [[ -f "$user_desktop" ]]; then
            sed -i "s|./up.sh [^']*|./up.sh $UP_FLAGS_OPEN_ONLY|" "$user_desktop"
            echo "✓ Rewrote $user_desktop to use --open-only (no compose race)."
        fi
    else
        echo "  ⓘ No desktop autostart yet — run './autostart.sh enable' as your"
        echo "    user to install the kiosk-Chrome opener."
    fi
}

cmd_disable_systemd() {
    need_root "disable-systemd" "$@"
    if [[ ! -f "$SYSTEMD_UNIT" ]]; then
        echo "ℹ No systemd unit at $SYSTEMD_UNIT (already removed)."
        return 0
    fi
    systemctl stop isidetector.service || true
    systemctl disable isidetector.service || true
    rm -f "$SYSTEMD_UNIT"
    systemctl daemon-reload
    echo "✓ Removed systemd unit + disabled."

    # If desktop autostart is still installed, restore the full up.sh flags
    # so it goes back to handling compose itself.
    if [[ -f "$DESKTOP_FILE" ]]; then
        sed -i "s|./up.sh [^']*|./up.sh $UP_FLAGS_FULL|" "$DESKTOP_FILE"
        echo "✓ Restored $DESKTOP_FILE to full up.sh (compose + browser)."
    fi
}

# ── enable-autologin: GDM3 / LightDM / SDDM ────────────────────────────────
cmd_enable_autologin() {
    local target_user="${1:-}"
    if [[ -z "$target_user" ]]; then
        echo "Usage: $0 enable-autologin USER" >&2
        exit 2
    fi
    need_root "enable-autologin" "$target_user"

    if ! getent passwd "$target_user" >/dev/null; then
        echo "✗ User '$target_user' does not exist." >&2
        exit 1
    fi

    local dm
    dm="$(detect_dm)"
    if [[ -z "$dm" ]]; then
        echo "✗ Couldn't detect the display manager (no GDM3 / LightDM / SDDM)." >&2
        echo "  Set auto-login manually via your distro's Settings → Users panel." >&2
        exit 1
    fi

    case "$dm" in
        gdm3)
            local conf="/etc/gdm3/custom.conf"
            mkdir -p "$(dirname "$conf")"
            touch "$conf"
            # Strip any prior AutomaticLogin* lines, then write fresh ones
            # under the [daemon] section.
            sed -i '/^AutomaticLoginEnable=/d; /^AutomaticLogin=/d' "$conf"
            if grep -q '^\[daemon\]' "$conf"; then
                sed -i "/^\[daemon\]/a AutomaticLoginEnable=true\nAutomaticLogin=$target_user" "$conf"
            else
                printf "\n[daemon]\nAutomaticLoginEnable=true\nAutomaticLogin=%s\n" "$target_user" >> "$conf"
            fi
            echo "✓ GDM3 auto-login set for '$target_user' in $conf"
            ;;
        lightdm)
            local conf="/etc/lightdm/lightdm.conf.d/50-isidetector-autologin.conf"
            mkdir -p "$(dirname "$conf")"
            cat > "$conf" <<EOF
[Seat:*]
autologin-user=$target_user
autologin-user-timeout=0
EOF
            echo "✓ LightDM auto-login set for '$target_user' in $conf"
            ;;
        sddm)
            local conf="/etc/sddm.conf.d/50-isidetector-autologin.conf"
            mkdir -p "$(dirname "$conf")"
            cat > "$conf" <<EOF
[Autologin]
User=$target_user
Session=plasma.desktop
EOF
            echo "✓ SDDM auto-login set for '$target_user' in $conf"
            echo "  ⓘ If your session is not 'plasma.desktop', edit the Session= line."
            ;;
    esac

    echo ""
    echo "ⓘ Takes effect on next reboot — we don't restart $dm now to avoid"
    echo "  logging you out mid-setup."
}

cmd_disable_autologin() {
    need_root "disable-autologin"
    local dm
    dm="$(detect_dm)"
    case "$dm" in
        gdm3)
            local conf="/etc/gdm3/custom.conf"
            if [[ -f "$conf" ]]; then
                sed -i '/^AutomaticLoginEnable=/d; /^AutomaticLogin=/d' "$conf"
                echo "✓ Removed AutomaticLogin entries from $conf"
            fi
            ;;
        lightdm)
            rm -f /etc/lightdm/lightdm.conf.d/50-isidetector-autologin.conf
            echo "✓ Removed /etc/lightdm/lightdm.conf.d/50-isidetector-autologin.conf"
            ;;
        sddm)
            rm -f /etc/sddm.conf.d/50-isidetector-autologin.conf
            echo "✓ Removed /etc/sddm.conf.d/50-isidetector-autologin.conf"
            ;;
        *)
            echo "ℹ No supported display manager detected — nothing to remove."
            ;;
    esac
    echo "ⓘ Takes effect on next reboot."
}

# ── status ─────────────────────────────────────────────────────────────────
cmd_status() {
    echo "─── IsiDetector standalone mode status ──────────────────────────"
    echo ""

    # Layer 1: auto-login
    local dm autologin_state
    dm="$(detect_dm)"
    autologin_state="unknown"
    case "$dm" in
        gdm3)
            if [[ -f /etc/gdm3/custom.conf ]] && grep -q '^AutomaticLogin=' /etc/gdm3/custom.conf 2>/dev/null; then
                autologin_state="ENABLED ($(grep '^AutomaticLogin=' /etc/gdm3/custom.conf | head -1 | cut -d= -f2))"
            else
                autologin_state="disabled"
            fi
            ;;
        lightdm)
            if [[ -f /etc/lightdm/lightdm.conf.d/50-isidetector-autologin.conf ]]; then
                autologin_state="ENABLED ($(grep '^autologin-user=' /etc/lightdm/lightdm.conf.d/50-isidetector-autologin.conf | cut -d= -f2))"
            else
                autologin_state="disabled (or set elsewhere)"
            fi
            ;;
        sddm)
            if [[ -f /etc/sddm.conf.d/50-isidetector-autologin.conf ]]; then
                autologin_state="ENABLED ($(grep '^User=' /etc/sddm.conf.d/50-isidetector-autologin.conf | cut -d= -f2))"
            else
                autologin_state="disabled (or set elsewhere)"
            fi
            ;;
        *)
            autologin_state="display manager not detected"
            ;;
    esac
    printf "  1. Auto-login   (%s):  %s\n" "${dm:-?}" "$autologin_state"

    # Layer 2: systemd
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
    printf "  2. Systemd unit:                %s\n" "$systemd_state"

    # Layer 3: desktop autostart
    if [[ -f "$DESKTOP_FILE" ]]; then
        local exec_line
        exec_line="$(grep '^Exec=' "$DESKTOP_FILE" | sed 's/^Exec=//')"
        printf "  3. Desktop autostart:           %s\n" "ENABLED"
        printf "       %s\n" "$exec_line"
    else
        printf "  3. Desktop autostart:           %s\n" "not installed"
    fi

    echo ""
    echo "  Recommended on a real site PC: all three enabled. See --help."
    echo ""
}

# ── Subcommand dispatch ────────────────────────────────────────────────────
cmd="${1:-status}"
shift || true

case "$cmd" in
    enable)            cmd_enable "$@" ;;
    disable)           cmd_disable "$@" ;;
    enable-systemd)    cmd_enable_systemd "$@" ;;
    disable-systemd)   cmd_disable_systemd "$@" ;;
    enable-autologin)  cmd_enable_autologin "$@" ;;
    disable-autologin) cmd_disable_autologin "$@" ;;
    status)            cmd_status ;;
    -h|--help)         sed -n '2,42p' "$0" | sed 's/^# *//' ;;
    *)
        echo "Unknown subcommand: $cmd" >&2
        echo "Try: $0 --help" >&2
        exit 2
        ;;
esac

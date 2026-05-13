#!/usr/bin/env bash
# ============================================================================
# IsiDetector — Standalone-mode (hands-free kiosk) one-shot installer
#
# Turns a fresh site PC into a hands-free kiosk by applying THREE layers in
# one go (or rolling them back in one go):
#
#   Layer 1 — OS auto-login           (GDM3 / LightDM / SDDM, writes AutomaticLogin=)
#   Layer 2 — boot-time compose       (systemd unit runs `docker compose up -d`)
#   Layer 3 — kiosk Chrome at login   (~/.config/autostart .desktop file)
#
# After `enable + reboot`, the site PC goes power-on → ~30-40 s → kiosk
# Chrome on the dashboard with the inference stack already running. Zero
# clicks.
#
# Usage:
#   ./autostart.sh enable [USER]     install all three layers (sudo, auto-escalates)
#                                    USER defaults to the invoking user; pass it
#                                    explicitly when running under sudo from cron etc.
#   ./autostart.sh disable           reverse all three layers in one go (sudo)
#                                    restores /etc/gdm3/custom.conf from its
#                                    .pre-autostart-* backup, removes the systemd
#                                    unit, removes the .desktop autostart file.
#   ./autostart.sh status            read-only — print state of all three layers
#   ./autostart.sh -h | --help       this help
#
# Notes:
#   - When Layer 2 (systemd) is active, Layer 3 (.desktop) is auto-rewritten
#     to use `up.sh --open-only` so the desktop layer doesn't race with
#     systemd to bring up compose.
#   - Layers 1 (autologin) and the X11 switch take effect on the NEXT REBOOT.
#     The script doesn't restart the display manager — that would log the
#     operator out mid-setup.
#   - Layer 1 backs up /etc/gdm3/custom.conf to .pre-autostart-<timestamp>
#     before any edit; `disable` restores the most recent such backup.
#   - For RustDesk to capture the screen after auto-login, the session must
#     be X11 not Wayland. `enable` sets WaylandEnable=false alongside the
#     autologin keys; `remote.sh setup` also writes this (with its own
#     .pre-remote-* backup that doesn't collide).
# ============================================================================

set -euo pipefail

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$SCRIPT_DIR}"

# Legacy globals — kept for the systemd-side rewrite logic + status read.
# When running under sudo (after need_root), $HOME is /root, so the
# DESKTOP_FILE path here points at root's autostart dir, not the operator's.
# That's why `_kiosk_install` resolves the right path from the target user
# instead of relying on these. cmd_status falls back to walking /home/* so
# it still works as a non-root read.
AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_FILE="$AUTOSTART_DIR/isidetector.desktop"

# Find the kiosk-Chrome .desktop file across every /home/*/.config/autostart
# regardless of which user invoked the script. Returns the first match, or
# empty string. Used by cmd_status so `sudo ./autostart.sh status` finds
# the operator's file even though sudo's HOME is /root.
find_desktop_file() {
    local f
    for d in "$HOME/.config/autostart" /home/*/.config/autostart; do
        f="$d/isidetector.desktop"
        if [[ -f "$f" ]]; then echo "$f"; return; fi
    done
    echo ""
}

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

# ── Target-user resolution ──────────────────────────────────────────────────
# enable orchestrates all three layers; Layer 1 needs to know which user
# gets AutomaticLogin=, Layer 3 writes a .desktop file in that user's
# autostart dir. The selection priority is:
#   1. explicit positional arg ($1)
#   2. $SUDO_USER (set when invoked via sudo)
#   3. $ORIG_USER (set by our own need_root re-exec)
#   4. first /home/* user with a real UID (1000+)
resolve_target_user() {
    local u="${1:-}"
    if [[ -z "$u" ]]; then
        u="${SUDO_USER:-${ORIG_USER:-}}"
    fi
    if [[ -z "$u" || "$u" = "root" ]]; then
        u=$(awk -F: '$3 >= 1000 && $3 < 65000 {print $1; exit}' /etc/passwd)
    fi
    if [[ -z "$u" ]]; then
        echo "✗ Could not resolve a target user. Pass one explicitly:" >&2
        echo "    sudo ./autostart.sh enable USERNAME" >&2
        return 1
    fi
    echo "$u"
}

target_user_home() {
    getent passwd "${1:?user required}" | cut -d: -f6
}

# ── Layer 3 internals: kiosk-Chrome desktop autostart ──────────────────────
# Writes ~/.config/autostart/isidetector.desktop in the target user's home.
# Called from cmd_enable after Layers 1 + 2 are in place.
_kiosk_install() {
    local target_user="${1:?user required}"
    local user_home; user_home=$(target_user_home "$target_user")
    if [[ -z "$user_home" || ! -d "$user_home" ]]; then
        echo "✗ Could not locate home directory for '$target_user'" >&2
        return 1
    fi
    local autostart_dir="$user_home/.config/autostart"
    local desktop_file="$autostart_dir/isidetector.desktop"

    if [[ ! -x "$INSTALL_DIR/up.sh" ]]; then
        echo "✗ Could not find executable up.sh at $INSTALL_DIR/up.sh" >&2
        echo "  Set INSTALL_DIR=/path/to/your/clone and re-run." >&2
        return 1
    fi
    mkdir -p "$autostart_dir"
    local flags
    flags="$(desktop_up_flags)"
    cat > "$desktop_file" <<EOF
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
    chmod 644 "$desktop_file"
    # Fix ownership — we may be running as root via sudo; the file needs
    # to belong to target_user so their session can read it at login.
    chown -R "${target_user}:${target_user}" "$autostart_dir" 2>/dev/null || true

    echo "  ✓ Layer 3 (kiosk Chrome) — wrote $desktop_file"
    if [[ "$flags" == *--open-only* ]]; then
        echo "      Mode: --open-only (systemd unit owns the compose lifecycle)"
    else
        echo "      Mode: full up.sh (compose + browser at login)"
    fi
}

_kiosk_uninstall() {
    local target_user="${1:?user required}"
    local user_home; user_home=$(target_user_home "$target_user")
    if [[ -z "$user_home" ]]; then
        echo "  ℹ Layer 3 — no home dir for '$target_user', nothing to remove"
        return 0
    fi
    local desktop_file="$user_home/.config/autostart/isidetector.desktop"
    if [[ -f "$desktop_file" ]]; then
        rm -f "$desktop_file"
        echo "  ✓ Layer 3 (kiosk Chrome) — removed $desktop_file"
    else
        echo "  ℹ Layer 3 — no $desktop_file (already disabled)"
    fi
}

# ── enable: one-shot install of all three layers ───────────────────────────
cmd_enable() {
    local target_user
    target_user=$(resolve_target_user "${1:-}") || exit 1
    need_root enable "$target_user"

    echo "─── IsiDetector standalone mode — enabling all three layers ────"
    echo "Target user: $target_user"
    echo ""

    # Layer 1 — autologin (sudo-required, already root after need_root)
    echo "▶ Layer 1 — OS auto-login"
    cmd_enable_autologin "$target_user" \
        | sed 's/^/  /'
    echo ""

    # Layer 2 — systemd unit (also sudo-required)
    echo "▶ Layer 2 — systemd boot-time compose"
    cmd_enable_systemd \
        | sed 's/^/  /'
    echo ""

    # Layer 3 — desktop autostart in target_user's home
    echo "▶ Layer 3 — kiosk Chrome at login"
    _kiosk_install "$target_user"
    echo ""

    echo "─── All three layers enabled ──────────────────────────────────"
    echo "Reboot to apply:  sudo reboot"
    echo ""
    echo "After reboot, expect:"
    echo "  • $target_user is auto-logged-in (no password prompt)"
    echo "  • Session is X11 (RustDesk capture works)"
    echo "  • Inference stack is up via systemd"
    echo "  • Kiosk Chrome opens to the dashboard within ~30 s of login"
}

# ── disable: one-shot rollback of all three layers ─────────────────────────
cmd_disable() {
    local target_user
    target_user=$(resolve_target_user "${1:-}") || exit 1
    need_root disable "$target_user"

    echo "─── IsiDetector standalone mode — disabling all three layers ───"
    echo "Target user: $target_user"
    echo ""

    # Reverse-order rollback. Layer 3 first because removing the .desktop
    # has zero side-effects on running boots; Layer 1 last because its
    # GDM3 edit may need its backup restored.

    # Layer 3 — desktop autostart
    echo "▶ Layer 3 — remove kiosk Chrome .desktop"
    _kiosk_uninstall "$target_user"
    echo ""

    # Layer 2 — systemd unit
    echo "▶ Layer 2 — remove systemd unit"
    if [[ -f "$SYSTEMD_UNIT" ]]; then
        cmd_disable_systemd | sed 's/^/  /'
    else
        echo "  ℹ no $SYSTEMD_UNIT — already disabled"
    fi
    echo ""

    # Layer 1 — autologin + Wayland-disable, restored from .pre-autostart-* backup
    echo "▶ Layer 1 — restore display-manager config from backup"
    cmd_disable_autologin | sed 's/^/  /'
    echo ""

    echo "─── All three layers disabled ─────────────────────────────────"
    echo "Reboot to apply the login-screen change:  sudo reboot"
    echo ""
    echo "After reboot, expect:"
    echo "  • Normal login prompt (no auto-login)"
    echo "  • Inference stack does NOT auto-start"
    echo "  • Kiosk Chrome does NOT auto-launch"
    echo "  • Tailscale + RustDesk (if installed via ./remote.sh) still work"
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
        echo "Internal: cmd_enable_autologin requires a target user (called from cmd_enable)" >&2
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

            # 1. Backup once with a timestamp suffix. Suffix `.pre-autostart-*`
            #    is distinct from remote.sh's `.pre-remote-*` so both scripts'
            #    backups coexist without overwriting each other.
            local backup="${conf}.pre-autostart-$(date +%Y%m%d-%H%M%S)"
            cp -a "$conf" "$backup" 2>/dev/null \
                && echo "  backup: $backup" \
                || echo "  ⚠ could not create backup; continuing"

            # 2. Strip every variant of the keys we manage. Each pattern is its
            #    own sed expression — no embedded newlines, no shell-escape
            #    gymnastics that bit us before.
            sed -i -E \
                -e '/^[[:space:]]*AutomaticLoginEnable[[:space:]]*=/d' \
                -e '/^[[:space:]]*AutomaticLogin[[:space:]]*=/d' \
                -e '/^[[:space:]]*WaylandEnable[[:space:]]*=/d' \
                -e '/^[[:space:]]*#[[:space:]]*WaylandEnable[[:space:]]*=/d' \
                "$conf"

            # 3. Make sure [daemon] exists. printf DOES interpret \n correctly
            #    (unlike sed's `a` command — the historical bug here was that
            #    bash double-quotes don't expand \n into newlines either, so
            #    a one-shot `sed -i ".../a A\nB\nC"` wrote one long line with
            #    literal backslash-n separators, breaking GDM3's parser).
            if ! grep -qE '^[[:space:]]*\[daemon\]' "$conf"; then
                printf '\n[daemon]\n' >> "$conf"
            fi

            # 4. Insert each key on its own line — one `sed -i ...a LINE` per
            #    key, no embedded newlines. Works on every GNU sed we'll meet.
            #    Keys land in reverse insertion order under [daemon]; GDM3
            #    parses by key, not file position, so ordering doesn't matter.
            for kv in \
                "WaylandEnable=false" \
                "AutomaticLoginEnable=true" \
                "AutomaticLogin=$target_user"; do
                sed -i "/^\[daemon\]/a ${kv}" "$conf"
            done

            # 5. Post-edit verification. If any expected line is missing or
            #    malformed, restore from backup rather than leave a corrupted
            #    custom.conf behind (which would break the login screen).
            if grep -qE '^WaylandEnable=false$' "$conf" \
               && grep -qE '^AutomaticLoginEnable=true$' "$conf" \
               && grep -qE "^AutomaticLogin=${target_user}$" "$conf"; then
                echo "✓ GDM3 auto-login set for '$target_user' in $conf (Wayland disabled for RustDesk)"
            else
                echo "  ⚠ post-edit verification failed — restoring from $backup"
                cp -a "$backup" "$conf" 2>/dev/null
                echo "✗ GDM3 custom.conf edit failed; inspect manually:"
                echo "    sudo cat $conf"
                return 1
            fi
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
            # Prefer restoring from the most recent .pre-autostart-* backup —
            # cleanest revert path, also recovers any [daemon] keys the
            # original sysadmin had that we stripped on enable.
            local newest_backup
            newest_backup=$(ls -t "${conf}".pre-autostart-* 2>/dev/null | head -1)
            if [[ -n "$newest_backup" && -r "$newest_backup" ]]; then
                cp -a "$newest_backup" "$conf"
                echo "✓ Restored $conf from $newest_backup"
            elif [[ -f "$conf" ]]; then
                # No backup found → surgical strip of AutomaticLogin* lines
                # only. Leave WaylandEnable= alone; that key is co-owned with
                # remote.sh, which has its own .pre-remote-* backups.
                sed -i -E \
                    -e '/^[[:space:]]*AutomaticLoginEnable[[:space:]]*=/d' \
                    -e '/^[[:space:]]*AutomaticLogin[[:space:]]*=/d' \
                    "$conf"
                echo "✓ Removed AutomaticLogin entries from $conf (WaylandEnable left intact)"
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

    # Layer 3: desktop autostart — search every user's autostart dir, not
    # just the script-invoker's $HOME (under sudo $HOME is /root and we'd
    # miss the operator's file).
    local desktop_file_found
    desktop_file_found="$(find_desktop_file)"
    if [[ -n "$desktop_file_found" ]]; then
        local exec_line
        exec_line="$(grep '^Exec=' "$desktop_file_found" | sed 's/^Exec=//')"
        printf "  3. Desktop autostart:           %s\n" "ENABLED ($desktop_file_found)"
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
    enable)   cmd_enable "$@" ;;
    disable)  cmd_disable "$@" ;;
    status)   cmd_status ;;
    -h|--help)
        # The header docstring is the single source of truth — print it back.
        sed -n '2,40p' "$0" | sed 's/^# *//'
        ;;
    *)
        echo "Unknown subcommand: $cmd" >&2
        echo "Available:  enable [USER] | disable | status | --help" >&2
        echo "Try: $0 --help" >&2
        exit 2
        ;;
esac

#!/usr/bin/env bash
# ============================================================================
# IsiDetector — Desktop auto-start helper for site PCs
#
# Installs (or removes) a desktop autostart entry that makes the GNOME /
# XFCE / KDE session run `up.sh --no-build --kiosk` right after login.
# Combined with the OS's auto-login setting, the workflow becomes:
#
#   power on → boot → auto-login → autostart fires → docker compose up
#                                                  → web container ready
#                                                  → Chrome opens fullscreen
#
# Subcommands:
#   ./autostart.sh enable    create the .desktop file (idempotent)
#   ./autostart.sh disable   remove the .desktop file
#   ./autostart.sh status    print what's currently installed
#
# What it does NOT do:
#   - Does NOT enable the OS auto-login. That lives in
#     /etc/gdm3/custom.conf (GNOME) or the equivalent for your desktop
#     environment, and we leave it to the operator to set via the
#     Settings GUI (Users → Automatic Login). The .desktop file we
#     install only fires AFTER a user is logged in (manually or
#     automatically).
#
# Usage:
#   ./autostart.sh enable          # default: target this clone
#   INSTALL_DIR=/home/user/fps ./autostart.sh enable
# ============================================================================

set -euo pipefail

# ── Paths ───────────────────────────────────────────────────────────────────
# The repo root we want the autostart to point at. Default = the parent of
# this script's deploy/_impl/ directory; overridable so the same wrapper
# works from any clone path the operator picked.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$SCRIPT_DIR}"

# The .desktop file lives in the user's autostart dir — picked up by every
# major Linux desktop environment (GNOME, KDE, XFCE, MATE, …).
AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_FILE="$AUTOSTART_DIR/isidetector.desktop"

# Flags passed to up.sh. --no-build means no internet needed at boot;
# --kiosk forces fullscreen Chrome with no UI affordances.
UP_FLAGS="--no-build --kiosk --force-cpu"

# ── Subcommand dispatch ────────────────────────────────────────────────────
cmd="${1:-status}"

case "$cmd" in
    enable)
        if [[ ! -x "$INSTALL_DIR/up.sh" ]]; then
            echo "✗ Could not find executable up.sh at $INSTALL_DIR/up.sh" >&2
            echo "  Set INSTALL_DIR=/path/to/your/clone and re-run." >&2
            exit 1
        fi
        mkdir -p "$AUTOSTART_DIR"
        cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=IsiDetector
Comment=Bring up the inference stack and open the dashboard in kiosk mode
Exec=/usr/bin/env bash -c 'cd "$INSTALL_DIR" && ./up.sh $UP_FLAGS'
Path=$INSTALL_DIR
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=10
EOF
        chmod 644 "$DESKTOP_FILE"
        echo "✓ Autostart enabled."
        echo "  File:  $DESKTOP_FILE"
        echo "  Runs: cd $INSTALL_DIR && ./up.sh $UP_FLAGS"
        echo "  Delay: 10 s after desktop login (lets Docker daemon warm up)."
        echo ""
        echo "  ⓘ For a fully-hands-off boot, also enable auto-login in:"
        echo "    Settings → Users → Automatic Login (your account)"
        ;;

    disable)
        if [[ -f "$DESKTOP_FILE" ]]; then
            rm -f "$DESKTOP_FILE"
            echo "✓ Removed $DESKTOP_FILE"
        else
            echo "ℹ Already disabled (no $DESKTOP_FILE)."
        fi
        ;;

    status)
        if [[ -f "$DESKTOP_FILE" ]]; then
            echo "✅ Autostart is ENABLED."
            echo "   File:  $DESKTOP_FILE"
            echo "   Runs:"
            grep '^Exec=' "$DESKTOP_FILE" | sed 's/^Exec=/     /'
        else
            echo "❌ Autostart is DISABLED. Enable with: ./autostart.sh enable"
        fi
        ;;

    -h|--help)
        sed -n '2,32p' "$0" | sed 's/^# *//'
        ;;

    *)
        echo "Unknown subcommand: $cmd" >&2
        echo "Usage: $0 {enable|disable|status|--help}" >&2
        exit 2
        ;;
esac

#!/usr/bin/env bash
# tools/flow_record_watch.sh — flow-triggered auto-recorder (morning watcher).
#
# Site staff use the line at random times, so a fixed recording schedule misses
# real usage. This watcher tails the line-crossing event log the web container
# writes (volume-mounted to the host at isidet/logs/events/events_YYYY-MM-DD.csv)
# and, when "flow" is established — FLOW_COUNT crossings within FLOW_SPAN seconds
# — records RECORD_SECONDS of the camera via tools/rtsp_record_30min.sh.
#
# Semantics (agreed with ops):
#   - Checking runs from WINDOW_START to WINDOW_END (default 06:00–12:00).
#   - The recorder is called in the FOREGROUND: while a recording is in
#     progress no checking happens, so two sessions can never overlap.
#   - A recording started before WINDOW_END runs its full RECORD_SECONDS,
#     even past the window. Only *checking* stops at WINDOW_END.
#   - After a recording ends, checking resumes until WINDOW_END — multiple
#     sequential recordings per morning are possible.
#   - Only events logged AFTER the watcher starts count (EOF baseline), so a
#     PC booted mid-morning (Persistent= timer catch-up) never triggers on a
#     stale burst from hours ago. Likewise, events logged DURING a recording
#     are already on video and are skipped afterwards — a fresh
#     FLOW_COUNT-in-FLOW_SPAN burst is required to start the next session.
#
# Self-contained — this script is BOTH the watcher and its installer. It has
# nothing to do with ./autostart.sh / kiosk mode; the timer it installs is an
# independent unit pair (isidetector-flowrec.service + .timer).
#
# Usage:
#   tools/flow_record_watch.sh              run the watcher now (what systemd calls)
#   tools/flow_record_watch.sh install      install + enable the daily 06:00 systemd
#                                           timer (sudo, auto-escalates). Persistent=true
#                                           → a PC booted mid-morning still gets that
#                                           morning's run.
#   tools/flow_record_watch.sh uninstall    stop everything, remove timer + service (sudo)
#   tools/flow_record_watch.sh status       read-only: timer/service state + next fire
#   tools/flow_record_watch.sh -h|--help    this help
#
# All settings are env overrides, which is also the test surface:
#
#   WINDOW_END=23:59 RECORD_SECONDS=30 POLL_SECONDS=2 \
#   EVENTS_DIR=/tmp/fr_events OUT_ROOT=/tmp/fr_out \
#   RTSP_URL=rtsp://127.0.0.1:8554/test tools/flow_record_watch.sh
#
# For persistent site overrides create deploy/flowrec.env (KEY=value lines,
# untracked) — the installed service loads it if present.
#
# No site-specific values live here: the camera URL comes from the web app's
# settings.json (via the recorder's non-TTY fallback) and the repo path is
# discovered from this script's own location.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

WINDOW_START="${WINDOW_START:-06:00}"       # checking starts (sleep until it if early)
WINDOW_END="${WINDOW_END:-12:00}"           # all CHECKING stops here
FLOW_COUNT="${FLOW_COUNT:-3}"               # crossings needed...
FLOW_SPAN="${FLOW_SPAN:-60}"                # ...within this many seconds = "flow"
RECORD_SECONDS="${RECORD_SECONDS:-7200}"    # 2 h per session
POLL_SECONDS="${POLL_SECONDS:-5}"           # event-log poll interval
MIN_FREE_MB="${MIN_FREE_MB:-6000}"          # skip recording below this free space
EVENTS_DIR="${EVENTS_DIR:-$REPO/isidet/logs/events}"
OUT_ROOT="${OUT_ROOT:-$REPO/recordings}"
RTSP_URL="${RTSP_URL:-}"                    # empty → recorder reads settings.json

log() { echo "[flowrec] $*"; }

# ── install / uninstall / status (systemd timer, self-contained) ────────────
FLOWREC_SERVICE="/etc/systemd/system/isidetector-flowrec.service"
FLOWREC_TIMER="/etc/systemd/system/isidetector-flowrec.timer"

need_root() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "▶ '$1' needs root — re-executing with sudo…"
    exec sudo -E env "ORIG_USER=${SUDO_USER:-$USER}" "$0" "$@"
  fi
}

cmd_install() {
  need_root install
  command -v systemctl >/dev/null 2>&1 || {
    echo "✗ systemd not available on this host — install by hand or run the watcher manually." >&2
    exit 1
  }
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "⚠ ffmpeg not found on the host — the watcher will refuse to start." >&2
    echo "  Install it first:  sudo apt install ffmpeg" >&2
  fi

  # Run as the clone's owner so recordings/ ownership stays consistent with
  # the rest of the repo (same rationale as the main isidetector unit).
  local svc_user
  svc_user="$(stat -c '%U' "$REPO")"
  if [ -z "$svc_user" ] || [ "$svc_user" = "root" ]; then
    svc_user="${ORIG_USER:-${SUDO_USER:-root}}"
  fi

  cat > "$FLOWREC_SERVICE" <<EOF
[Unit]
Description=IsiDetector flow-triggered auto-recorder (morning watcher)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$svc_user
WorkingDirectory=$REPO
# Optional per-site/test overrides (WINDOW_END=, RECORD_SECONDS=, ...) —
# create the file only when you need it; it is not tracked in git.
EnvironmentFile=-$REPO/deploy/flowrec.env
ExecStart=$REPO/tools/flow_record_watch.sh
Nice=10
EOF

  cat > "$FLOWREC_TIMER" <<EOF
[Unit]
Description=Start the IsiDetector flow recorder every morning at 06:00

[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

  chmod 644 "$FLOWREC_SERVICE" "$FLOWREC_TIMER"
  systemctl daemon-reload
  systemctl enable --now isidetector-flowrec.timer

  echo "✓ Flow recorder installed + timer enabled."
  echo "  Watcher:     $REPO/tools/flow_record_watch.sh (user: $svc_user)"
  echo "  Fires:       every day at 06:00 (catches up if the PC was off then)"
  echo "  Behaviour:   ${FLOW_COUNT} crossings in ${FLOW_SPAN}s → record ${RECORD_SECONDS}s → resume checking, until $WINDOW_END"
  echo "  Recordings:  $OUT_ROOT/flow_*/   (NOT auto-pruned — clean up old"
  echo "               sessions manually; worst case ~21 GB per morning)"
  echo "  Watch it:    journalctl -fu isidetector-flowrec"
  systemctl list-timers isidetector-flowrec.timer --no-pager 2>/dev/null | sed 's/^/  /' || true
}

cmd_uninstall() {
  need_root uninstall
  if [ ! -f "$FLOWREC_TIMER" ] && [ ! -f "$FLOWREC_SERVICE" ]; then
    echo "ℹ Flow recorder not installed (nothing to remove)."
    return 0
  fi
  systemctl disable --now isidetector-flowrec.timer 2>/dev/null || true
  # Stopping the service TERMs any in-progress recording; the recorder traps
  # it and finalizes the current video chunk before exiting.
  systemctl stop isidetector-flowrec.service 2>/dev/null || true
  rm -f "$FLOWREC_TIMER" "$FLOWREC_SERVICE"
  systemctl daemon-reload
  echo "✓ Flow recorder timer + service removed."
  echo "  Existing recordings under $OUT_ROOT are untouched."
}

cmd_status() {
  if [ ! -f "$FLOWREC_TIMER" ]; then
    echo "Flow recorder: not installed  (install with: tools/flow_record_watch.sh install)"
    return 0
  fi
  if systemctl is-enabled --quiet isidetector-flowrec.timer 2>/dev/null; then
    echo "Flow recorder: ENABLED (fires 06:00 daily)"
  else
    echo "Flow recorder: installed but timer disabled"
  fi
  if systemctl is-active --quiet isidetector-flowrec.service 2>/dev/null; then
    echo "Watcher:       running NOW"
  else
    echo "Watcher:       not running right now (normal outside 06:00–12:00)"
  fi
  systemctl list-timers isidetector-flowrec.timer --no-pager 2>/dev/null || true
}

case "${1:-run}" in
  install)   cmd_install;   exit 0 ;;
  uninstall) cmd_uninstall; exit 0 ;;
  status)    cmd_status;    exit 0 ;;
  run)       : ;;                       # fall through to the watcher below
  -h|--help)
    # The header docstring is the single source of truth — print it back.
    sed -n '2,49p' "$0" | sed 's/^# *//'; exit 0 ;;
  *)
    echo "Unknown subcommand: $1" >&2
    echo "Available:  install | uninstall | status | --help  (no args = run the watcher)" >&2
    exit 2 ;;
esac

# ── Watcher ──────────────────────────────────────────────────────────────────
command -v ffmpeg >/dev/null 2>&1 || {
  log "ffmpeg not found on the host — install with: sudo apt install ffmpeg" >&2
  exit 1
}

# Single instance: guards a manual/test run alongside the systemd-started one
# (two watchers would mean two concurrent ffmpeg recordings).
LOCK="/tmp/isidetector-flowrec.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  log "another flow_record_watch is already running — exiting"
  exit 0
fi

mkdir -p "$OUT_ROOT"

START_EPOCH=$(date -d "today $WINDOW_START" +%s)
END_EPOCH=$(date -d "today $WINDOW_END" +%s)
now=$(date +%s)
if [ "$now" -ge "$END_EPOCH" ]; then
  log "already past $WINDOW_END — nothing to do today"
  exit 0
fi
if [ "$now" -lt "$START_EPOCH" ]; then
  log "before $WINDOW_START — sleeping $(( START_EPOCH - now ))s until the window opens"
  sleep $(( START_EPOCH - now ))
fi

do_record() {
  local avail_mb
  avail_mb=$(df --output=avail -m "$OUT_ROOT" | tail -1 | tr -d ' ')
  if [ "$avail_mb" -lt "$MIN_FREE_MB" ]; then
    log "SKIP recording: only ${avail_mb} MB free on $OUT_ROOT (< ${MIN_FREE_MB} MB)" >&2
    return 0                                # keep checking; space may be freed
  fi
  local out="$OUT_ROOT/flow_$(date +%Y%m%d_%H%M%S)"
  log "flow detected — recording ${RECORD_SECONDS}s to $out"
  # </dev/null forces the recorder's non-interactive branch: camera URL from
  # settings.json (unless RTSP_URL overrides), progress line every 5 min.
  # A recorder failure must not kill the watcher — log and resume checking.
  "$REPO/tools/rtsp_record_30min.sh" "$RTSP_URL" "$out" "$RECORD_SECONDS" </dev/null \
    || log "recorder exited nonzero — resuming checks" >&2
  log "recording session finished"
}

# ── Detection loop ───────────────────────────────────────────────────────────
# LAST_LINE = number of CSV lines already consumed (starts at current EOF so
# pre-existing events never trigger). WIN holds the epoch seconds of the last
# FLOW_COUNT unconsumed events (sliding window).
EVENTS_FILE="$EVENTS_DIR/events_$(date +%F).csv"
LAST_LINE=0
[ -f "$EVENTS_FILE" ] && LAST_LINE=$(wc -l < "$EVENTS_FILE")
log "watching $EVENTS_DIR (from line $(( LAST_LINE + 1 ))) until $WINDOW_END — trigger: ${FLOW_COUNT} crossings in ${FLOW_SPAN}s"

WIN=()
while :; do
  now=$(date +%s)
  if [ "$now" -ge "$END_EPOCH" ]; then
    log "reached $WINDOW_END — checking stopped"
    exit 0
  fi

  EVENTS_FILE="$EVENTS_DIR/events_$(date +%F).csv"   # recomputed → rollover-safe
  if [ -f "$EVENTS_FILE" ]; then
    total=$(wc -l < "$EVENTS_FILE")
    [ "$total" -lt "$LAST_LINE" ] && LAST_LINE=0     # file replaced/truncated → resync
    if [ "$total" -gt "$LAST_LINE" ]; then
      while IFS=, read -r ts _rest; do
        LAST_LINE=$(( LAST_LINE + 1 ))
        [ -z "$ts" ] || [ "$ts" = "ts" ] && continue          # blank / header row
        ep=$(date -d "$ts" +%s 2>/dev/null) || continue        # unparsable → skip
        WIN+=("$ep")
        [ "${#WIN[@]}" -gt "$FLOW_COUNT" ] && WIN=("${WIN[@]:1}")
        if [ "${#WIN[@]}" -eq "$FLOW_COUNT" ] && [ $(( WIN[-1] - WIN[0] )) -le "$FLOW_SPAN" ]; then
          do_record                                  # blocks for the whole session
          WIN=()
          # Events logged during the recording are already on video — jump to EOF.
          LAST_LINE=$(wc -l < "$EVENTS_FILE" 2>/dev/null || echo 0)
          continue 2                                 # outer loop: re-check the window
        fi
      done < <(tail -n +"$(( LAST_LINE + 1 ))" "$EVENTS_FILE")
    fi
  fi
  sleep "$POLL_SECONDS"
done

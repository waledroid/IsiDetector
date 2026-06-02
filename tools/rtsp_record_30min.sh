#!/usr/bin/env bash
# tools/rtsp_record_30min.sh — record up to 30 minutes of an RTSP stream to
# segmented MP4, with a live countdown progress bar.
#
# Lightweight + reliable: ffmpeg copies the camera's already-encoded H.264 to disk
# (no re-encode → near-zero CPU, no quality loss), over TCP (robust), split into
# 10-minute chunks so a hiccup costs one chunk, not the whole session. A hard
# 30-minute wall-clock cap holds even across reconnects, then it terminates
# cleanly (the last chunk is finalized).
#
# ffmpeg runs in the background so the shell can draw a countdown bar; ffmpeg's own
# output (incl. the harmless "Timestamps unset" notice) goes to <OUT_DIR>/ffmpeg.log.
#
# Camera URL:
#   - Pass it explicitly as arg 1 (wins, no prompt), OR
#   - Run with no URL: the default is read from the web app's settings.json
#     (`rtsp_url`) — i.e. whatever camera THIS site uses — and you confirm [Y/n].
#     Answer 'n' to type a different URL. (No per-site URL is hardcoded here.)
#
# Usage:
#   tools/rtsp_record_30min.sh                       # confirm settings.json camera, 30 min
#   tools/rtsp_record_30min.sh "rtsp://user:pass@ip:554/path"   # explicit URL
#   tools/rtsp_record_30min.sh "" recordings/test 30           # OUT_DIR + 30 s test
#   SEGMENT_SECONDS=120 tools/rtsp_record_30min.sh             # smaller 2-min chunks
#
# Stop early with Ctrl-C — the current chunk is finalized cleanly.
set -euo pipefail

# ── Resolve the camera URL ───────────────────────────────────────────────────
# Default = the camera currently configured in the web app (settings.json), so the
# tool follows whatever this site actually uses — no hardcoded per-site URL/creds.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETTINGS="$SCRIPT_DIR/../webapp/isitec_app/settings.json"
DEFAULT_URL=""
if [ -r "$SETTINGS" ]; then
  DEFAULT_URL="$(python3 -c "import json; print(json.load(open('$SETTINGS')).get('rtsp_url',''))" 2>/dev/null || true)"
fi
[ -z "$DEFAULT_URL" ] && DEFAULT_URL="rtsp://admin:admin@192.168.1.88:554/1"   # generic fallback

if [ -n "${1:-}" ]; then
  RTSP_URL="$1"                              # explicit URL wins — no prompt
elif [ -t 0 ]; then                          # interactive: confirm the default camera
  echo "Default camera (from settings.json):"
  echo "    $DEFAULT_URL"
  read -r -p "Use this camera? [Y/n] " _ans
  case "${_ans:-y}" in
    [Nn]*) read -r -p "Enter the RTSP URL to record: " RTSP_URL ;;
    *)     RTSP_URL="$DEFAULT_URL" ;;
  esac
  [ -z "${RTSP_URL:-}" ] && { echo "No URL given. Aborting." >&2; exit 1; }
else
  RTSP_URL="$DEFAULT_URL"                     # non-interactive (piped/background): use default
fi

OUT_DIR="${2:-recordings/$(date +%Y%m%d_%H%M%S)}"
MAX_SECONDS="${3:-1800}"          # hard cap: 30 minutes
SEGMENT_SECONDS="${SEGMENT_SECONDS:-600}"   # 10-minute chunks

command -v ffmpeg >/dev/null 2>&1 || {
  echo "ffmpeg not found. Install:  sudo apt install ffmpeg   (or: conda install -c conda-forge ffmpeg)" >&2
  exit 1
}

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/ffmpeg.log"

fmt_hms() { printf '%d:%02d:%02d' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )) $(( $1 % 60 )); }

PID=0
start_ffmpeg() {   # $1 = seconds to record this run (the remaining budget)
  ffmpeg -hide_banner -loglevel warning \
    -fflags +genpts -rtsp_transport tcp -i "$RTSP_URL" \
    -t "$1" -c copy -an \
    -f segment -segment_time "$SEGMENT_SECONDS" -reset_timestamps 1 \
    -strftime 1 "$OUT_DIR/cam_%Y%m%d_%H%M%S.mp4" \
    >>"$LOG" 2>&1 &
  PID=$!
}

draw_bar() {   # $1 elapsed  $2 total  $3 nfiles
  local elapsed=$1 total=$2 nfiles=$3 width=30
  local pct=$(( total > 0 ? elapsed * 100 / total : 0 ))
  [ "$pct" -gt 100 ] && pct=100
  local filled=$(( pct * width / 100 ))
  local bar
  bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' $(( width - filled )) '' | tr ' ' '-')"
  printf '\r  [%s] %3d%%  %s / %s  | %d chunk(s)  ' \
    "$bar" "$pct" "$(fmt_hms "$elapsed")" "$(fmt_hms "$total")" "$nfiles"
}

echo "Recording : $RTSP_URL"
echo "Output    : $OUT_DIR  (${SEGMENT_SECONDS}s chunks, log: $LOG)"
echo "Max time  : $(fmt_hms "$MAX_SECONDS")  — Ctrl-C to stop early"

STOP=0
trap 'STOP=1' INT TERM
START=$(date +%s)
DEADLINE=$(( START + MAX_SECONDS ))
TTY=0; [ -t 1 ] && TTY=1
last_log=0

start_ffmpeg "$MAX_SECONDS"

while [ "$STOP" -eq 0 ]; do
  now=$(date +%s)
  elapsed=$(( now - START ))
  remaining=$(( DEADLINE - now ))
  [ "$remaining" -le 0 ] && break

  # Reconnect if ffmpeg died early (stream drop) with time still left.
  if ! kill -0 "$PID" 2>/dev/null; then
    [ "$TTY" -eq 1 ] && printf '\n'
    echo "[rtsp_record] stream dropped — reconnecting ($(fmt_hms "$remaining") left)…" >&2
    start_ffmpeg "$remaining"
  fi

  nfiles=$(find "$OUT_DIR" -maxdepth 1 -name 'cam_*.mp4' 2>/dev/null | wc -l)
  if [ "$TTY" -eq 1 ]; then
    draw_bar "$elapsed" "$MAX_SECONDS" "$nfiles"
  elif [ $(( elapsed - last_log )) -ge 300 ]; then   # non-TTY: a line every 5 min
    echo "[rtsp_record] $(fmt_hms "$elapsed") / $(fmt_hms "$MAX_SECONDS")  ($nfiles chunks)"
    last_log=$elapsed
  fi
  sleep 1
done

# Graceful stop: SIGINT lets ffmpeg write the current chunk's index before exiting.
if kill -0 "$PID" 2>/dev/null; then
  kill -INT "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
fi
[ "$TTY" -eq 1 ] && printf '\n'
echo "Done ($(fmt_hms $(( $(date +%s) - START ))) recorded). Files:"
ls -lh "$OUT_DIR"/cam_*.mp4 2>/dev/null || true

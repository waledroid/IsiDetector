#!/usr/bin/env bash
# io.sh — see the "wire" (relay pulse) in action, no hardware needed.
#
#   ./io.sh board [serial|modbus] [port] [bind_ip]   on-screen relay board (default: serial 4196 0.0.0.0)
#   ./io.sh status                                   live state of the relay-pulse publisher
#   ./io.sh test <channel>                           fire one pulse (works even while OFF)
#   ./io.sh on|off                                   toggle the wire output live
#
# Typical demo:  terminal 1: ./io.sh board       → Settings: driver Serial, device socket://127.0.0.1:4196, ON, Save
#                                               → every crossing prints CH1/CH2 ● ON … ○ OFF width=50 ms here.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
API="${ISI_API:-http://localhost:9501}"
cmd="${1:-help}"; shift || true

token() {
    local pw="${DEV_PASSWORD:-}"
    if [[ -z "$pw" ]]; then read -rsp "Dev password: " pw; echo >&2; fi
    curl -sf -X POST "$API/api/dev-auth" -H 'Content-Type: application/json' -d "{\"password\":\"$pw\"}" \
        | python3 -c 'import sys,json; print(json.load(sys.stdin)["token"])'
}

case "$cmd" in
    board)
        mode="${1:-serial}"; port="${2:-}"; bind="${3:-0.0.0.0}"
        [[ -z "$port" ]] && { [[ "$mode" == "modbus" ]] && port=1502 || port=4196; }
        echo "▶ Relay board emulator ($mode) on $bind:$port"
        if [[ "$mode" == "modbus" ]]; then echo "  App setting → driver: Modbus TCP · device: <this-ip>:$port"
        else echo "  App setting → driver: Serial · device: socket://<this-ip>:$port · protocol: Numato (or LCUS)"; fi
        exec python3 "$HERE/dio_board.py" "$mode" "$bind" "$port" ;;
    status)
        T=$(token); curl -sf "$API/api/dio" -H "X-Dev-Token: $T" | python3 -m json.tool ;;
    test)
        ch="${1:-1}"; T=$(token)
        curl -sf -o /dev/null -X POST "$API/api/dio/test" -H "X-Dev-Token: $T" -H 'Content-Type: application/json' -d "{\"channel\":$ch}" || { echo "pulse ch$ch: NOT queued (device dead / queue full)"; exit 1; }
        sleep 0.5   # let the worker fire it, then read back the truth
        curl -sf "$API/api/dio" -H "X-Dev-Token: $T" | python3 -c "
import sys, json
s = json.load(sys.stdin)['dio']
ok = s.get('connected') and s.get('last_channel') == $ch and not s.get('last_error')
print('pulse ch$ch:', 'FIRED ✓' if ok else 'FAILED ✗', '· connected=%s fired=%s err=%s' % (s.get('connected'), s.get('fired'), s.get('last_error')))" ;;
    on|off)
        T=$(token); v=$([[ "$cmd" == on ]] && echo true || echo false)
        curl -sf -X POST "$API/api/settings" -H "X-Dev-Token: $T" -H 'Content-Type: application/json' -d "{\"dio_enabled\":$v}" \
            | python3 -c 'import sys,json; d=json.load(sys.stdin); print("wire output:", "ON" if d["settings"].get("dio_enabled") else "OFF")' ;;
    *)
        sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//' ;;
esac

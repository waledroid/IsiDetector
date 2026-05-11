# ✅ Pre-Site Test Checklist

Verify every box before leaving the office. Tick as you go. **Goal: walk into the site knowing every code path has been exercised at least once on the dev bench.**

Run from `~/fps` on the dev box. CPU mode (`COMPOSE_MODE=cpu`) reflects the real site PC.

---

## 1. Build + boot

- [ ] `git pull origin fps` → no merge conflicts
- [ ] `git log --oneline -5` → expected recent commits present
- [ ] `./up.sh --force-cpu` → builds + starts cleanly, no `[FAIL]` lines
- [ ] `docker compose -p deploy ps` → `web` container `Up` and healthy
- [ ] `curl -sS http://localhost:9501/ -o /dev/null -w "%{http_code}\n"` → `200`
- [ ] `docker compose -p deploy logs web | grep -iE "error|traceback"` → clean (only known-benign lines, if any)
- [ ] Container env carries `COMPOSE_MODE=cpu` and `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;udp`

## 2. Mode + config plumbing

- [ ] `curl -sS http://localhost:9501/api/mode | python3 -m json.tool` → `mode: "cpu"`, `config_files: ["common.yaml", "cpu.yaml"]`
- [ ] Settings → Mode banner shows "⚡ CPU mode" + the loaded YAML files
- [ ] RF-DETR Configuration group is **hidden** on CPU mode
- [ ] Mode 2 entry in the model dropdown is **hidden** on CPU mode
- [ ] `cpu.yaml` knobs visible in container logs at start: `cpu_threads=8`, `performance_hint=LATENCY`, `match_thresh=0.7`

## 3. Model picker + INT8 detection

- [ ] `/api/models` lists `best_int8.xml` (NNCF OpenVINO IR pair gate works)
- [ ] `/api/models` lists `best.int8qdq.onnx` if present
- [ ] FP32 `.xml` model loads → mode footer = `OpenVINO • CPU`
- [ ] INT8 `.xml` model loads → mode footer = `OpenVINO INT8 • CPU`
- [ ] INT8 `.onnx` model loads → mode footer = `ONNX INT8 • CPU`
- [ ] Container logs show `Type: YOLO | Precision: INT8` line on INT8 load

## 4. Inference pipeline

- [ ] Stream Start with site-camera source → `/api/stats` shows `is_running: true`
- [ ] Performance tab loads (dev-auth gated)
- [ ] `forward_ms.p50` ≤ 12 ms with FP32, ≤ 9 ms with INT8 on dev box CPU
- [ ] `latency_ms.p50` − `forward_ms.p50` is small (< 2 ms)
- [ ] `frame_drops` ≥ 0 (camera-paced is OK, the diagnostic gates need this number visible)
- [ ] CPU Util < 30 % when streaming (lots of headroom available)
- [ ] CPU Model + ML Features chips visible (`avx`, `avx2` minimum; site PC may add `avx512f` / `avx512_vnni`)
- [ ] `track_frame_drop` increments are visible in stats when expected

## 5. ROI configurator

- [ ] Settings → Camera → "Show 'Set ROI' button on landing page" toggle works
- [ ] Set ROI button + Clear ROI button appear/hide on save
- [ ] Drag-rectangle on canvas: starts at correct location (no letterbox offset bug)
- [ ] Drag survives crossing the canvas edge (pointer-capture fix)
- [ ] Save ROI → `/api/settings` POST succeeds, `roi_points` persisted
- [ ] Clear ROI → `roi_points: []`, "Current ROI" reads "none (full frame)"
- [ ] After Stop/Start with ROI set, inference works on cropped region (check `frame.shape` in logs)

## 6. CLAHE preprocess toggle

- [ ] Settings → Camera → CLAHE checkbox visible
- [ ] Toggle ON + Save + Stop + Start → container logs show `🛡️ CLAHE preprocess enabled`
- [ ] `forward_ms` increases by ~0.5 ms with CLAHE on (matches expected cost)
- [ ] Visual: stream with CLAHE shows reduced glare on white surfaces
- [ ] Toggle OFF + Stop + Start → CLAHE log line gone, latency back down

## 7. UDP sort triggers

- [ ] Settings → Sorter → UDP host/port editable + persists
- [ ] On a line-crossing detection, `tcpdump -i lo -n udp port 9502` shows one datagram per crossing
- [ ] Datagram payload is valid JSON with `class`, `id`, `ts` keys
- [ ] `/api/performance` shows `udp.p50` / `udp.p95` populated
- [ ] Two close-together objects → two distinct datagrams in same frame
- [ ] `/api/udp` POST to retarget live → next event hits new target without stream restart

## 8. Encode throttle + display

- [ ] `/video_feed` (Flask MJPEG) streams smoothly in browser
- [ ] Visual FPS feels ≥ 10 (throttle is 2:1 so inference 25 fps → display 12.5 fps)
- [ ] Stop → Start cycle leaves stream working
- [ ] No `latest_annotated is None` errors in logs

## 9. Camera diagnostics

- [ ] `./cam_status.sh` runs to completion with current dev camera
- [ ] Output shows `📹 Stream: WxH @ N fps codec=…` from container logs
- [ ] At least one URL variant probe succeeds
- [ ] `./cam_status.sh --no-variants` works (faster path)

## 10. Network script (`net.sh`)

- [ ] `./net.sh show` → no sudo, prints current IP/gateway/DNS + UDP target
- [ ] `./net.sh test` → all green or yellow-skip (no `[FAIL]` on offline-tolerant checks)
- [ ] `./net.sh setup` → interactive prompt walks each NIC (dry-run if no spare NIC on dev box)
- [ ] Sudo escalation works (passes ORIG_ARGS through)

## 11. Remote access (`remote.sh`) — full cycle

- [ ] `./remote.sh status` on a clean box → shows `not-installed` for both
- [ ] `sudo ./remote.sh remove` on a system that's already clean → idempotent, no errors
- [ ] `sudo ./remote.sh setup` (interactive Gmail SSO path) → auto-detects auth completion, no manual press-Enter required
- [ ] Tailscale device appears in `https://login.tailscale.com/admin/machines`
- [ ] `./remote.sh status` → `Display: gdm3 \| session: x11`, Tailscale + RustDesk both running
- [ ] `/var/log/isidetector/remote-state.json` exists with plaintext password
- [ ] Password is `Isitec69+` (fleet default)
- [ ] Direct IP `100.x.x.x:21118` from another laptop's RustDesk client → connects + accepts `Isitec69+`
- [ ] 9-digit ID + password path also works (fallback when direct IP blocked)
- [ ] `sudo ./remote.sh remove` → full wipe, verification block reports CLEAN

## 12. Auto-start / kiosk

- [ ] `./autostart.sh status` → reports state of all three layers
- [ ] Layer 3 (`enable`) writes `~/.config/autostart/isidetector.desktop`
- [ ] Settings → Camera → "Auto-start stream on boot" toggle persists
- [ ] After container restart with auto_start=true + last_weights set → stream auto-starts (visible in logs)

## 13. Documentation

- [ ] `start.md` renders without broken links
- [ ] `site-install.md` uses `~/fps` everywhere (not `~/logistic`)
- [ ] `remote-setup.md` uses `~/fps` everywhere
- [ ] `list.md` includes printed-doc bullets for `site-install.md` + `remote-setup.md`
- [ ] `net.png` renders inline in `site-install.md` on GitHub

## 14. Hardware to carry

- [ ] Site PC pre-imaged with Ubuntu 22.04 + Docker installed
- [ ] `~/fps` cloned, on fps branch, up to date with origin
- [ ] Tested boot → autologin → kiosk (if going to a hands-free site)
- [ ] All items from `list.md` packed in the case
- [ ] Printed copies: `site-install.md`, `remote-setup.md`, `net.png`

## 15. Office-side prep

- [ ] Tailscale tailnet has spare ACL slot for the new device
- [ ] Office laptop joined to the tailnet, can `tailscale ssh` to existing nodes
- [ ] RustDesk installed on office laptop
- [ ] `isivision` Gmail credentials accessible to the engineer
- [ ] Customer's network sheet received (subnet, gateway, DNS, sorter IP/port)
- [ ] Camera ordered + delivery confirmed (Basler acA1440-73gc + Computar 8mm + LED bar)

---

## How to use this list

1. Check off as you verify each item on the dev bench.
2. If anything fails — fix or document the workaround **before** leaving.
3. After the site visit, mark items that passed in production and which didn't → feeds the next iteration of this checklist.
4. This file is committed to the repo so each site visit's go/no-go state is reproducible.

---

📋 Pair with [`list.md`](list.md) (packing checklist) and [`site-install.md`](site-install.md) (on-site runbook).

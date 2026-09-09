# IsiDetector — Site demo runbook (fps branch · Flask · UDP)

Audience: the client's team on-site. Goal: show **how it works, how it's operated, and how it's
diagnosed** — no slides. Everything below is what's on screen with the `fps` branch, Flask backend,
CPU mode, UDP sort trigger. Nothing here needs the relay/wire feature.

Budget: ~25 min. Functions (8) · Operations (7) · Diagnosis (7) · questions (rest).

---

## 0. Day before — on the dev box (done 2026-09-05)

- fps head is `1e744c1` (2026-07-30), clean, identical on `origin/fps` and `isitec/fps`.
- **Site PC checked live over Tailscale SSH (2026-09-05 16:20):** `root@100.66.111.95` (host `isi-linux`,
  user `isi-linux`, checkout `/home/isi-linux/fps`). Already at `1e744c1`, working tree clean, container
  code byte-identical to the checkout, Flask + CPU, `DEV_PASSWORD` set (value in `deploy/.env` on the
  site PC). Container healthy for 4 weeks, uptime 31 days, zero errors in the last 24 h, camera
  2880×1620 @ 25 fps HEVC. **No update needed.** Site values differ slightly from the rehearsal:
  conf 0.55, line 71 %, camera sub-stream (`stream=1.sdp`).
- USB bundle `~/fps-2026-09-05.bundle` (207 MB) exists as a fallback only; not needed.
- Gap on site: `webapp/isitec_app/uploads/` is **empty** — no Plan B recorded video yet.
- **Client-facing address, verified from a PC on the client's network (2026-09-05):**
  `http://192.168.17.179:9501`. This is the DHCP address on the client's cable (USB adapter, MAC
  `00:0E:C6:C2:8E:B9`, 24 h lease, unchanged for 31 days). Ask the client's IT for a DHCP reservation
  so it never moves. The other two addresses (192.168.1.50 camera net, 10.0.0.5 automate net) are
  not reachable from client desks.
- Rehearsed the stack locally in CPU mode with a recorded belt video → see §6 for what was verified.
- The rehearsal stack is still **running on the dev box**: http://localhost:9501 (Flask, CPU, fps
  `1e744c1`, dev password `change-me`). Click through the whole order once yourself before you go.
  Stop it with `cd ~/logistic-fps && docker compose down`.

## 1. Morning of — on the site PC (15 min, before anyone is in the room)

```bash
cd ~/fps
git log --oneline -1                # → 1e744c1 (confirmed 2026-09-05). If older, see §7.
# Stack is normally already running (autostart, 4 weeks healthy). Only if it isn't:
./up.sh --force-cpu --no-build      # --no-build: no internet on site. Chrome opens http://localhost:9501
docker compose ps                   # web container Up
docker compose logs web | grep -E "📹 Stream:|error|Traceback"   # camera FPS line, no tracebacks
```

Then in the browser:

- [ ] **About** tab → confirm mode banner says CPU and the loaded YAMLs (`common.yaml`, `cpu.yaml`).
- [ ] Unlock the dev tabs once: **Access** (in the header) → dev password (`DEV_PASSWORD` in `deploy/.env`,
      default `change-me`). **Performance** and **Settings** tabs appear. Stay logged in.
- [ ] **Settings → Camera**: `rtsp_url` is the site camera, CLAHE **off**, ROI **on** (site crop).
- [ ] **Settings → Sorter**: `10.0.0.1 : 9502` (the automate).
- [ ] **Settings → Tracking Line**: horizontal, 70 %, top→bottom (matches the site belt).
- [ ] **Live Inference**: Start with the site camera, watch 30 s of real parcels counting. Stop.
- [ ] Second terminal, leave it open on the second screen / behind the browser:
      `sudo ./udp_monitor.sh watch` → shows each datagram leaving toward the automate.
      (A local Python listener will NOT see them — they go to 10.0.0.1, not localhost.)
- [ ] `./net.sh show` → the client cable still has 192.168.17.179. Open it once from your own laptop
      on the client network. That is the address you give them for their screens.
- [ ] Plan B ready: a recorded belt video already in `webapp/isitec_app/uploads/`
      (`cam_20260602_105526.mp4` is the site camera itself, June 2). Start once to confirm it plays.

If anything above fails → §7 before deciding what to show.

---

## 2. FUNCTIONS — "what it does" (8 min)

Start on **Live Inference**, source = site camera (Plan B: the recorded file).

1. **Start.** Point at the picture: coloured masks (green carton, orange polybag), the ID on
   each parcel, the counting line.
   *"A camera looks at the belt. For every parcel the software says carton or polybag, follows it
   with a number, and the moment its front edge crosses the line it counts it once."*
2. **Counters** on the right go up one at a time. Show a crossing → counter increments.
   *"Front edge, not centre: the sorter gets the signal as early as physically possible."*
3. **The UDP terminal** (the `udp_monitor.sh watch` window): one line per crossing.
   ```
   {"class": "carton", "seq": 118, "id": 42, "ts": "…"}
   ```
   *"This is the whole contract with your sorter. `class` is what to do; `seq` counts up by one so
   the PLC can prove nothing was lost; `id` lets it ignore a duplicated packet."*
4. **Two parcels close together** → two lines in the terminal, two counts. Nothing merged.
5. **Sorter outputs LED** (UDP dot under the source panel) blinks on each event — the operator
   sees the trigger leave without a terminal.

Do NOT promise the relay/wire pulse; it is not on this branch.

---

## 3. OPERATIONS — "how you run it" (7 min)

Everything an operator or technician touches, in the order they'd meet it.

1. **Daily life:** the PC boots → the stack starts by itself (`./autostart.sh status` in the terminal
   shows the systemd unit), Chrome opens the page, and if *Auto-start stream* is on it resumes the
   last camera + model with no click. Power cut → same thing on return.
2. **Language**: header EN/FR switch. Whole UI, live.
3. **Stop / Start** and **source**: camera URL or a file. Browse to the recorded file, Start → same
   pipeline, same counts. *"This is how we test a change without touching the line."*
4. **Model dropdown**: grouped by format (OpenVINO / ONNX — CPU mode lists only what runs here).
   Pick the other OpenVINO model **while running** → hot-swap in ~0.3 s, counters and IDs preserved,
   no restart. (Colours do **not** change: both are YOLO. Colours flip only on YOLO ↔ RF-DETR.)
   *"Model updates never cost you a count."*
5. **Settings** (dev-locked so operators can't break it), read top to bottom — this *is* the runtime
   state of the line:
   | Group | What it decides |
   |---|---|
   | Camera | which camera, ROI crop of the belt, CLAHE glare filter, auto-start on boot |
   | Sorter (UDP target) | where the trigger goes — change it, Save, next event goes there. No restart. |
   | Runtime mode | CPU / GPU banner + YAMLs loaded |
   | YOLO Configuration | which weight file, confidence threshold |
   | Tracking Line | orientation, position slider, belt direction |
   | Counting accuracy | `count_interpolate` (frame-gap tolerant crossing) and time-dedup |
   Persisted in `settings.json`; survives reboot and software update.
6. **Tracking Line live**: drag the slider → the line moves on the video immediately. Flip belt
   direction → the anchor edge flips. *"Commissioning takes minutes, not a code change."*
7. **Analytics** tab: Production Summary for the day, 24 h / 7 d / 30 d chart from the per-event CSV,
   **Export Events** (pick a from/to date) → CSV `ts,class,id,seq` so the automate's log can be
   reconciled line by line.
   Session Comparison (last 5) shows drift between runs.

---

## 4. DIAGNOSIS — "how you know what's wrong" (7 min)

**Performance** tab: seven groups, each a green / amber / red dot with a threshold.

| Group | Question it answers |
|---|---|
| Hardware | is the PC healthy (CPU %, RAM, temperature) |
| Throughput | are we keeping up with the belt (FPS, frame drops) |
| Detection Quality | is the AI seeing clearly (confidence, detections/frame) |
| Tracking | are parcels followed end to end (track losses) |
| Counting Rate | are parcels counted once (per class, per hour) |
| UDP Sort Trigger | is every event leaving, and how fast (publish count, p50/p95 µs) |
| Session Health | uptime, errors, last event |

**Tell the real story from the first site — "it's slow, your software can't keep up":**

1. Throughput shows ~17 FPS. Slow. But whose fault?
2. `latency_ms` p50 ≈ 12 ms, yet 1000/17 = 58 ms available per frame. We use 12, wait 46.
   **Idle 80 % of the time.**
3. `frame_drops` = 0, CPU ≈ 13 %. A busy PC dropping frames would be us. An idle PC dropping
   nothing is **waiting on the camera**.
4. Confirm in the terminal: `docker compose logs web | grep "📹 Stream:"` → the camera advertises
   17 fps. (`./cam_status.sh` does the full probe.)

*"Conclusion in one sentence: the camera sends 17 pictures a second, we could handle 80. Replace the
camera or fix the lighting — no software change."*

**Expect one amber light on this footage: Tracking.** `id_ratio` = unique IDs ÷ crossings; amber
above 2, red above 5. On the June 2 site video it reads ~2.4: the tracker re-numbers a parcel while it
is far from the line, the count is still exactly one per parcel. Use it as the example of "an amber
light tells the technician *what* to look at": here, lighting/camera FPS, not the sorter.
Detection Quality is green at the site confidence (0.7): amber below 0.75 avg, red below 0.55 or
when >25 % of detections are low-confidence.

**Second story — "the sorter isn't receiving":**
`./udp_monitor.sh status` shows target + route + interface; `watch` shows datagrams leaving. If they
leave the PC, the problem is the cable/switch/PLC side. `./net.sh test` proves egress end to end.

**Third — "counts are low":** Detection Quality confidence falling = lighting/dirt. Show the CLAHE
toggle and say we measured it: on this footage it cost 21 % of cartons, so it stays off here.

---

## 4b. Performance tab — reading guide (Surveillance des performances)

One line per number. Lights: 🟢 fine · 🟡 look · 🔴 act. The thresholds are the ones in the code.
Site reading of 2026-09-05 16:00 in the last column.

### État de la session — Session Health
| Number | Means | Fine when | Site now |
|---|---|---|---|
| Uptime | how long this stream has run without a restart | any | 385 h (16 days) |
| Status | LIVE = frames are arriving and being processed | LIVE | LIVE |
| Errors | exceptions caught in the loop since start | 0 (🟡 >0, 🔴 >5) | 0 |
| CUDA OOM | GPU ran out of memory (GPU hosts only) | 0 (🔴 >0) | 0 |
| Heartbeat | seconds since the loop last reported alive | <10 s (🟡 >10, 🔴 >30 = loop stuck) | 0 s |

### Matériel — Hardware
| Number | Means | Fine when | Site now |
|---|---|---|---|
| CPU Util | share of the whole PC busy | <80 % (🟡 >80, 🔴 >95) | 23 % |
| CPU Temp | processor temperature | <75 °C (🟡 >75, 🔴 >90 = dust, fan, enclosure) | 72 °C ⚠ close |
| CPU Freq / Cores / Model | what the PC is; not a health signal | — | i7-10710U, 12 threads |
| ML Features | CPU instruction sets the model can use (avx2 = OK) | avx2 present | avx, avx2, f16c, fma |
| System RAM | memory used / total | <85 % (🟡 >85, 🔴 >95) | 3.4 / 31 GB, 11 % |

### Débit — Throughput
| Number | Means | Fine when | Site now |
|---|---|---|---|
| FPS | frames processed per second = what the camera delivers | ≥20 (🟡 <20, 🔴 <10) | 25.0 |
| Latency | total time per frame: decode + model + tracking | <50 ms (🟡 >50, 🔴 >100) | 13.4 ms |
| Forward | the model alone | most of Latency | 13.1 ms |
| Tracker | ByteTrack alone | ~1 ms | 0.1 ms |
| Frame Drops | frames the camera failed to deliver, since start | 🟡 >100, 🔴 >1000 **cumulative** | 483 over 16 days = 1 every 50 min |

Rule of thumb: **1000 ÷ FPS is the time available per frame.** At 25 fps that is 40 ms; we use 13.
If Latency is far below that budget and Frame Drops barely move, the PC is waiting on the camera,
not the other way round.

### Qualité de détection — Detection Quality
| Number | Means | Fine when | Site now |
|---|---|---|---|
| Avg Confidence | how sure the model is, averaged over recent detections | ≥0.75 (🟡 <0.75, 🔴 <0.55) | 0.73 🟡 |
| Low-Conf Rate | share of detections under 0.60 | ≤10 % (🟡 >10 %, 🔴 >25 %) | 9.8 % |
| Dets / Frame | parcels seen per frame right now | >0 when the belt runs; 0.0 = belt empty | 0.0 (idle) |
| Mask Coverage | how much of the box the mask fills; low = fragmented shapes | ~0.8+ | 0.81 |

Falling confidence with the same parcels = **lighting, glare, dirty lens, or camera moved**. Not the
software. The fix is on the belt, and the light tells you before counts start to slip.

### Suivi — Tracking
| Number | Means | Fine when | Site now |
|---|---|---|---|
| Unique IDs | tracker numbers handed out since start | — | 72 163 |
| Total Crossings | parcels counted at the line | — | 21 224 |
| ID Ratio | Unique IDs ÷ Crossings; 1.0 = every parcel kept one number | ≤2 (🟡 >2, 🔴 >5) | 3.40 🟡 |

An amber ID Ratio means parcels get renumbered on the way to the line (occlusion, jitter, lighting).
Counting is still one per parcel as long as the number is stable **at the line**, which is why the
count light is green while this one is amber. Rising toward 5 = look at lighting and camera FPS.

### Taux de comptage — Counting Rate
| Number | Means | Fine when | Site now |
|---|---|---|---|
| Carton / Polybag | totals this session, and the hourly rate | 🟡 nothing counted for 5 min while running, 🔴 10 min | 6 817 (17.7/h) · 37 229 (96.5/h) |

The hourly rate is over the whole session, so it dilutes over nights and weekends. Compare it with
the Analytics 24 h chart for "today".

### UDP Sort Trigger
| Number | Means | Fine when |
|---|---|---|
| Published | datagrams sent to the sorter this session | grows with crossings |
| p50 / p95 / p99 / max | time to hand one datagram to the network, in µs | p95 <500 µs (🟡 ≥500, 🔴 ≥1000) |

Green here proves the message **left the PC**. If the sorter still sees nothing, the problem is cable,
switch, or PLC: run `./udp_monitor.sh watch` on the PC to see the datagrams leaving, then `./net.sh test`.

### The 30-second read, in order
1. **Session** LIVE, Heartbeat near 0, Errors 0 → the software is alive.
2. **Throughput** FPS ≈ camera FPS, Latency ≪ 1000/FPS, Frame Drops flat → PC is not the limit.
3. **Detection** confidence ≥0.75 → the camera sees well. Below → lighting/lens.
4. **Tracking** ID Ratio ≤2 → parcels are followed cleanly. Above → same lighting/FPS causes.
5. **Counting** moving and **UDP** green → the sorter is being told. If not → network side.

Two things to say about today's site screen: CPU temperature 72 °C is 3 °C under the amber line
(check the PC's airflow); Avg Confidence 0.73 is just under 0.75 with the polybag-heavy flow, which
is what the CLAHE/lighting discussion is about.

---

## 5. Likely questions

- **No GPU on our PC?** This *is* the CPU path: OpenVINO, ~40 FPS on recorded video, more than any
  site camera delivers.
- **Duplicates on the network?** `id` + `seq` in every datagram.
- **Latency camera → trigger?** ~12 ms inference + tracking; UDP publish in µs (histogram on screen).
- **Accuracy?** 94–97 % mask mAP50 depending on model size; the CPU model is 94 %.
- **Updates without internet?** USB: a git bundle + `./up.sh --no-build`. Settings survive.
- **Remote support?** RustDesk/Tailscale via `./remote.sh` when the site allows a network path.
- **Wire / relay trigger instead of UDP?** Built and tested in the office, not on this site's
  build yet. Roadmap item, can be added without touching UDP.
- **Next steps:** UDP receipt confirmation from the PLC side, higher-FPS camera, LED bar on the belt.

---

## 6. What was rehearsed on the dev box (2026-09-05, CPU mode, Flask, fps `1e744c1`)

Fresh `Dockerfile.cpu` build from the fps worktree, site `settings.json` (ROI crop, line 70 %
horizontal top→bottom, CLAHE off, conf 0.7), source = `cam_20260602_105526.mp4` (site camera).
Every step below was exercised through the same endpoints the UI calls, plus a UDP listener on the
host to witness the datagrams.

| Step | Result |
|---|---|
| Boot | `Running on http://0.0.0.0:9501` in ~15 s; `/api/mode` → cpu, `common.yaml + cpu.yaml` |
| Dev unlock | `/api/dev-auth` with `change-me` → Performance + Settings available |
| Model list | 6 YOLO entries: 2× OpenVINO `model.xml`, `best_int8.xml`, 2× `best.onnx`, `best.fp16.onnx` |
| Start + count | running in <5 s; 7 cartons counted on the first 40 s of the clip |
| UDP | retargeted live to the host; 7 datagrams received, `seq` 1→7 with no gap, `{"class","seq","ts","id"}`; publish p50 ≈ 110 µs, max < 600 µs |
| Performance | hardware / throughput / counting / udp / session green; detection green at conf 0.7; **tracking amber** (id_ratio 2.4, see §4) |
| Throughput | 22 FPS on the file, forward 14 ms, latency 15 ms, tracker 0.8 ms, frame_drops 0 |
| Hot-swap | yolo26n_320_200 → "22-06-2026 black" while running: 0.31 s, counts kept (3 → 3, stream never stopped) |
| Line | `POST /api/line` position 0.6 → applied live, read back |
| Analytics | `/api/chart?period=24h` returns buckets + series; `/api/events/export?from=…&to=…` → CSV `ts,class,id,seq` |
| Video | `/video_feed` MJPEG streams (≈ 450 KB in 3 s) |
| Stop | clean; container log has no tracebacks (only the known supervision ByteTrack deprecation warning) |

Not rehearsed here: the real RTSP camera and the 10.0.0.1 route (site only) — that's what §1 is for.

---

## 7. If it breaks

| Symptom | Do |
|---|---|
| Site PC is behind (`git log -1` ≠ `1e744c1`) | `git update-index --skip-worktree webapp/isitec_app/settings.json` (keeps site settings) → `git fetch /media/$USER/<usb>/fps-2026-09-05.bundle fps && git merge --ff-only FETCH_HEAD` → `./up.sh --force-cpu` (image rebuild uses the local layer cache; no dependency changed since May). If the build wants the network, stay on the old commit and demo it. |
| Chrome opens, page blank | `docker compose logs -f web`; wait for `Running on http://`. `./up.sh --open-only` re-opens. |
| Camera Start fails | `./cam_status.sh` (probes URL variants). Plan B: recorded file. Message is identical. |
| No datagrams in `udp_monitor.sh watch` | check Settings → Sorter target; `./udp_monitor.sh test` (loopback self-test) proves the publisher. |
| Perf/Settings tabs missing | not logged in: Access → dev password. |
| Counts look wrong | ROI crop and line position first (Settings), CLAHE off, then confidence 0.7. |
| Stack won't start at all | `docker compose down && ./up.sh --force-cpu --no-build`. Last resort: reboot; autostart brings it back. |

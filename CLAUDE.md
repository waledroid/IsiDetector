# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **You're on the `deploy` branch.**
>
> This is the **runtime-only subset** of the project, meant for site PCs that only need to run the inference stack. It does NOT contain:
>
> - `compression/` — the office-workstation compression tool
> - `mkdocs/` — documentation source and built site
> - `isidet/src/training/` — the trainers, hooks, and BaseTrainer
> - `isidet/scripts/` — training/eval/debug scripts (run_train, run_val, prep_rfdetr_data, extract_frames, onnx_checker, etc.)
>
> If you need any of the above, switch to the `main` branch, which is the full project source. Day-to-day development happens on `main`; `deploy` is rebuilt from it whenever the runtime surface changes. Everything below this note still applies to what IS on this branch — commands that reference `isidet/scripts/` or `compression/` simply don't exist here.

## TL;DR — start here

For anything operational on a site PC, open **`start.md`** at the repo root — the one-page cheat-sheet for install → daily start → network lock-down → update → troubleshooting.

This file is the deep reference. Read `start.md` first; come back here when you need the "why".

---

## Project Overview

**IsiDetector** is a modular, config-driven instance segmentation pipeline for industrial parcel detection (cartons and polybags on conveyor belts). Two model families — **YOLO** (CNN) and **RF-DETR** (Transformer) — switchable via a single config line. Two peer web backends — **Flask** + **FastAPI** — share the same inference core. Deploys identically on **GPU** (NVIDIA CUDA) and **CPU-only** (Intel OpenVINO) hardware.

## Branch model — two branches, one source of truth

| Branch | Purpose | Contains |
|---|---|---|
| **`main`** | Canonical dev branch. Day-to-day work happens there. | The full project: training code, compression tool, docs, data prep, everything |
| **`deploy`** (this one) | Lean runtime subset shipped to site PCs. Derived from `main`. | Runtime-only — no `compression/`, no `mkdocs/`, no `isidet/src/training/`, no `isidet/scripts/` |

Site PCs clone **`deploy`**. Office / dev workstations clone **`main`**. Refresh procedure (run from office PC, not here): see `main`'s CLAUDE.md and the commit message of `a6ead19` for the canonical deletion set.

## Repository layout (deploy branch — lean)

```
logistic/
├── isidet/
│   ├── src/{inference,shared,utils,preprocess}/   # runtime inference code only
│   ├── configs/                                    # train.yaml + optimizers/
│   ├── models/pretrained/                          # yolo26n.pt + yolov8m-seg.pt baselines
│   └── rfdetr_service.py                           # GPU sidecar entry (if GPU host)
├── webapp/
│   ├── isitec_app/         # Flask backend
│   └── isitec_api/         # FastAPI backend (WebSocket-first)
└── deploy/
    ├── Dockerfile{,.cpu,.rfdetr}
    ├── docker-compose{,.cpu,.gpu}.yml
    └── _impl/*.sh           # real bodies — up / run_start / install / net
```

Plus root thin wrappers (`up.sh` / `run_start.sh` / `install.sh` / `net.sh`) that `exec` into `deploy/_impl/`, a `compose.yaml` re-export for repo-root `docker compose` commands, and `start.md` / `README.md` / `CLAUDE.md` (this file).

**Runtime conventions** that let the restructure be seamless:

- Thin-wrapper scripts at the repo root `exec` into `deploy/_impl/`. `./up.sh` at root continues to work from any CWD.
- `PYTHONPATH=isidet` is set inside every Docker image, every shell wrapper, and every webapp `sys.path` hack — so every `from src.X import Y` resolves without a single rewrite.
- Project-name pin: root `compose.yaml` has `name: deploy` so plain `docker compose down/logs/ps/exec` from the repo root targets the same containers that `./up.sh` (which `cd`s into `deploy/`) creates.

## Common Commands

### Deployment (Docker — production & daily ops)

```bash
./run_start.sh              # first time per host: Docker + NVIDIA toolkit + build images
./up.sh                     # daily starter; auto GPU vs CPU, waits for readiness, opens Chrome
./up.sh --force-cpu         # force CPU image
./up.sh --force-gpu         # require CUDA

docker compose down         # stop the stack (works from repo root via compose.yaml include)
docker compose logs -f web  # tail live logs
docker compose ps           # list running containers
```

Two-container architecture:
- **`web`** — Flask (`webapp/isitec_app/`) or FastAPI (`webapp/isitec_api/`) + ONNX Runtime, Ultralytics YOLO, OpenVINO, and (on GPU hosts) TensorRT. Ports 9501 TCP + 9502 UDP.
- **`rfdetr` sidecar** — Isolated PyTorch + `rfdetr` library for native `.pth` inference. Port 9510 internal-only. Only built on GPU hosts (gated by `profiles: [gpu]`); skipped on CPU-only.

Site-PC deployment walkthrough: `start.md` at the repo root.

### Inference from CLI (debug, via docker exec)

```bash
# Enter the container and run one of the inference entry points — note that on
# the deploy branch, isidet/scripts/ isn't shipped. The web UI is the primary
# operator interface; CLI inference is a debug/rescue path.
docker compose exec web bash
# then inside the container, write a one-liner or scp a script in.
```

### Network lock-down (site PC workflow — `net.sh`)

```bash
./net.sh show        # current IP/gateway/DNS + UDP target (read-only, no sudo, offline-tolerant)
./net.sh setup       # interactive multi-NIC freeze (auto-escalates to sudo)
./net.sh test        # reachability + live UDP egress probe; offline/no-docker → yellow skip
./net.sh apply       # legacy single-NIC freeze of the DHCP-issued config (auto-escalates to sudo)
./net.sh revert      # back to DHCP (auto-escalates to sudo)
```

`setup` is the right entry point for typical site PCs: two LAN NICs (e.g. `enp1s0` for the camera subnet, `enp2s0` for the automate subnet), no default gateway, no internet uplink. It enumerates every physical NIC via `ip -o link` (so down interfaces are still listed), prompts per NIC for `static / dhcp / skip`, then writes via `nmcli connection modify` — or `connection add` if no profile exists yet — with `autoconnect=yes` + `autoconnect-priority=100` so the config survives reboot. Gateway and DNS are optional in the static path (blank is fine on a no-uplink site).

`apply` is the older single-NIC flow that freezes whatever the active DHCP profile picked up; useful on a normal LAN with a real default gateway.

`test`/`apply`/`revert`/`setup` re-exec themselves via `sudo -E` if not already root (preserves flags via `ORIG_ARGS`). Read-only `show` runs as the regular user. Not for WSL2 or Ubuntu Server — gracefully errors with "NetworkManager not installed" if that's the host. The internet-reachability and Docker-egress checks in `test` degrade to yellow `skip` lines when the WAN cable is unplugged or the web container isn't running, so offline runs produce no `[FAIL]` noise.

### Standalone-mode helpers (`autostart.sh`)

Three independent layers turn a fresh site PC into a hands-free kiosk. Each layer can be enabled or reverted on its own — they compose but don't depend on each other.

```bash
sudo ./autostart.sh enable-autologin USER   # Layer 1 — OS skips login screen (GDM3/LightDM/SDDM)
sudo ./autostart.sh enable-systemd          # Layer 2 — docker compose up at boot via systemd
./autostart.sh enable                       # Layer 3 — desktop autostart opens kiosk Chrome on login
./autostart.sh status                       # print state of all three
sudo ./autostart.sh disable-autologin       # reverse Layer 1
sudo ./autostart.sh disable-systemd         # reverse Layer 2
./autostart.sh disable                      # reverse Layers 2 + 3 (leaves auto-login alone)
```

- **Layer 1** (`enable-autologin`) auto-detects the display manager and writes `AutomaticLogin=USER` to the right config (`/etc/gdm3/custom.conf`, LightDM `lightdm.conf.d/`, or SDDM `sddm.conf.d/`). Takes effect on next reboot — script does not restart the DM, that would log the operator out mid-setup.
- **Layer 2** (`enable-systemd`) installs `/etc/systemd/system/isidetector.service` that runs `docker compose up -d` from the install directory, ordered after `docker.service` + `network-online.target`. `User=` is set to the install-dir owner so settings.json file ownership stays consistent across systemd-managed and operator-managed runs.
- **Layer 3** (`enable`) writes `~/.config/autostart/isidetector.desktop`, which the desktop session runs ~10 s after login. Auto-rewrites itself to use `up.sh --open-only` (skip compose, just wait + open Chrome) when Layer 2 is also installed, so the two layers don't race.

Combined with the in-app **Settings → Camera → Auto-start stream on boot** toggle, the full hands-free path is: power on → ~30–40 s → kiosk Chrome on the dashboard with the stream running, zero clicks.

`up.sh --open-only` is a new flag that skips `docker compose up/down` entirely, waits briefly for `tcp/9501`, then opens the browser. Used by Layer 3 when Layer 2 owns the compose lifecycle.

---

## Architecture

IsiDetector uses a **Config-Driven + Registry + Strategy** pattern. On this branch, only the inference half is present.

### Layer 1: Configuration (`isidet/configs/`)

- `isidet/configs/train.yaml` — master switchboard: `model_type`, `nc`, `class_names`, `hooks`, inference settings.
- `isidet/configs/optimizers/*.yaml` — on `main` only (only used at training time).

### Layer 2: Registry (`isidet/src/shared/registry.py`)

- Three singleton registries: `TRAINERS`, `HOOKS`, `PREPROCESSORS`.
- Decorator-based: `@HOOKS.register('SomeHook')` on a class, then `HOOKS.get('SomeHook')` to retrieve it.
- On `deploy`, trainers aren't shipped — the registry still exists but inference code doesn't touch it.

### Layer 3: Training

Not present on this branch. See `main`.

### Layer 4: Inference (`isidet/src/inference/`)

Five backends, selected automatically by file extension in `StreamHandler._build_engine()`:

| Class | Extension | Device | Notes |
|---|---|---|---|
| `YOLOInferencer` | `.pt` | CUDA / CPU | Ultralytics native |
| `RFDETRInferencer` / `RemoteRFDETRInferencer` | `.pth` | CUDA | Native direct, or HTTP to rfdetr sidecar when running inside Docker |
| `OptimizedONNXInferencer` | `.onnx` | CUDA / CPU | Auto-detects YOLO vs RF-DETR family, handles post/pre-NMS YOLO, DETR class-index offset, CUDNN preload |
| `OpenVINOInferencer` | `.xml` | Intel CPU | Fastest CPU backend. **Hard-refuses** RF-DETR `.xml` at load time (OpenVINO 2026 mistranslates the transformer's Einsum ops) |
| `TensorRTInferencer` | `.engine` | NVIDIA GPU | Per-host compiled engines, 1.5–3× faster than ONNX-CUDA |

`BaseInferencer` holds shared preprocessing helpers. Auto-discovery walks `isidet/models/yolo/**/weights/*` and `isidet/models/rfdetr/**/*` with per-device priority (GPU: native/TensorRT first; CPU: OpenVINO/ONNX first).

### Layer 5: Support modules

- `isidet/src/shared/vision_engine.py` — unified orchestrator: ByteTrack tracking, line-crossing counting, annotation, telemetry.
- `isidet/src/utils/event_logger.py` — one CSV row per line-crossing to `isidet/logs/events/events_YYYY-MM-DD.csv`; daily rollover + 30-day rolling retention. Source of truth for the `/api/chart` history.
- `isidet/src/preprocess/clahe_engine.py` — SpecularGuard: CLAHE on LAB L-channel for industrial glare on polybags.

### Web Platform

Flask (`webapp/isitec_app/`) and FastAPI (`webapp/isitec_api/`) share `isidet/src/` via PYTHONPATH and serve identical UI.

- `app.py` — routes + session init (Flask has `/video_feed` MJPEG; FastAPI has `/ws/video` WebSocket + `/video_feed` MJPEG fallback).
- `stream_handler.py` — background inference thread, session state, locale (en/fr/de), UDP publisher.
- Endpoints: `POST /api/{start,stop,upload,language,dev-auth,dev-logout,udp,line,settings,belt_status}`, `GET /api/{stats,performance,chart?period=…,models,dev-check,settings,udp,line,snapshot}`. `GET /api/snapshot` returns one full-resolution raw camera frame as JPEG (the latest from `LiveReader.get_frame()`); used by the Live-page ROI configurator to draw the crop rectangle on a true full-res frame. Returns 404 when the stream isn't running.
- FastAPI-only: `/ws/video` (binary JPEG stream), `/ws/stats` (500 ms JSON tick). `/docs` and `/docs/{subpath}` serve the built MkDocs site on both backends (needs the site/ dir volume-mounted in; bare deploy branch doesn't ship pre-built docs).

## Persistent hot-swap

`VisionEngine.swap_inferencer(new_inferencer)` replaces the model **in place** without tearing down session state. Preserved across a swap:

- `class_totals` (running counts keyed by class name)
- `counted_ids` (set of ByteTrack IDs already triggered)
- The ByteTrack tracker instance (IoU matching carries tracks across the swap)
- `LineZone` position + anchor
- `EventLogger` (event CSV keeps appending to the same per-day file)

Rebuilt: the inferencer reference and palette-indexed annotators (mask/box/label), so per-class colours reflect the new model's class-ID convention. Swap latency ~2 s on GPU (CUDNN kernel cache is primed via `preload_onnx()` at container boot).

## Trigger semantics (sorter-first)

Line-crossing fires on the **leading edge** of the bbox — the side that enters the line zone first given belt direction. Maximises the sorter gate's reaction window. Mapping lives in `isidet/src/shared/vision_engine.py _ANCHOR_MAP`:

| Orientation | Belt direction | Anchor (`sv.Position`) |
|---|---|---|
| vertical | left_to_right | `CENTER_RIGHT` |
| vertical | right_to_left | `CENTER_LEFT` |
| horizontal | top_to_bottom | `BOTTOM_CENTER` |
| horizontal | bottom_to_top | `TOP_CENTER` |

`VisionEngine.process_frame()` returns a **list** of events; `_inference_loop` iterates and publishes one UDP datagram per crossing. Two close-together objects in the same frame both trigger their sort gates.

**Class-ID conventions** (intentional divergence for visual swap confirmation):

- YOLO emits `class_id ∈ {0, 1}` → palette slots `[0, 1]`
- RF-DETR emits `class_id ∈ {1, 2}` (DETR reserves index 0 for background) → palette slots `[1, 2]`

Operators see colours flip on YOLO ↔ RF-DETR swap as confirmation the swap took effect. Counts are keyed by class **name**, so the count dict is unaffected.

## UDP sorting broadcast (`stream_handler.py` — `UDPPublisher`)

On every line-crossing event, a ~60-byte JSON datagram is fired to the sorting machine controller. **One datagram per crossing** (including multiple-per-frame when objects are close together).

```json
{"class": "carton", "id": 42, "ts": "2026-03-31T14:23:45.312847"}
```

- `id` is the ByteTrack tracker ID — lets the sorter dedupe if the network duplicates the datagram. Optional; older consumers that only read `class` keep working.
- Transport: `socket.SOCK_DGRAM` — single socket created once at stream start, reused per event, no queue.
- Timestamp: microsecond ISO format (`.isoformat()`).
- Default target: `127.0.0.1:9502` (controller on same machine).
- **Configuration priority** (highest wins):
  1. Runtime API: `POST /api/udp {"host": "...", "port": ...}` — retargets live, no restart needed
  2. Env var: `UDP_HOST`, `UDP_PORT`
  3. `isidet/configs/train.yaml` → `inference.udp.host / port`
  4. Hardcoded default: `127.0.0.1:9502`
- Per-datagram latency histogram (p50 / p95 / p99 / max µs) surfaces in `/api/performance` so the automation engineer can see the real sort-trigger budget.

**Consumer-side minimal implementation:**

```python
import socket, json
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 9502))
while True:
    data, _ = sock.recvfrom(1024)       # blocks until event arrives — no polling
    event = json.loads(data)
    trigger_sort_gate(event["class"])   # act on "carton" or "polybag"
```

## Data flow (web inference)

```
POST /api/start → StreamHandler (hot-swap path if same source)
  → _build_engine() picks inferencer by file extension
  → VisionEngine.swap_inferencer(new) — counts/tracker/line preserved
  → _inference_loop:
      per frame:
        engine.process_frame(frame) → (annotated, detections, new_events[])
          └─ inside VisionEngine: each new crossing → EventLogger.log(class, id)
                                                     (→ isidet/logs/events/events_YYYY-MM-DD.csv)
        for event in new_events:
          latency_ns = UDPPublisher.publish(class, event_id)  → datagram → sorter (port 9502)
          monitor.track_udp_publish(latency_ns)                → histogram
          update last_detected + class_totals
      latest_annotated JPEG → /video_feed (MJPEG) or /ws/video (WebSocket)
  → /api/stats: live counts  →  /api/chart: event-log-bucketed history (24h/7d/30d)
```

## Classes

- `0: carton` (green in UI)
- `1: polybag` (orange in UI)

## Dependencies

- **`requirements.txt`** (repo root) — dev / host-side helpers. Mostly irrelevant on the deploy branch; kept for symmetry with `main`.
- **`deploy/requirements-deploy.txt`** — runtime deps baked into the web container (Flask/FastAPI, opencv, supervision, ultralytics, openvino, rfdetr).

The Docker images pull `torch` + `onnxruntime` separately (CPU or GPU wheels depending on the Dockerfile variant), then install `requirements-deploy.txt` on top.

## Settings keys (`webapp/isitec_*/settings.json`)

Single source of truth for operator-tunable runtime parameters. The Settings UI reads/writes via `GET`/`POST /api/settings`. Backend rejects unknown keys; values are validated server-side before being persisted. Two server-only keys are written by `stream_handler` itself and stripped from any client POST.

| Key | Type | Default | Purpose |
|---|---|---|---|
| `yolo_weights` / `rfdetr_weights` | str (path) | per-build | Model file the operator selected; respected on next Start. |
| `yolo_imgsz` / `detr_imgsz` | int | 320 / 416 | Inference input size hint. **OpenVINO `.xml` ignores this** (input shape is baked in at export); Ultralytics `.pt` and dynamic ONNX honor it. |
| `yolo_conf` / `detr_conf` | float | 0.55 / 0.35 | Confidence threshold. |
| `line_orientation` | `vertical`/`horizontal` | vertical | LineZone orientation. |
| `line_position` | float [0..1] | 0.5 | Fraction of frame width (vertical) or height (horizontal). After ROI crop, fraction is relative to the cropped frame. |
| `belt_direction` | `left_to_right`/`right_to_left`/`top_to_bottom`/`bottom_to_top` | left_to_right | Picks the bbox leading-edge anchor (see Trigger semantics). |
| `cpu_threads` | int [1..64] | 8 | OpenVINO `INFERENCE_NUM_THREADS`. |
| `skip_masks` / `skip_traces` | bool | false / false | Render shortcuts; significant FPS bump on busy belts. |
| `rtsp_url` | str | per-build | Saved camera URL used by the **📡 Site Camera** landing-page button. |
| `udp_host` / `udp_port` | str / int | 10.0.0.2 / 9502 | Sorter target. **Live-retargets** on save (publisher updates without stream restart). |
| `auto_start` | bool | false | If true, container boot replays the last successful Start (saved camera + last-used model) — no operator click needed. |
| `last_model_type` | str | "" | **Server-written.** Recorded after a successful Start; used by `auto_start`. Client POSTs cannot set this. |
| `last_weights` | str | "" | **Server-written.** Same as above. |
| `roi_enabled` | bool | false | If true, exposes the **📐 Set ROI** button on the Live Inference page. ROI crop only applies if both `roi_enabled` AND a valid 4-point `roi_points` are set. |
| `roi_points` | list of 0 or 4 `[x,y]` pairs | [] | Operator-drawn corner points in original camera-frame pixel coords. Backend computes the axis-aligned bounding rectangle and applies it as a numpy-slice crop in `_inference_loop` before the pre-engine resize. Any error → `self.roi = None` latch + log; pipeline never breaks. |

## Notes

- Pretrained weights tracked in repo: `isidet/models/pretrained/{yolo26n.pt,yolov8m-seg.pt}`. Everything else in `isidet/models/` is gitignored — populate via scp from the office PC.
- **Line defaults:** `line_position = 0.5` (centred), `belt_direction = "left_to_right"`. Operators change them via the Tracking Line settings panel or `POST /api/line`. After ROI crop, "0.5" means middle of the **cropped** frame (i.e. middle of the belt) — no code change needed, the line layer reads `frame.shape[:2]` which is now post-crop.
- **OpenVINO YOLO preprocess is resize-first.** `openvino_inferencer.preprocess()` runs `_letterbox` + `cv2.dnn.blobFromImage` (fused C++ swapRB + scale + transpose + add-batch) on the model-sized canvas instead of the raw input frame. Saves ~2 ms/frame on i7-10710U at 1080p input vs. the old transpose+astype+`/255` numpy chain.
- **ONNX session caching was removed** in favour of `preload_onnx()` which warms the CUDNN kernel cache at boot and discards the session. Cross-thread `session.run()` on a shared CUDA session triggered multi-second stream-sync stalls; per-thread rebuild reusing driver-level kernel cache avoids that at ~2 s cost per swap instead of 30–80 s.
- **Deployment marker** `deploy/.deployment.env` is written by `run_start.sh` (records `COMPOSE_MODE=gpu|cpu`) and consumed by `up.sh` to pick the right compose profile. Gitignored — per-host state.
- **Site PC workflow is entirely on this branch** — never `git checkout main` on a site PC unless you're explicitly using the office-PC toolchain here.
- **`settings.json` git-pull conflict avoidance** — operator's edits diverge from upstream every time we add a new key. Set `git update-index --skip-worktree webapp/isitec_app/settings.json webapp/isitec_api/settings.json` once after first clone and `git pull` will leave on-site values alone forever. New upstream keys appear in operator's file the first time they Save the Settings panel.

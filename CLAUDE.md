# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## TL;DR — start here

For anything operational, open **`start.md`** at the repo root — it's the one-page cheat-sheet for the task at hand:

- On **`main`**: dev setup, training, compression, deploy-to-site handoff.
- On **`deploy`**: site PC install, daily start, network lock-down, update, troubleshooting.

This file is the deep reference — architecture, trigger semantics, UDP protocol, data flows. Read `start.md` first; come back here when you need the "why".

---

## Project Overview

**IsiDetector** is a modular, config-driven instance segmentation pipeline for industrial parcel detection (cartons and polybags on conveyor belts). Two model families — **YOLO** (CNN) and **RF-DETR** (Transformer) — switchable via a single config line. Two peer web backends — **Flask** + **FastAPI** — share the same inference core. Deploys identically on **GPU** (NVIDIA CUDA) and **CPU-only** (Intel OpenVINO) hardware.

## Branch model — two branches, one source of truth

| Branch | Purpose | Contains |
|---|---|---|
| **`main`** | Canonical dev branch. Day-to-day work happens here. | The full project: training code, compression tool, docs, data prep, everything |
| **`deploy`** | Lean runtime subset shipped to site PCs. Derived from `main`. | Runtime-only — no `compression/`, no `mkdocs/`, no `isidet/src/training/`, no `isidet/scripts/` |

Site PCs clone the **`deploy`** branch. Office / dev workstations clone **`main`**. To refresh `deploy` after a runtime-relevant change lands on `main`:

```bash
git checkout deploy
git merge --no-commit main
# Replay the deletion set documented in the initial deploy commit (a6ead19):
git rm -r compression mkdocs isidet/src/training isidet/scripts
git rm isidet/a.py compress.sh deploy/_impl/compress.sh
# Keep only the runtime-affecting subset of Dockerfile edits (no scripts/ COPY)
git commit
git push origin deploy
```

If the runtime surface changed very little, a fast-forward merge won't work (history diverges); re-create `deploy` from `main` by replaying the deletion set — the list is stable and committed in `a6ead19`'s commit message.

## Repository layout (5 buckets, on `main`)

```
logistic/
├── isidet/        # ML core: src/, scripts/, configs/, data/, models/, runs/, logs/
├── webapp/        # isitec_app/ (Flask) + isitec_api/ (FastAPI)
├── compression/   # compression package (./compress.sh wrapper) — main only
├── mkdocs/        # mkdocs.yml + docs/ + built site/ — main only
└── deploy/        # Dockerfile*, docker-compose*.yml, _impl/*.sh, .env*
```

On `deploy`, `compression/` and `mkdocs/` aren't present; `isidet/` has only `{src/inference,src/shared,src/utils,src/preprocess,configs,models,rfdetr_service.py}` (no `src/training`, no `scripts`).

**Runtime conventions** that let the restructure be seamless:

- Thin-wrapper scripts at the repo root (`up.sh`, `run_start.sh`, `install.sh`, `net.sh`; plus `compress.sh` on `main` only) `exec` into `deploy/_impl/`. `./up.sh` at root continues to work from any CWD.
- `PYTHONPATH=isidet` is set inside every Docker image, every shell wrapper, and every webapp `sys.path` hack — so every `from src.X import Y` resolves without a single rewrite.
- Project-name pin: root `compose.yaml` has `name: deploy` so plain `docker compose down/logs/ps/exec` from the repo root targets the same containers that `./up.sh` (which `cd`s into `deploy/`) creates.

## Common Commands

### Deployment (Docker — production & daily ops)

```bash
./run_start.sh              # first time per host: Docker + NVIDIA toolkit + build images
./up.sh                     # daily starter; auto GPU vs CPU, waits for readiness, opens Chrome
./up.sh --force-cpu         # force CPU image (testing CPU path on a GPU host)
./up.sh --force-gpu         # require CUDA

docker compose down         # stop the stack (works from repo root via compose.yaml include)
docker compose logs -f web  # tail live logs
docker compose ps           # list running containers
```

Two-container architecture:
- **`web`** — Flask (`webapp/isitec_app/`) or FastAPI (`webapp/isitec_api/`) + ONNX Runtime, Ultralytics YOLO, OpenVINO, and (on GPU hosts) TensorRT. Ports 9501 TCP + 9502 UDP.
- **`rfdetr` sidecar** — Isolated PyTorch + `rfdetr` library for native `.pth` inference. Port 9510 internal-only. Only built on GPU hosts (gated by `profiles: [gpu]`); skipped on CPU-only.

Full walkthrough: `mkdocs/docs/deployment.md`.

### Training (office workstation / main branch)

```bash
python isidet/scripts/run_train.py                                       # model type in configs/train.yaml
python isidet/scripts/run_train.py --resume isidet/models/yolo/<date>/weights/last.pt
python isidet/scripts/run_train.py --config isidet/configs/train.yaml
```

CWD = repo root, `PYTHONPATH=isidet` (auto-set by every wrapper and Dockerfile; set manually with `export PYTHONPATH=$PWD/isidet` if you're bypassing them).

### Inference from CLI

```bash
python isidet/scripts/run_live.py  --weights isidet/models/yolo/<date>/weights/best.pt --source 0
python isidet/scripts/run_live.py  --weights isidet/models/rfdetr/<date>/checkpoint_best_ema.pth --source "rtsp://..."
python isidet/scripts/run_live.py  --weights isidet/models/rfdetr/<date>/inference_model.onnx --source "rtsp://..."
python isidet/scripts/run_infer.py --weights best.pt --source isidet/data/images
```

### Web platform (bare-metal dev, not Docker)

```bash
python webapp/isitec_app/app.py                                             # Flask on :9501
uvicorn isitec_api.app:app --host 0.0.0.0 --port 9501 --app-dir webapp      # FastAPI, same UI
```

FastAPI has **full parity** with Flask (same REST endpoints, same stream handler, same settings schema) **plus** `/ws/video` + `/ws/stats` WebSocket endpoints that replace polling. The Settings page dropdown groups models by file format via `<optgroup>` (TensorRT / PyTorch / OpenVINO / ONNX).

### Model Export & Compression (main branch only)

```bash
# Manual ONNX + OpenVINO + TensorRT export pipeline
python -m src.inference.export_engine --model-dir isidet/models/yolo/<date>/weights --format onnx openvino tensorrt

# Interactive / scriptable compression + format conversion
./compress.sh                                               # interactive menu
./compress.sh --model PATH --stage fp16|int8|int8_qdq|sim|openvino_fp16
./compress.sh --model foo.pt --convert pt-openvino          # full pt → onnx → sim → .xml pipeline

# ONNX sanity
python isidet/scripts/check_onnx.py
python isidet/scripts/rfdetr_onnx_checker.py
```

Full reference: `mkdocs/docs/compression.md`.

### Network lock-down (site PC only — `net.sh`)

```bash
./net.sh show        # current IP/gateway/DNS + UDP target (read-only, no sudo)
./net.sh manual      # bilingual (FR/EN) protocol sheet for the automaticien
./net.sh test        # 5 checks incl. live UDP egress probe (auto-escalates to sudo)
./net.sh apply       # freeze DHCP config as static NM (auto-escalates to sudo)
./net.sh revert      # back to DHCP (auto-escalates to sudo)
```

`test`/`apply`/`revert` re-exec themselves via `sudo -E` if not already root (preserves flags via `ORIG_ARGS`). Read-only `show`/`manual` run as the regular user. Not for WSL2 or Ubuntu Server — gracefully errors with "NetworkManager not installed" if that's the host.

### Docs

```bash
cd mkdocs && mkdocs serve    # dev server at http://127.0.0.1:8000
cd mkdocs && mkdocs build    # static site to mkdocs/site/
```

The built site is volume-mounted into the web container at `/opt/isitec/webapp/isitec_app/static/docs`, served at `http://localhost:9501/docs`.

### Dataset preparation

```bash
python isidet/scripts/prep_rfdetr_data.py    # YOLO polygon TXT → COCO JSON
python isidet/scripts/extract_frames.py      # MP4 → still frames
```

---

## Architecture

IsiDetector uses a **Config-Driven + Registry + Strategy** pattern across five layers.

### Layer 1: Configuration (`isidet/configs/`)

- `isidet/configs/train.yaml` — master switchboard: `model_type`, `nc`, `class_names`, `hooks`, inference settings.
- `isidet/configs/optimizers/yolo_optim.yaml` — YOLO hyperparams (200 epochs, AdamW, CosineAnnealing).
- `isidet/configs/optimizers/rfdetr_optim.yaml` — RF-DETR hyperparams (encoder lr=1e-5, head lr=1e-4, EMA).
- Merged at runtime in `isidet/scripts/run_train.py`.

### Layer 2: Registry (`isidet/src/shared/registry.py`)

- Three singleton registries: `TRAINERS`, `HOOKS`, `PREPROCESSORS`.
- Decorator-based: `@TRAINERS.register('yolo')` on a class, then `TRAINERS.get('yolo')` to retrieve it.
- Eliminates hardcoded `if model_type == 'yolo'` branches throughout the code.

### Layer 3: Training (`isidet/src/training/`)

- `BaseTrainer` (ABC) — enforces `build_model()` / `train()` / `evaluate()` / `export()` + hook lifecycle (`before_train`, `after_epoch`, `after_train`).
- `YOLOTrainer` — wraps Ultralytics, auto-generates `data.yaml`, bridges Ultralytics callbacks to hooks.
- `RFDETRTrainer` — wraps Roboflow RF-DETR (DINOv2 backbone), separate lr per parameter group.

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
- `isidet/src/utils/analytics_logger.py` — hourly CSV snapshots with daily rollover to `isidet/logs/`.
- `isidet/src/preprocess/clahe_engine.py` — SpecularGuard: CLAHE on LAB L-channel for industrial glare on polybags.
- `isidet/src/training/hooks/industrial_logger.py` — epoch-level stats hook (GPU mem, losses).

### Web Platform

Flask (`webapp/isitec_app/`) and FastAPI (`webapp/isitec_api/`) share `isidet/src/` via PYTHONPATH and serve identical UI.

- `app.py` — routes + session init (Flask has `/video_feed` MJPEG; FastAPI has `/ws/video` WebSocket + `/video_feed` MJPEG fallback).
- `stream_handler.py` — background inference thread, session state, locale (en/fr/de), UDP publisher.
- Endpoints: `POST /api/{start,stop,upload,language,dev-auth,dev-logout,udp,line,settings,belt_status}`, `GET /api/{stats,performance,chart?period=…,models,dev-check,settings,udp,line}`.
- FastAPI-only: `/ws/video` (binary JPEG stream), `/ws/stats` (500 ms JSON tick). `/docs` and `/docs/{subpath}` serve the built MkDocs site on both backends.

## Persistent hot-swap

`VisionEngine.swap_inferencer(new_inferencer)` replaces the model **in place** without tearing down session state. Preserved across a swap:

- `class_totals` (running counts keyed by class name)
- `counted_ids` (set of ByteTrack IDs already triggered)
- The ByteTrack tracker instance (IoU matching carries tracks across the swap)
- `LineZone` position + anchor
- `DailyLogger` (CSV keeps writing to the same file)

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

## Data flow

**Training:**
```
run_train.py → merge YAMLs → TRAINERS.get(model_type) → trainer.train()
  → call_hooks('before_train') → epoch loop → call_hooks('after_epoch')
  → trainer.evaluate() → trainer.export('onnx')
```

**Web inference:**
```
POST /api/start → StreamHandler (hot-swap path if same source)
  → _build_engine() picks inferencer by file extension
  → VisionEngine.swap_inferencer(new) — counts/tracker/line preserved
  → _inference_loop:
      per frame:
        engine.process_frame(frame) → (annotated, detections, new_events[])
        for event in new_events:
          latency_ns = UDPPublisher.publish(class, event_id)  → datagram → sorter (port 9502)
          monitor.track_udp_publish(latency_ns)                → histogram
          update last_detected + class_totals
      DailyLogger.update(class_totals) → hourly CSV snapshots
      latest_annotated JPEG → /video_feed (MJPEG) or /ws/video (WebSocket)
  → /api/stats: live counts  →  /api/chart: CSV-aggregated history
```

## Dataset formats

- **YOLO** (`isidet/data/isi_3k_dataset/`): `images/{train,val,test}/` + `labels/{train,val,test}/` (polygon TXT).
- **RF-DETR** (`isidet/data/rfdetr_dataset/`): `{train,valid,test}/images/` + `{train,valid,test}/_annotations.coco.json`.
- `isidet/scripts/prep_rfdetr_data.py` converts YOLO → COCO format.

## Classes

- `0: carton` (green in UI)
- `1: polybag` (orange in UI)

## Dependencies

Three `requirements*.txt` files, each scoped:

- **`requirements.txt`** (repo root) — dev / host deps (mkdocs, onnxsim, etc.) for the office workstation.
- **`deploy/requirements-deploy.txt`** — runtime deps baked into the web container (Flask/FastAPI, opencv, supervision, ultralytics, openvino, rfdetr).
- **`compression/requirements.txt`** — compression tool deps (rich, questionary, onnxconverter-common). `main` branch only.

The Docker images pull `torch` + `onnxruntime` separately (CPU or GPU wheels depending on the Dockerfile variant), then install `requirements-deploy.txt` on top. No single "install everything" recipe is maintained — always scope by use-case.

## Notes

- No formal test framework — reference commands in `testing.txt`.
- No linting config — no `.flake8`, `pyproject.toml` tool config, or `.pylintrc`.
- Pretrained weights tracked in repo: `isidet/models/pretrained/{yolo26n.pt,yolov8m-seg.pt}`. Everything else in `isidet/models/` and `isidet/runs/` is gitignored.
- **Adding a new model:** implement `BaseTrainer` / `BaseInferencer`, decorate with `@TRAINERS.register('name')`, set `model_type: name` in `isidet/configs/train.yaml`.
- **Line defaults:** `line_position = 0.5` (centred), `belt_direction = "left_to_right"`. Operators change them via the Tracking Line settings panel or `POST /api/line`.
- **ONNX session caching was removed** in favour of `preload_onnx()` which warms the CUDNN kernel cache at boot and discards the session. Cross-thread `session.run()` on a shared CUDA session triggered multi-second stream-sync stalls; per-thread rebuild reusing driver-level kernel cache avoids that at ~2 s cost per swap instead of 30–80 s.
- **Deployment marker** `deploy/.deployment.env` is written by `run_start.sh` (records `COMPOSE_MODE=gpu|cpu`) and consumed by `up.sh` to pick the right compose profile. Gitignored — per-host state.
- **Site PC workflow** is entirely on the `deploy` branch — never require a site PC to `git checkout main`. The deploy branch's own `CLAUDE.md` (note: `start.md` is the operational entry point on site) reflects the trimmed scope.

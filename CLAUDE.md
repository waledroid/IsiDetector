# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

!!! You're on the `deploy` branch.

This is the **runtime-only subset** of the project, meant for site PCs
that only need to run the inference stack. It does NOT contain:

- `compression/` — the office-workstation compression tool
- `mkdocs/` — documentation source and built site
- `isidet/src/training/` — the trainers, hooks, and BaseTrainer
- `isidet/scripts/` — training/eval/debug scripts (run_train, run_val,
  prep_rfdetr_data, extract_frames, onnx_checker, etc.)

If you need any of the above, switch to the `main` branch, which is
the full project source. Day-to-day development happens on `main`;
`deploy` is rebuilt from it whenever the runtime surface changes.

Everything below this note still applies — commands that reference
`isidet/scripts/` or `compression/` simply won't work on this branch;
use `main` for those.

## Project Overview

**IsiDetector** is a modular, config-driven instance segmentation pipeline for industrial parcel detection (cartons and polybags on conveyor belts). It supports two model families — YOLO (CNN) and RF-DETR (Transformer) — switchable via a single config line.

## Repository layout (5 buckets)

```
logistic/
├── isidet/       # ML core: src/, scripts/, configs/, data/, models/, runs/, logs/
├── webapp/       # isitec_app/ (Flask) + isitec_api/ (FastAPI)
├── compression/  # compression package (./compress.sh wrapper)
├── mkdocs/       # mkdocs.yml + docs/ + built site/
└── deploy/       # Dockerfile*, docker-compose*.yml, _impl/*.sh, .env*
```

Thin-wrapper scripts at the repo root (`up.sh`, `compress.sh`, `run_start.sh`, `install.sh`, `net.sh`) exec into `deploy/_impl/` — so muscle-memory `./up.sh` keeps working. Inside those scripts and inside Docker, `PYTHONPATH` points at `isidet/` so every `from src.X import Y` keeps resolving without being rewritten.

## Common Commands

### Deployment (Docker — production & daily ops)
```bash
# First time on a fresh host — installs Docker, NVIDIA toolkit, builds images.
# Writes .deployment.env recording gpu|cpu mode. Run once per machine.
./run_start.sh

# Every day — starts both containers, waits for ONNX preload marker, opens Chrome.
# Reads .deployment.env to pick GPU vs CPU compose profile.
./up.sh

# Restart after a code or config change
docker compose down && ./up.sh

# Tail live logs (Ctrl+C stops only the tail — containers keep running)
docker compose logs -f web
docker compose logs -f rfdetr
```

Two-container architecture:
- **`web`** — Flask app (`webapp/isitec_app/`) or FastAPI (`webapp/isitec_api/`), ONNX Runtime, Ultralytics YOLO, TensorRT (when `.engine` is used). Port 9501 (HTTP), 9502/udp (sorter telemetry).
- **`rfdetr` sidecar** — Isolated PyTorch + `rfdetr` library for native `.pth` inference. Port 9510 (internal only).

Full walkthrough (host prereqs, packaging for site delivery, troubleshooting) in `mkdocs/docs/deployment.md`.

### Training

Python commands below assume **CWD = repo root** and **`PYTHONPATH=isidet`** (the `deploy/_impl/*.sh` wrappers and the Dockerfiles set this automatically — outside those, `export PYTHONPATH=$PWD/isidet`).

```bash
# Train with current config (model type set in isidet/configs/train.yaml)
python isidet/scripts/run_train.py

# Resume YOLO from checkpoint
python isidet/scripts/run_train.py --resume isidet/models/yolo/<date>/weights/last.pt

# Custom config
python isidet/scripts/run_train.py --config isidet/configs/train.yaml
```

### Inference
```bash
# Live stream (RTSP or USB webcam)
python isidet/scripts/run_live.py --weights isidet/models/rfdetr/<date>/checkpoint_best_ema.pth --source "rtsp://..."
python isidet/scripts/run_live.py --weights isidet/models/yolo/<date>/weights/best.pt --source 0

# ONNX (optimized GPU inference)
python isidet/scripts/run_live.py --weights isidet/models/rfdetr/<date>/inference_model.onnx --source "rtsp://..."

# Batch inference
python isidet/scripts/run_infer.py --weights best.pt --source isidet/data/images
```

### Web Platform (dev / bare-metal)
```bash
# Flask backend (default)
python webapp/isitec_app/app.py               # http://0.0.0.0:9501

# FastAPI backend (alternative) — same UI, same endpoints, async via WebSocket
uvicorn isitec_api.app:app --host 0.0.0.0 --port 9501 --app-dir webapp

# Production deployment → use ./up.sh (see Deployment above).
```

### Model Export
```bash
# Export YOLO to ONNX
yolo export model=isidet/models/yolo/<date>/weights/best.pt format=onnx imgsz=640 opset=12 simplify=True nms=True

# Full pipeline: ONNX → simplified ONNX → OpenVINO (.xml) → TensorRT (.engine)
python -m src.inference.export_engine --model-dir isidet/models/yolo/<date>/weights \
    --format onnx openvino tensorrt

# Interactive / scriptable compression (fp16, int8, sim, openvino_fp16, format conversions)
./compress.sh                                          # interactive menu
./compress.sh --model PATH --stage fp16                # one-shot compression
./compress.sh --model PATH --convert pt-openvino       # full pt → onnx → sim → OV pipeline

# Validate ONNX model
python isidet/scripts/check_onnx.py
python isidet/scripts/rfdetr_onnx_checker.py
```

### Documentation
```bash
cd mkdocs && mkdocs serve    # Dev server at http://127.0.0.1:8000
cd mkdocs && mkdocs build    # Build static site to mkdocs/site/
```

### Dataset Preparation
```bash
# Convert YOLO polygon format → COCO (for RF-DETR)
python isidet/scripts/prep_rfdetr_data.py

# Extract frames from video
python isidet/scripts/extract_frames.py
```

## Architecture

IsiDetector uses a **Config-Driven + Registry + Strategy** pattern across five layers:

### Layer 1: Configuration (`isidet/configs/`)
- `isidet/configs/train.yaml` — master switchboard: `model_type`, `nc`, `class_names`, `hooks`, inference settings
- `isidet/configs/optimizers/yolo_optim.yaml` — YOLO hyperparams (200 epochs, AdamW, CosineAnnealing)
- `isidet/configs/optimizers/rfdetr_optim.yaml` — RF-DETR hyperparams (encoder lr=1e-5, head lr=1e-4, EMA)
- Both are merged at runtime in `isidet/scripts/run_train.py`

### Layer 2: Registry (`isidet/src/shared/registry.py`)
- Three singleton registries: `TRAINERS`, `HOOKS`, `PREPROCESSORS`
- Decorator-based: `@TRAINERS.register('yolo')` on a class, then `TRAINERS.get('yolo')` to retrieve it
- Eliminates hardcoded `if model_type == 'yolo'` branches throughout the code

### Layer 3: Training (`isidet/src/training/`)
- `BaseTrainer` (ABC) — enforces `build_model()`, `train()`, `evaluate()`, `export()` + hook lifecycle (`before_train`, `after_epoch`, `after_train`)
- `YOLOTrainer` — wraps Ultralytics, auto-generates `data.yaml`, bridges Ultralytics callbacks to hooks
- `RFDETRTrainer` — wraps Roboflow RF-DETR (DINOv2 backbone), separate lr per parameter group

### Layer 4: Inference (`isidet/src/inference/`)
Five backends, selected automatically by file extension in `StreamHandler._build_engine()`:

| Class | Extension | Device | Notes |
|---|---|---|---|
| `YOLOInferencer` | `.pt` | CUDA / CPU | Ultralytics native |
| `RFDETRInferencer` / `RemoteRFDETRInferencer` | `.pth` | CUDA | Native direct, or HTTP to rfdetr sidecar when running inside Docker |
| `OptimizedONNXInferencer` | `.onnx` | CUDA / CPU | Auto-detects YOLO vs RF-DETR family from output names/shapes. Handles both post-NMS and pre-NMS YOLO, DETR class-index offset, CUDNN preload |
| `OpenVINOInferencer` | `.xml` | Intel CPU | Fastest CPU backend |
| `TensorRTInferencer` | `.engine` | NVIDIA GPU only | Per-host compiled engines, 1.5–3× faster than ONNX-CUDA |

`BaseInferencer` holds shared preprocessing helpers. Auto-discovery of default weights walks `isidet/models/yolo/**/weights/*` and `isidet/models/rfdetr/**/*` with per-device extension priority — GPU hosts prefer native or TensorRT, CPU hosts prefer OpenVINO or ONNX.

### Layer 5: Support Modules
- **`isidet/src/shared/vision_engine.py`** — unified orchestrator: ByteTrack tracking, line-crossing counting, annotation, telemetry
- **`isidet/src/utils/analytics_logger.py`** — hourly CSV snapshots with daily rollover to `isidet/logs/`
- **`isidet/src/preprocess/clahe_engine.py`** — SpecularGuard: CLAHE on LAB L-channel for industrial glare on polybags
- **`isidet/src/training/hooks/industrial_logger.py`** — epoch-level stats hook (GPU mem, losses)

### Web Platform (`webapp/isitec_app/` Flask, `webapp/isitec_api/` FastAPI)
Two peer backends share `isidet/src/` (via PYTHONPATH) and serve identical UI.

- `webapp/isitec_app/app.py` — Flask routes on port 9501; MJPEG stream at `/video_feed`; REST API at `/api/*`.
- `webapp/isitec_api/app.py` — FastAPI equivalent; pushes stats over `/ws/stats` WebSocket (500 ms) instead of polling.
- Both use `stream_handler.py` — background inference thread, session state, locale switching (en/fr/de), UDP publisher.
- Key endpoints: `POST /api/start`, `POST /api/stop`, `POST /api/upload`, `GET /api/stats`, `GET /api/chart?period=24h|7d|30d|live`, `GET|POST /api/udp`, `GET|POST /api/line`, `GET|POST /api/settings`.

### Persistent Hot-Swap
`VisionEngine.swap_inferencer(new_inferencer)` replaces the model **in place** without tearing down session state. Across a model swap, these persist:
- `class_totals` (running counts keyed by class name)
- `counted_ids` (set of ByteTrack IDs already triggered)
- the ByteTrack tracker instance (IoU matching carries tracks across the swap)
- `LineZone` position + anchor
- `DailyLogger` (CSV keeps writing to the same file)

Only the inferencer reference and palette-indexed annotators (mask/box/label) are rebuilt so per-class colours reflect the new model's class-ID convention. Swap latency is ~2 s on GPU (CUDNN kernel cache is primed via `preload_onnx()` at container boot).

### Trigger Semantics (sorter-first)
Line-crossing fires on the **leading edge** of the bbox — the side that enters the line zone first given belt direction. This maximises the sorter gate's reaction window. Mapping lives in `isidet/src/shared/vision_engine.py _ANCHOR_MAP`:

| Orientation | Belt direction | Anchor (`sv.Position`) |
|---|---|---|
| vertical | left_to_right | `CENTER_RIGHT` |
| vertical | right_to_left | `CENTER_LEFT` |
| horizontal | top_to_bottom | `BOTTOM_CENTER` |
| horizontal | bottom_to_top | `TOP_CENTER` |

`VisionEngine.process_frame()` returns a **list** of events (previously a single event); `_inference_loop` iterates and publishes one UDP datagram per crossing. Two close-together objects in the same frame both trigger their sort gates.

**Class-ID conventions** (intentional divergence for visual swap confirmation):
- YOLO emits `class_id ∈ {0, 1}` → palette slots `[0, 1]`
- RF-DETR emits `class_id ∈ {1, 2}` (DETR reserves index 0 for background) → palette slots `[1, 2]`

Operators see colours flip on YOLO ↔ RF-DETR swap as confirmation the swap took effect. Counts are keyed by class **name** (string), so the count dict is unaffected.

### UDP Sorting Broadcast (`webapp/isitec_app/stream_handler.py` — `UDPPublisher`)
On every line-crossing event, a ~60-byte JSON datagram is fired to the sorting machine controller. **One datagram per crossing** (including multiple-per-frame when objects are close together).
```json
{"class": "carton", "id": 42, "ts": "2026-03-31T14:23:45.312847"}
```
- `id` is the ByteTrack tracker ID — lets the sorter dedupe if the network duplicates the datagram. Optional; older consumers that only read `class` keep working.
- Transport: `socket.SOCK_DGRAM` — single socket created once at stream start, reused per event, no queue
- Timestamp: microsecond ISO format (`.isoformat()`)
- Default target: `127.0.0.1:9502` (controller on same machine)
- **Configuration priority** (highest wins):
  1. Runtime API: `POST /api/udp {"host": "...", "port": ...}` — retargets live, no restart needed
  2. Environment variable: `UDP_HOST`, `UDP_PORT`
  3. `isidet/configs/train.yaml` → `inference.udp.host / port`
  4. Hardcoded default: `127.0.0.1:9502`
- Docker: `-p 9502:9502` to expose port, or `-e UDP_HOST=192.168.1.50` for remote controller

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

### Data Flow (Training)
```
run_train.py → merge YAMLs → TRAINERS.get(model_type) → trainer.train()
  → call_hooks('before_train') → epoch loop → call_hooks('after_epoch')
  → trainer.evaluate() → trainer.export('onnx')
```

### Data Flow (Web Inference)
```
POST /api/start → StreamHandler (hot-swap path if same source)
  → _build_engine() picks inferencer by file extension
  → VisionEngine.swap_inferencer(new) — counts/tracker/line preserved
  → _inference_loop:
      per frame:
        engine.process_frame(frame) → (annotated, detections, new_events[])
        for event in new_events:
          UDPPublisher.publish(class, event_id) → datagram → sorter (port 9502)
          update last_detected + class_totals
      DailyLogger.update(class_totals) → hourly CSV snapshots
      latest_annotated JPEG → /video_feed (MJPEG) or /ws/stats (FastAPI)
  → /api/stats: live counts  →  /api/chart: CSV-aggregated history
```

## Dataset Formats

- **YOLO** (`isidet/data/isi_3k_dataset/`): `images/{train,val,test}/` + `labels/{train,val,test}/` (polygon TXT)
- **RF-DETR** (`isidet/data/rfdetr_dataset/`): `{train,valid,test}/images/` + `{train,valid,test}/_annotations.coco.json`
- Use `isidet/scripts/prep_rfdetr_data.py` to convert from YOLO to COCO format

## Dependencies

No root-level `requirements.txt`. Install per use-case:
```bash
# Training (YOLO)
pip install ultralytics supervision opencv-python pyyaml

# Training (RF-DETR)
pip install rfdetr supervision opencv-python pyyaml pandas matplotlib

# Web platform
pip install Flask>=3.0.0 opencv-python supervision numpy
```

## Classes
- `0: carton` (green in UI)
- `1: polybag` (orange in UI)

## Notes
- No formal test framework — reference commands in `testing.txt`
- No linting config — no `.flake8`, `pyproject.toml` tool config, or `.pylintrc`
- Pretrained weights committed to repo: `isidet/models/pretrained/yolo26n.pt`, `isidet/models/pretrained/yolov8m-seg.pt`
- `isidet/models/` holds trained run outputs; `isidet/runs/` holds Ultralytics YOLO run artifacts
- To add a new model: implement `BaseTrainer`/`BaseInferencer`, decorate with `@TRAINERS.register('name')`, and set `model_type: name` in `isidet/configs/train.yaml`
- **Line defaults**: `line_position = 0.5` (centred), `belt_direction = "left_to_right"`. Operators change them via the Tracking Line settings panel or `POST /api/line {"belt_direction": "..."}`.
- **ONNX session caching was removed** in favour of `preload_onnx()` which warms the CUDNN kernel cache at boot and discards the session. Cross-thread `session.run()` on a shared CUDA session triggered multi-second stream-sync stalls; per-thread rebuild reusing driver-level kernel cache avoids that at ~2 s cost per swap instead of 30–80 s.
- **Deployment**: `./run_start.sh` bootstraps a fresh host (Docker + nvidia-container-toolkit + builds images), `./up.sh` is the daily starter (picks GPU/CPU compose profile from `deploy/.deployment.env`, waits for preload marker, opens Chrome). Full walkthrough in `mkdocs/docs/deployment.md`.

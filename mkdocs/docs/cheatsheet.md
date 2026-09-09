# IsiDetector Cheat-Sheet

Config-driven instance-segmentation pipeline: counts **cartons** and **polybags** crossing a line on a conveyor, fires one **UDP datagram per crossing** to the sorting machine. Two model families (YOLO, RF-DETR), five inference backends, two peer web backends (Flask / FastAPI), one Docker stack for both GPU and CPU-only hosts.

---

## 1. Modularity pattern

**Why:** swap models/hooks/preprocessors from YAML — no pipeline code changes.

One generic `Registry` class (`src/shared/registry.py`), three singletons:

| Registry | Members | Selected by |
|---|---|---|
| `TRAINERS` | `YOLOTrainer`, `RFDETRTrainer` | `train.yaml → model_type` |
| `HOOKS` | `IndustrialLogger` | `train.yaml → hooks:` |
| `PREPROCESSORS` | `SpecularGuard` (CLAHE) | webapp preprocess chain |

- Register with a decorator: `@TRAINERS.register('yolo')` · retrieve with `TRAINERS.get(name)`.
- `BaseTrainer` (ABC): `build_model / train / evaluate / export` + hook stages `before_train → before_epoch → after_epoch → after_train`.
- `BaseInferencer` (ABC): shared preprocessing/rescaling; 6 concrete backends picked by **file extension** (§5) — the one deliberate `if/elif` left.
- **Add a model:** subclass both bases → decorate → set `model_type:` in `train.yaml`.

---

## 2. System tree

```
logistic/
├── up.sh · run_start.sh · net.sh · compress.sh   # wrappers → deploy/_impl/
├── compose.yaml                                  # includes deploy compose, project name "deploy"
├── isidet/
│   ├── src/shared/       # registry, vision_engine (orchestrator), crossing, dedup_gate
│   ├── src/training/     # base_trainer, trainers/{yolo,rfdetr}, hooks/
│   ├── src/inference/    # base + 6 backends + export_engine
│   ├── src/preprocess/   # clahe_engine (SpecularGuard)
│   ├── src/utils/        # event_logger (per-crossing CSV, 30-day retention)
│   ├── scripts/          # run_train, run_live, run_infer, prep_rfdetr_data, extract_frames
│   ├── configs/          # train.yaml, optimizers/, inference/{common,cpu,gpu}.yaml
│   └── rfdetr_service.py # GPU sidecar HTTP service
├── webapp/isitec_app/    # Flask   ─┐ same UI + REST on :9501
├── webapp/isitec_api/    # FastAPI ─┘ + WebSockets /ws/video /ws/stats
├── compression/          # office-only CLI (stages/, rich+questionary menu)
├── deploy/               # Dockerfile (GPU) · Dockerfile.cpu · Dockerfile.rfdetr · compose files
└── mkdocs/               # docs source + built site (served at :9501/docs)
```

**Docker stack:** `web` (:9501 UI, :9502 UDP out) · `rfdetr` sidecar (:9510, GPU hosts only) · `docs` (:9505).

---

## 3. Datasets & classes

**Classes:** `0 = carton` (green) · `1 = polybag` (orange). COCO/RF-DETR shifts +1 (0 = background) — operators see colours flip on model swap, counts (keyed by name) don't change.

| Dataset | Format | Train / Val | Size | Status |
|---|---|---|---|---|
| `dataset_v2` | YOLO polygon-seg | 4 968 / 518 | 1.8 GB | **Active** |
| `universal_dataset` | YOLO + COCO json | 2 899 / 243 | 3.7 GB | RF-DETR source |
| `rfdetr_dataset` | COCO | 2 898 / 242 | 18 MB | symlinks **broken** (stale paths) |
| `isi_3k_dataset` | LabelMe, unsplit | 1 236 imgs | 756 MB | raw pool |

- Label = `class_id` + normalized polygon vertices, one line per instance.
- `prep_rfdetr_data.py`: YOLO → COCO layout (symlinks images, renames val→valid). `extract_frames.py`: video → frames.

---

## 4. Web app

**Ingestion** — `POST /api/start` with a polymorphic `source`: webcam index, `rtsp://` URL, uploaded file path, or empty = saved Site-Camera URL. A reader thread keeps only the freshest frame; RTSP auto-reconnects.

**Endpoints** (Flask = FastAPI parity; 🔒 = dev-token):

| Group | Endpoints |
|---|---|
| Stream | `POST /api/start` `stop` `upload` · `GET /video_feed` (MJPEG) |
| Live data | `GET /api/stats` `mode` `snapshot` `models` |
| History | `GET /api/chart?period=live\|24h\|7d\|30d` · `report` · `events/export` (CSV) |
| Config 🔒 | `GET/POST /api/settings` `udp` `line` · `POST belt_status` · `GET performance` |
| FastAPI only | `WS /ws/video` (JPEG ~30 fps) · `WS /ws/stats` (500 ms) · `/swagger` |
| Docs | `GET /docs` — this site |

**Key util — `StreamHandler`** (one per backend): picks the inference backend, runs the inference loop (frame → ROI → resize → CLAHE → engine → UDP publish → JPEG out), owns session state.

**Hot-swap:** Start with same source + new model = model replaced in place (~2 s) — counts, tracker, line, event log all preserved.

---

## 5. Detection

Backend chosen by weight file extension:

| Ext | Class | Library | Notes |
|---|---|---|---|
| `.pt` | `YOLOInferencer` | ultralytics | CUDA/CPU |
| `.pth` | `RFDETRInferencer` / `Remote…` | rfdetr / HTTP sidecar | GPU; remote when in Docker |
| `.onnx` | `OptimizedONNXInferencer` | onnxruntime | auto-detects YOLO vs DETR, INT8 aware |
| `.xml` | `OpenVINOInferencer` | openvino | fastest CPU; **refuses RF-DETR** (broken op translation) |
| `.engine` | `TensorRTInferencer` | tensorrt + pycuda | fastest GPU, per-host compiled |

**`VisionEngine.process_frame`** — the counting core:
`predict → ByteTrack → LineZone crossing → DedupGate → count + CSV event log → annotate` → returns a **list** of events (two parcels in one frame = two triggers).

**Leading-edge trigger** — crossing fires on the bbox side entering first (max sorter reaction time):

| Orientation | Direction | Anchor |
|---|---|---|
| vertical | left→right / right→left | `CENTER_RIGHT` / `CENTER_LEFT` |
| horizontal | top→bottom / bottom→top | `BOTTOM_CENTER` / `TOP_CENTER` |

**SpecularGuard:** CLAHE on the LAB L-channel — kills polybag glare, keeps colour.

---

## 6. Training, deployment & UDP

### Training

`run_train.py`: merge `train.yaml` + optimizer YAML → registry lookup → `train → evaluate → export`.

| | YOLO | RF-DETR |
|---|---|---|
| Library | ultralytics (`yolo26*-seg.pt`) | rfdetr (DINOv2 backbone) |
| Recipe | 200 ep · AdamW 5e-4 · cosine | 100 ep · dual LR (head 1e-4, encoder 1e-5) · EMA · grad-accum 8 |
| Extras | auto-writes `data.yaml` | CPU-offloaded mask upsampling (OOM guard) |

**Export chain:** `.pt/.pth → ONNX → onnxsim → OpenVINO .xml → TensorRT .engine`.

### Deployment

| Image | Base | Backends |
|---|---|---|
| web GPU | cuda 12.8 + torch cu128 | all five |
| web CPU | python:3.11-slim | OpenVINO, ONNX, .pt |
| rfdetr sidecar | cuda 12.8, pinned rfdetr | native .pth (isolated deps) |

- `run_start.sh` — one-time bootstrap: installs Docker (+ NVIDIA toolkit), builds, writes `COMPOSE_MODE=gpu|cpu` marker.
- `up.sh` — daily: reads marker, auto-falls back to CPU if driver missing, waits for readiness, opens browser.

### UDP sort trigger

- Datagram: `{"class": "carton", "seq": 17, "id": 42, "ts": "…µs ISO…"}` (~70 B). `seq` = gap-free loss detector; `id` = tracker id for dedupe.
- One socket created once, reused; retargetable live via `/api/udp` — priority: API > env `UDP_HOST/PORT` > `train.yaml` > `127.0.0.1:9502`.
- Per-send latency → p50/p95/p99 histogram in `/api/performance` (green < 500 µs).
- ONNX CUDNN cache pre-warmed at boot → hot-swap ~2 s instead of 30–80 s.

---

## 7. Compression (office GPU only)

`./compress.sh` — interactive menu, or `--model PATH --stage NAME` / `--convert NAME`.

| Stage | Effect | Library |
|---|---|---|
| `fp16` | ½ size, <0.5 % mAP loss | onnxconverter-common |
| `int8` | ~75 % smaller, synthetic calibration | onnxruntime quantize_static (QDQ) |
| `int8qdq` | INT8 with real calibration images | onnxruntime quantize_static |
| `sim` | graph simplification | onnxsim |
| `openvino_fp16` | FP16 OpenVINO IR | openvino |

**Conversions:** `pt-onnx` · `onnx-sim` · `onnx-openvino` · `openvino-fp16` · **`pt-openvino`** = full `.pt → .onnx → sim → .xml` chain.

---

## 8. How to use

**Dev (bare metal, `main`)** — repo root, `PYTHONPATH=isidet`:

```bash
python isidet/scripts/run_train.py                          # or --resume …/last.pt
python isidet/scripts/run_live.py --weights best.pt --source 0     # webcam | rtsp | file
python webapp/isitec_app/app.py                             # Flask :9501
uvicorn isitec_api.app:app --port 9501 --app-dir webapp     # FastAPI :9501
```

**Site (Docker)**

```bash
./run_start.sh                    # once per host
./up.sh                           # daily start → http://localhost:9501
docker compose down && ./up.sh    # restart after change
docker compose logs -f web
./net.sh show|test|apply|revert   # network lock-down + UDP probe
```

**Operator loop** — `./up.sh` → pick Mode (1 YOLO · 2 RF-DETR/GPU) + source → **Start**. Same source + new model = hot-swap. Charts: Live/24h/7d/30d; CSV export in Analytics.

**Export & compress (office)**

```bash
python -m src.inference.export_engine --model-dir …/weights --format onnx openvino tensorrt
./compress.sh --model foo.pt --convert pt-openvino
```

**Troubleshoot**

```bash
docker compose logs web | tail -30            # startup hangs
curl -s localhost:9501/api/models             # empty model dropdown
curl -s localhost:9501/api/udp && ./net.sh test   # no sorter triggers
nvidia-smi && ./run_start.sh                  # GPU fell back to CPU
docker compose build --no-cache && ./up.sh    # stale image
```

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**IsiDetector** is a modular, config-driven instance segmentation pipeline for industrial parcel detection (cartons and polybags on conveyor belts). It supports two model families — YOLO (CNN) and RF-DETR (Transformer) — switchable via a single config line.

## Common Commands

### Training
```bash
# Train with current config (model type set in configs/train.yaml)
python scripts/run_train.py

# Resume YOLO from checkpoint
python scripts/run_train.py --resume models/yolo/<date>/weights/last.pt

# Custom config
python scripts/run_train.py --config configs/train.yaml
```

### Inference
```bash
# Live stream (RTSP or USB webcam)
python scripts/run_live.py --weights models/rfdetr/<date>/checkpoint_best_ema.pth --source "rtsp://..."
python scripts/run_live.py --weights models/yolo/<date>/weights/best.pt --source 0

# ONNX (optimized GPU inference)
python scripts/run_live.py --weights models/rfdetr/<date>/inference_model.onnx --source "rtsp://..."

# Batch inference
python scripts/run_infer.py --weights best.pt --source data/images
```

### Web Platform
```bash
# Start Flask app on http://0.0.0.0:9501
python isitec_app/app.py

# Docker
docker build -t isitec-visionai -f isitec_app/Dockerfile .
docker run -p 9501:9501 isitec-visionai
```

### Model Export
```bash
# Export YOLO to ONNX
yolo export model=models/yolo/<date>/weights/best.pt format=onnx imgsz=640 opset=12 simplify=True nms=True

# Validate ONNX model
python scripts/check_onnx.py
python rfdetr_onnx_checker.py
```

### Documentation
```bash
mkdocs serve    # Dev server at http://127.0.0.1:8000
mkdocs build    # Build static site to ./site/
```

### Dataset Preparation
```bash
# Convert YOLO polygon format → COCO (for RF-DETR)
python scripts/prep_rfdetr_data.py

# Extract frames from video
python scripts/extract_frames.py
```

## Architecture

IsiDetector uses a **Config-Driven + Registry + Strategy** pattern across five layers:

### Layer 1: Configuration (`configs/`)
- `configs/train.yaml` — master switchboard: `model_type`, `nc`, `class_names`, `hooks`, inference settings
- `configs/optimizers/yolo_optim.yaml` — YOLO hyperparams (200 epochs, AdamW, CosineAnnealing)
- `configs/optimizers/rfdetr_optim.yaml` — RF-DETR hyperparams (encoder lr=1e-5, head lr=1e-4, EMA)
- Both are merged at runtime in `scripts/run_train.py`

### Layer 2: Registry (`src/shared/registry.py`)
- Three singleton registries: `TRAINERS`, `HOOKS`, `PREPROCESSORS`
- Decorator-based: `@TRAINERS.register('yolo')` on a class, then `TRAINERS.get('yolo')` to retrieve it
- Eliminates hardcoded `if model_type == 'yolo'` branches throughout the code

### Layer 3: Training (`src/training/`)
- `BaseTrainer` (ABC) — enforces `build_model()`, `train()`, `evaluate()`, `export()` + hook lifecycle (`before_train`, `after_epoch`, `after_train`)
- `YOLOTrainer` — wraps Ultralytics, auto-generates `data.yaml`, bridges Ultralytics callbacks to hooks
- `RFDETRTrainer` — wraps Roboflow RF-DETR (DINOv2 backbone), separate lr per parameter group

### Layer 4: Inference (`src/inference/`)
- `BaseInferencer` — shared aspect-ratio-aware preprocessing and rescaling
- `YOLOInferencer`, `RFDETRInferencer` — model-specific wrappers
- `OptimizedONNXInferencer` — ONNX Runtime with CUDA/GPU execution providers, dynamic batching, stats tracking

### Layer 5: Support Modules
- **`src/shared/vision_engine.py`** — unified orchestrator: ByteTrack tracking, line-crossing counting, annotation, telemetry
- **`src/utils/analytics_logger.py`** — hourly CSV snapshots with daily rollover to `logs/`
- **`src/preprocess/clahe_engine.py`** — SpecularGuard: CLAHE on LAB L-channel for industrial glare on polybags
- **`src/training/hooks/industrial_logger.py`** — epoch-level stats hook (GPU mem, losses)

### Web Platform (`isitec_app/`)
- `app.py` — Flask routes on port 9501; MJPEG stream at `/video_feed`; REST API at `/api/*`
- `stream_handler.py` — background inference thread, statistics management, locale switching (en/fr/de)
- Key endpoints: `POST /api/start`, `POST /api/stop`, `POST /api/upload`, `GET /api/stats`, `GET /api/chart?period=24h|7d|30d|live`

### Data Flow (Training)
```
run_train.py → merge YAMLs → TRAINERS.get(model_type) → trainer.train()
  → call_hooks('before_train') → epoch loop → call_hooks('after_epoch')
  → trainer.evaluate() → trainer.export('onnx')
```

### Data Flow (Web Inference)
```
Flask /api/start → StreamHandler → VisionEngine(inferencer, config)
  → ByteTrack (ID continuity) → line-crossing count → DailyLogger (CSV)
  → /video_feed: annotated MJPEG frames
  → /api/stats: live counts  →  /api/chart: CSV-aggregated history
```

## Dataset Formats

- **YOLO** (`data/isi_3k_dataset/`): `images/{train,val,test}/` + `labels/{train,val,test}/` (polygon TXT)
- **RF-DETR** (`data/rfdetr_dataset/`): `{train,valid,test}/images/` + `{train,valid,test}/_annotations.coco.json`
- Use `scripts/prep_rfdetr_data.py` to convert from YOLO to COCO format

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
- Pretrained weights committed to repo: `yolo26n.pt`, `yolov8m-seg.pt`, `rf-detr-seg-medium.pt`
- `models/` holds trained run outputs; `runs/` holds Ultralytics YOLO run artifacts
- To add a new model: implement `BaseTrainer`/`BaseInferencer`, decorate with `@TRAINERS.register('name')`, and set `model_type: name` in `train.yaml`

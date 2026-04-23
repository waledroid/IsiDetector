---
hide:
  - navigation
---

<div class="hero" markdown>

# 📦 IsiDetector

<p class="hero-subtitle">
  A modular, config-driven instance segmentation pipeline for industrial parcel detection.
  <br/>
  Switch between <strong>YOLOv26-seg</strong> and <strong>RF-DETR-Seg</strong> with one line of YAML.
</p>

[Get Started :material-arrow-right:](getting-started.md){ .md-button .md-button--primary }
[Architecture :material-map:](architecture/overview.md){ .md-button }

</div>

---

## Why IsiDetector?

IsiDetector is built for **logistics and warehouse** environments where you need to detect and segment parcels — specifically **cartons** and **polybags** — on conveyor belts under challenging industrial lighting.

<div class="grid-container" markdown>
<div class="card" markdown>

### 🧠 Two Architectures, One Interface

Train with a CNN-based **YOLOv26-seg** for speed, or a Transformer-based **RF-DETR-Seg** (DINOv2 backbone) for global context. Both share the exact same API.

</div>
<div class="card" markdown>

### ⚙️ Config-Driven Everything

No code changes needed. Your model architecture, optimizer, learning rate schedule, augmentation strategy, and early stopping rules are all in YAML.

</div>
<div class="card" markdown>

### 🔌 Plugin Hook System

Attach loggers, alerters, or metric trackers by name in your config. Write a class, register it, add one line to YAML — done.

</div>
<div class="card" markdown>

### 🏭 Industrial Preprocessing

Built-in CLAHE-based `SpecularGuard` handles polybag glare and deep shadows in LAB colour space without distorting the actual parcel colours.

</div>
</div>

---

## The Pipeline at a Glance

<div class="pipeline-flow">
<span class="step">📄 train.yaml</span>
<span class="arrow">→</span>
<span class="step">🚀 run_train.py</span>
<span class="arrow">→</span>
<span class="step">🔍 Registry Lookup</span>
<span class="arrow">→</span>
<span class="step">🔨 Trainer.train()</span>
<span class="arrow">→</span>
<span class="step">📊 Trainer.evaluate()</span>
<span class="arrow">→</span>
<span class="step">📦 Trainer.export()</span>
</div>

---

## Repository Structure — Five Buckets

The repo is split into five top-level buckets, one per concern, with a handful of thin-wrapper `.sh` scripts at the root so `./up.sh`, `./compress.sh` etc. keep working. Inside every runtime entry point, `PYTHONPATH=isidet/` keeps every `from src.…` import resolving without a single code-level rewrite.

```text
logistic/
├── isidet/                    # 🏗️  ML core — everything training/inference needs
│   ├── src/
│   │   ├── inference/         #   5 backends (YOLO, RF-DETR, ONNX, OpenVINO, TensorRT)
│   │   ├── training/          #   BaseTrainer + YOLO / RF-DETR trainers + hooks
│   │   ├── shared/            #   registry.py, vision_engine.py
│   │   ├── preprocess/        #   CLAHE SpecularGuard
│   │   └── utils/             #   analytics_logger.py (daily CSV rollover)
│   ├── scripts/               #   run_train.py / run_live.py / run_infer.py / …
│   ├── configs/               #   train.yaml + optimizers/*.yaml
│   ├── data/                  #   Training & validation datasets (gitignored)
│   ├── models/                #   Trained weights + pretrained/ baseline .pt
│   ├── runs/                  #   Ultralytics run artefacts
│   └── logs/                  #   Hourly CSV analytics
│
├── webapp/                    # 🌐 Web front-ends — two peers sharing isidet/src
│   ├── isitec_app/            #   Flask (canonical reference)
│   └── isitec_api/            #   FastAPI (WebSocket streaming, feature-parity)
│
├── compression/               # 🗜️  Interactive model-compression tool
│   │   (python -m compression, or ./compress.sh)
│   ├── cli.py                 #   Interactive questionary menu
│   ├── __main__.py            #   argparse front-door for scripted runs
│   ├── convert_ops.py         #   pt→onnx, onnx→sim, onnx→openvino, OV FP16
│   └── stages/                #   fp16, int8, int8_qdq, sim compression stages
│
├── mkdocs/                    # 📑 Documentation — this site
│   ├── mkdocs.yml
│   ├── docs/                  #   Markdown source (you're reading it)
│   └── site/                  #   Built static HTML (volume-mounted at /docs)
│
├── deploy/                    # 🚢 Deployment — Docker + host scripts
│   ├── Dockerfile             #   GPU image (nvidia/cuda:12.8 base)
│   ├── Dockerfile.cpu         #   CPU-only image (python:3.11-slim + OpenVINO)
│   ├── Dockerfile.rfdetr      #   Isolated RF-DETR sidecar
│   ├── docker-compose.yml     #   Base stack (web + rfdetr-sidecar)
│   ├── docker-compose.{gpu,cpu}.yml   #  Overlays for GPU/CPU profiles
│   └── _impl/                 #   Real shell-script bodies (see table below)
│
└── (thin wrappers at root)
    up.sh, compress.sh, run_start.sh, install.sh, net.sh
```

### Shell scripts at a glance

Every `.sh` at the repo root is a 3-line wrapper that `exec`s into `deploy/_impl/`. The real logic and what each script does:

| Script | What it does | When to run |
|---|---|---|
| `install.sh` | Clones the repo onto a blank Ubuntu PC, makes scripts executable, optionally hands off to `run_start.sh`. | **Once**, on a fresh site PC that doesn't yet have the repo |
| `run_start.sh` | Host bootstrap — installs Docker Engine + (if GPU) NVIDIA Container Toolkit, builds both images, writes `deploy/.deployment.env` recording GPU/CPU mode. | **Once per machine** after `install.sh` |
| `up.sh` | Daily starter — picks the compose profile from `.deployment.env`, `docker compose up -d`, waits for the ONNX preload marker, opens Chrome on `http://localhost:9501`. | **Every day** / after a `docker compose down` |
| `compress.sh` | Opens the interactive compression menu (or runs one-shot with `--model … --stage/--convert …`). Accepts any `.pt` / `.pth` / `.onnx` / `.xml` path, inside or outside the repo. | Whenever you want to shrink, convert, or benchmark a model (office workstation, not site PC) |
| `net.sh` | Freezes the site PC's DHCP-issued IP / gateway / DNS into a static NetworkManager config so the automate's firewall whitelist doesn't drift. Also prints a ready-to-email mini-manual for the automaticien describing the UDP sort-trigger protocol. | **Once per site**, after `up.sh` is stable and the sorter is on the LAN |


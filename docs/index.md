---
hide:
  - navigation
---

<div class="hero" markdown>

# 📦 IsiDetector

<p class="hero-subtitle">
  A modular, config-driven instance segmentation pipeline for industrial parcel detection.
  <br/>
  Switch between <strong>YOLOv12-seg</strong> and <strong>RF-DETR-Seg</strong> with one line of YAML.
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

Train with a CNN-based **YOLOv12-seg** for speed, or a Transformer-based **RF-DETR-Seg** (DINOv2 backbone) for global context. Both share the exact same API.

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

## Project Structure

```text
logistic/
├── isitec_app/                       # 🌐 VisionAI Platform (Flask)
│   ├── app.py                        # Backend entrypoint
│   ├── stream_handler.py             # Async inference management
│   ├── templates/                    # Material UI index
│   └── static/                       # CSS/JS & Assets
├── configs/                          # ⚙️ YAML configurations
│   ├── train.yaml                    # Master switchboard
│   └── optimizers/                   # Hyperparameter presets
├── src/                              # 🏗️ Core Engine
│   ├── shared/registry.py            # Global module registry
│   ├── training/                     # Training logic
│   │   ├── trainers/                 # YOLOv12 & RF-DETR implementations
│   │   └── hooks/                    # Logging & Metrics system
│   └── inference/                    # High-speed ONNX & Torch engines
├── models/                           # 🧠 Pretrained & Exported weights
├── data/                             # 📊 Training & Validation sets
├── docs/                             # 📑 Documentation source
└── site/                             # 🌍 Built MkDocs static site
```

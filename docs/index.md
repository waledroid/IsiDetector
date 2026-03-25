---
hide:
  - navigation
---

<div class="hero" markdown>

# 📦 IsiDetector

<p class="hero-subtitle">
  A modular, config-driven instance segmentation pipeline for industrial parcel detection.
  <br/>
  Switch between <strong>YOLOv8-Seg</strong> and <strong>RF-DETR-Seg</strong> with one line of YAML.
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

Train with a CNN-based **YOLOv8-Seg** for speed, or a Transformer-based **RF-DETR-Seg** (DINOv2 backbone) for global context. Both share the exact same API.

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
├── configs/
│   ├── train.yaml                    # Master switchboard
│   └── optimizers/
│       ├── yolo_optim.yaml           # YOLO hyperparameters
│       └── rfdetr_optim.yaml         # RF-DETR hyperparameters
├── scripts/
│   ├── run_train.py                  # Training entrypoint
│   ├── run_infer.py                  # Inference entrypoint
│   ├── run_live.py                   # Live camera inference
│   └── ...
├── src/
│   ├── shared/registry.py            # Registry pattern
│   ├── training/
│   │   ├── base_trainer.py           # Abstract contract
│   │   ├── trainers/
│   │   │   ├── yolo.py               # YOLOv8-Seg trainer
│   │   │   └── rfdetr.py             # RF-DETR-Seg trainer
│   │   └── hooks/
│   │       └── industrial_logger.py  # Epoch logger
│   ├── inference/                    # Inference engines
│   └── preprocess/                   # CLAHE engine
└── data/                             # Datasets
```

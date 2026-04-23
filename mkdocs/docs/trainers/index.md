---
hide:
  - toc
---

# Trainers

Two model architectures, one unified interface.

<div class="grid-container" markdown>
<div class="card" markdown>

### ⚡ YOLOv26-Seg

CNN-based, NMS-free instance segmentation. One-to-one label assignment during training, fast inference, high batch sizes. Wraps Ultralytics with hook bridging and dynamic augmentation injection.

[:material-arrow-right: Read More](yolo.md)

</div>
<div class="card" markdown>

### 🤖 RF-DETR-Seg

Transformer-based instance segmentation. DINOv2 backbone, deformable attention, dual learning rates. Global context understanding for overlapping parcels.

[:material-arrow-right: Read More](rfdetr.md)

</div>
</div>

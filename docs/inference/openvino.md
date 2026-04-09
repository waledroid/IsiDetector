# OpenVINO Engine

The `OpenVINOInferencer` is optimized for **CPU-only deployment** on Intel hardware. It loads models exported to OpenVINO IR format (`.xml` + `.bin`) via the [Export Engine](export.md).

---

## When to Use

- **No NVIDIA GPU** on the deployment machine
- **Intel CPUs** — OpenVINO uses AVX-512 / VNNI instructions for 2-5x speedup over ONNX CPU
- **Edge gateways** — low-power Intel NUCs, industrial PCs

!!! tip
    On Intel CPUs, OpenVINO is significantly faster than ONNX Runtime CPU. Always prefer `.xml` over `.onnx` for CPU-only deployments.

---

## Technical Overview

:material-file-code: **Source**: `src/inference/openvino_inferencer.py`

### Model Loading

```python
import openvino as ov

core = ov.Core()
model = core.read_model("openvino/model.xml")
compiled = core.compile_model(model, "CPU")
infer_request = compiled.create_infer_request()
```

The inferencer auto-detects:

- **Input dimensions** from the model metadata (supports 416, 512, 640, etc.)
- **Model type** (YOLO vs RF-DETR) from output tensor names
- **Number of classes** from the output shape

### Postprocessing

Identical to the ONNX engine — the model graph is the same, just compiled differently:

| Model | Box Format | Box Space | Score Type | Masks |
|---|---|---|---|---|
| **YOLO** | `[x1, y1, x2, y2]` | Pixel (0-imgsz) | Direct confidence | Proto-based (32 coefficients) |
| **RF-DETR** | `[cx, cy, w, h]` | Normalized (0-1) | Sigmoid of logits | Per-detection masks |

---

## Usage

=== "CLI"

    ```bash
    python scripts/run_live.py \
        --weights models/rfdetr/31-03-2026_1117/openvino/model.xml \
        --source 0
    ```

=== "Web App"

    Select the `model.xml` file from the Settings dropdown. The platform auto-detects it as OpenVINO and forces CPU execution.

=== "Python"

    ```python
    from src.inference.openvino_inferencer import OpenVINOInferencer

    engine = OpenVINOInferencer(
        model_path="openvino/model.xml",
        conf_threshold=0.5
    )
    detections = engine.predict_frame(frame)
    ```

---

## How to Generate OpenVINO Models

See the [Export Engine](export.md) documentation. Quick reference:

```bash
# From existing ONNX
python -m src.inference.export_engine \
    --model-dir models/rfdetr/31-03-2026_1117 \
    --format openvino

# From raw weights (auto-exports ONNX first)
python -m src.inference.export_engine \
    --weights runs/segment/models/yolo/best.pt \
    --format openvino
```

Output: `openvino/model.xml` + `openvino/model.bin` alongside the source weights.

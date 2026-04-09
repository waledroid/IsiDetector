# Export Engine

The export engine converts trained model weights into optimized deployment formats. It runs **automatically after training** and is also available as a **standalone CLI tool** for manual conversion.

---

## Supported Formats

| Format | Extension | Hardware | Use Case |
|---|---|---|---|
| **ONNX** | `.onnx` / `.sim.onnx` | GPU (CUDA) or CPU | Universal format, good GPU performance |
| **OpenVINO** | `.xml` + `.bin` | CPU (Intel optimized) | Best for CPU-only deployments |
| **TensorRT** | `.engine` | GPU (NVIDIA only) | Maximum GPU throughput, architecture-specific |

---

## Output Structure

After export, all formats are placed alongside the source weights:

```
models/rfdetr/31-03-2026_1117/
+-- checkpoint_best_ema.pth         # Original weights
+-- inference_model.onnx            # Raw ONNX (from training)
+-- inference_model.sim.onnx        # Optimized ONNX (onnxsim)
+-- openvino/
|   +-- model.xml                   # OpenVINO IR
|   +-- model.bin
+-- tensorrt/
    +-- model.engine                # TensorRT (if available)

runs/segment/models/yolo/yolo26m_640_200/weights/
+-- best.pt                         # Original weights
+-- best.onnx                       # ONNX (from Ultralytics export)
+-- best.sim.onnx                   # Optimized ONNX
+-- openvino/
|   +-- model.xml
|   +-- model.bin
+-- tensorrt/
    +-- model.engine
```

---

## Automatic Export (After Training)

Both trainers call the export engine automatically after `trainer.export()`:

```python
# In YOLOTrainer.export() and RFDETRTrainer.export():
from src.inference.export_engine import run_pipeline
run_pipeline(model_dir=output_dir, formats={'openvino', 'tensorrt'})
```

This means every completed training run produces all deployment formats automatically. If any conversion fails (e.g., TensorRT not installed), training still succeeds — it's wrapped in a try/except.

---

## Manual CLI Usage

:material-file-code: **Source**: `src/inference/export_engine.py`

### Auto-Discover and Convert

```bash
# Discover ONNX in a model directory, convert to all formats
python -m src.inference.export_engine \
    --model-dir models/rfdetr/31-03-2026_1117 \
    --format all
```

### Selective Formats

```bash
# Only OpenVINO (for CPU deployment)
python -m src.inference.export_engine \
    --model-dir runs/segment/models/yolo/yolo26m_640_200/weights \
    --format openvino

# Only optimize ONNX (no conversion)
python -m src.inference.export_engine \
    --model-dir models/rfdetr/28-03-2026_2257 \
    --format onnx
```

### From Raw Weights

```bash
# No pre-existing ONNX — exports from .pt first
python -m src.inference.export_engine \
    --weights runs/segment/models/yolo/yolo26m_416_200/weights/best.pt \
    --format all

# Override image size (default: read from model's training config)
python -m src.inference.export_engine \
    --weights best.pt --imgsz 512 \
    --format onnx openvino
```

### Format Flags

| `--format` | What It Does |
|---|---|
| `onnx` | Runs `onnxsim.simplify()` on the ONNX model |
| `openvino` | Converts ONNX to OpenVINO IR via `openvino.convert_model()` |
| `tensorrt` | Converts ONNX to TensorRT engine via `trtexec` (skips if not installed) |
| `all` | All of the above (default) |

---

## ONNX Discovery Logic

The export engine searches for the best available ONNX before converting:

1. **`.sim.onnx`** found → use it (already optimized)
2. **`.onnx`** found → run `onnxsim` to produce `.sim.onnx`
3. **No ONNX** → look for `.pt`/`.pth` weights and export ONNX first

---

## Image Size Handling

The ONNX model **bakes in** the input size at export time. The export engine handles this automatically:

- **YOLO `.pt`**: Reads `imgsz` from the model's training config (`model.overrides['imgsz']`)
- **RF-DETR `.pth`**: Uses the `resolution` parameter from training
- **`--imgsz` flag**: Overrides auto-detection

At inference time, all engines (ONNX, OpenVINO, TensorRT) read the input dimensions from the model metadata — no hardcoding needed.

---

## Dependencies

All required packages are already installed:

| Package | Purpose |
|---|---|
| `onnx` | ONNX model format |
| `onnxsim` | Graph simplification |
| `openvino` | OpenVINO conversion |
| `trtexec` | TensorRT engine builder (optional, GPU-only) |

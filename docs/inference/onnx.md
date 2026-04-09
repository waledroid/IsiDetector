# High-Speed ONNX Engine

The `OptimizedONNXInferencer` runs inference via **ONNX Runtime** with automatic GPU (CUDA) or CPU fallback. It supports both YOLO and RF-DETR models with full instance segmentation (bounding boxes + masks).

---

## Technical Overview

:material-file-code: **Source**: `src/inference/onnx_inferencer.py`

### 1. Instance Segmentation Masks

**YOLO masks** use a proto-mask architecture. The model outputs 32 mask prototype maps and per-detection coefficients. The engine decodes these via matrix multiplication:

```python
# [N, 32] @ [32, proto_h * proto_w] → [N, proto_h * proto_w]
masks_raw = coeffs @ proto.reshape(32, -1)
masks = sigmoid(masks_raw)  # Per-pixel probability
```

Each mask is cropped to its bounding box and resized to the original image dimensions.

**RF-DETR masks** are per-detection predictions (108x108 each), resized and thresholded directly.

### 2. Coordinate Space Handling

The two model families output boxes in different formats:

| Model | Format | Space | Conversion |
|---|---|---|---|
| **YOLO** (nms=True export) | `[x1, y1, x2, y2]` | Pixel (0-imgsz) | Scale by `orig / model` ratio |
| **RF-DETR** | `[cx, cy, w, h]` | Normalized (0-1) | Convert center→corner, multiply by `orig_w/h` |

### 3. Dynamic Input Scaling
The engine reads the required input shape directly from the `.onnx` file metadata. This ensures compatibility with any model resolution (416, 512, 640, etc.) without code changes.

### 4. GPU/CPU Auto-Detection

```python
if 'CUDAExecutionProvider' in ort.get_available_providers():
    # Use GPU with optimized CUDA settings
else:
    # Fallback to CPU
```

!!! warning "Conflict Detection"
    If both `onnxruntime` and `onnxruntime-gpu` are installed, the CPU version may shadow the GPU version. The engine logs a warning with the fix: `pip uninstall onnxruntime -y && pip install onnxruntime-gpu`.

---

## When to Use

- **GPU available**: Best balance of speed and compatibility
- **CPU fallback**: Works but slower than [OpenVINO](openvino.md) on Intel CPUs
- **Cross-platform**: ONNX runs on any hardware with onnxruntime installed

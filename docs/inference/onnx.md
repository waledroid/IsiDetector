# High-Speed ONNX Engine

The `ONNXInferencer` is specifically designed for high-throughput industrial environments. It bypasses the overhead of heavy deep learning frameworks like PyTorch or Ultralytics during the inference phase, using **ONNX Runtime** for maximum CPU/GPU efficiency.

---

## Technical Overview

:material-file-code: **Source**: `src/inference/onnx_inferencer.py`

### 1. Vectorized Mask Decoding
A major bottleneck in segmentation is decoding the model's Proto-masks into human-readable masks. While standard implementations use slow Python loops, our `ONNXInferencer` uses **NumPy Broadcasting** (Vectorization) to process all masks simultaneously.

Instead of:
```python
# Slow way (Python Loop)
for i in range(n):
    mask = sigmoid(matrix_mult(protos, coeffs[i]))
```

We use:
```python
# Fast way (Vectorized)
# Compute all masks in a single matrix operation
masks = sigmoid(protos @ coeffs.T)
```

### 2. Dynamic Input Scaling
Unlike hardcoded scripts, this engine dynamically extracts the required input shape directly from the `.onnx` file metadata. This ensures compatibility with any model resolution (e.g., 320x320, 640x640, or 1280x1280) without code changes.

### 3. Automatic Class Mapping
The engine automatically handles the differences between YOLO and RF-DETR output structures:
- **YOLO**: Dynamically slices the score tensor based on the class count in your `train.yaml`.
- **RF-DETR**: Automatically shifts class indices by +1 to align with Transformer-based COCO conventions.

---

## When to Use?
- **Live Streams**: When processing 30 FPS RTSP feeds.
- **Edge Devices**: When running on hardware with limited CPU power.
- **Production**: This is our most stable and performant engine for long-term deployment.

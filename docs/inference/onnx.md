# High-Speed ONNX Engine

The `OptimizedONNXInferencer` runs inference via **ONNX Runtime** with automatic GPU (CUDA) or CPU fallback. It supports both the YOLO family (CSPDarknet-based, including the NMS-free yolov26) and RF-DETR (DINOv2 transformer), with full instance segmentation — bounding boxes *and* masks — for either.

The engine introspects the `.onnx` file at load time and adapts its preprocessing and postprocessing to match the model family. No manual configuration is required — drop an exported `.onnx` in place and it works.

---

## Technical Overview

:material-file-code: **Source**: `src/inference/onnx_inferencer.py`

### 1. Automatic Model-Family Detection

At load time the engine inspects the ONNX graph's output tensor names and shapes to classify the model:

| Family | Output signature | Detection path |
|---|---|---|
| **YOLO post-NMS** (`nms=True` export) | one output `[1, N≤300, 6]` or `[1, N, 6+32]` | Columns are `[x1, y1, x2, y2, score, class, (mask_coeffs)]` |
| **YOLO pre-NMS / raw** (`nms=False` export) | one output with an anchor-axis size > 1000 | Per-class scores; class-aware NMS run in Python via `cv2.dnn.NMSBoxes` |
| **RF-DETR** | outputs named `dets`, `labels`, `masks` (or `pred_boxes`, `pred_logits`, `pred_masks`) | `sigmoid → topk` across the `(queries × classes)` matrix |

The `is_rfdetr` flag is set from output names; YOLO post-vs-pre-NMS is decided at inference time from the tensor shape. Both paths are handled inside `predict_frame()`.

### 2. Preprocessing: Family-Specific

Each family expects a specific input recipe. Mismatching these is a silent bug — boxes still appear but accuracy collapses.

| Step | YOLO | RF-DETR |
|---|---|---|
| Colour | BGR → **RGB** (cv2 gives BGR, training used RGB) | BGR → **RGB** |
| Resize | **Letterbox** pad to `(imgsz, imgsz)` with grey 114 fill, preserves aspect ratio | **Stretch** resize to `(imgsz, imgsz)` (matches `rfdetr` library's `F.resize(res, res)`) |
| Range | `/255` → [0, 1] | `/255` → [0, 1] |
| Normalisation | none | **ImageNet** `(x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]` — required by DINOv2 backbone |
| Layout | HWC → CHW, add batch dim, NCHW float32 | same |

The letterbox transform stores `(ratio, pad_x, pad_y)` so boxes can be mapped back to the original frame coordinates after inference.

### 3. Postprocessing

#### YOLO (post-NMS)

```python
preds = outputs[0][0]                      # (N, 6 + 32?)
conf  = preds[:, 4]                        # single score column
cls   = preds[:, 5].astype(int)            # single class column
coeffs = preds[:, 6:] if seg else None

# Filter by confidence, invert letterbox, clip to frame
keep  = conf > self.conf_threshold
boxes = (preds[keep, :4] - (pad_x, pad_y, pad_x, pad_y)) / ratio
```

#### YOLO (pre-NMS / raw)

Raw output is `(1, 4 + nc + 32, A)` where `A` is the anchor count (e.g. 3549 for 416 input). The engine transposes to `(A, 4 + nc + 32)`, takes per-class max scores, applies per-class NMS via `cv2.dnn.NMSBoxes`, converts `cxcywh → xyxy`, then inverts the letterbox.

#### RF-DETR

Replicates the native `.pth` postprocess from `src/training/trainers/rfdetr.py:92-140`:

```python
# Slice away background (index 0) and noise classes — keep only fine-tuned slots
logits = logits[:, 1 : 1 + self.nc]        # (Q, 2)
probs  = sigmoid(logits)                   # stable clip to [-50, 50]

# Top-K over the flattened (query × class) matrix
flat = probs.reshape(-1)
topk = argpartition(-flat, k)[:k]
scores    = flat[topk]
query_idx = topk // nc
class_ids = (topk % nc) + 1                # +1 restores 1-indexed convention
```

Why `logits[:, 1:1+nc]`: the pretrained head has 91 COCO slots. Index 0 is a **background** class that stays near-zero on real detections; fine-tuned classes (carton, polybag) live at indices 1 and 2. Slicing skips the dead column and discards unused COCO slots so pretraining noise can't fire spurious detections.

The `+1` on `class_ids` restores 1-indexed class IDs that match the RF-DETR `.pth` inferencer's convention — this is what keeps mask and box palette colours distinct from YOLO on hot-swap.

### 4. Mask Decoding — Bbox-Confined Upsample

YOLO segmentation decodes masks via matrix factorisation:

```python
# (N, 32) @ (32, Ph * Pw) → (N, Ph, Pw)
masks_raw = sigmoid(coeffs @ proto.reshape(32, -1))
```

Naively upsampling each (Ph × Pw) mask to the full frame size (e.g. 1920 × 1080) N times is expensive. The engine instead:

1. Maps each detection's bounding box from original frame → letterboxed model → proto coordinates.
2. Crops the mask to that proto-space region only.
3. Resizes just the crop to the bbox size in original pixels.
4. Pastes into a zero-filled full-frame `bool` mask.

Cost scales with total bbox area, not frame area — roughly 50× faster than full-frame upsampling on 1080p sources. RF-DETR masks use the same approach.

### 5. CUDA Kernel Preload at App Startup

First-time ONNX session creation on CUDA includes cuDNN algorithm autotuning — several seconds for DINOv2-backed RF-DETR. `preload_onnx(model_path)` builds a session at app boot, runs one dummy inference to trigger kernel compilation, then **discards the session**. The compiled kernels stay resident in the CUDA driver cache for the rest of the process lifetime; when the actual swap later builds a fresh session, it reuses those kernels and completes in ~2 seconds instead of 5–8.

!!! note "Why discard instead of keeping the session"
    Early iterations cached the session object across threads. ONNX Runtime's CUDA EP binds each session to the stream of the thread that built it; calling `run()` from another thread triggers stream re-synchronisation that can cost tens of seconds. The driver-level cache is thread-agnostic, so the workaround is: warm the cache, discard the session, rebuild on demand in the consumer thread.

`StreamHandler.__init__` spawns a background daemon that reads the default RF-DETR ONNX path from `settings.json` and calls `preload_onnx`.

### 6. Dynamic Input Shape

The engine reads the required input shape directly from the `.onnx` graph metadata (`session.get_inputs()[0].shape`). Models at 416, 432, 512, 640, or any other resolution work without code changes. A dynamic batch dimension is pinned to 1 at inference.

### 7. GPU / CPU Auto-Selection

```python
available = ort.get_available_providers()
if 'CUDAExecutionProvider' in available:
    providers = [('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'HEURISTIC',
    }), 'CPUExecutionProvider']
else:
    providers = ['CPUExecutionProvider']
```

Always check `session.get_providers()[0]` to verify CUDA actually got picked — CPU fallback is silent.

!!! warning "`onnxruntime` vs `onnxruntime-gpu` conflict"
    If both packages are installed in the same environment, the CPU build shadows the GPU one and `CUDAExecutionProvider` quietly disappears. The engine detects this at import time and auto-uninstalls the CPU package, falling back to a warning if removal fails: `pip uninstall onnxruntime -y`.

### 8. Numerically Stable Sigmoid

Mask logits can reach magnitudes of `-125` for background regions. Naive `1 / (1 + exp(-x))` overflows float32 at those values. The engine's internal `_sigmoid()` clips logits to `[-50, 50]` before `exp` — saturation at those magnitudes is unchanged, but `RuntimeWarning: overflow encountered in exp` is silenced.

---

## Performance Notes

| Stage | Typical cost @ 416 input |
|---|---|
| Preprocess (BGR→RGB, letterbox/stretch, transpose, normalise) | 1–3 ms |
| `session.run()` CUDA forward | YOLOv26n: ~5–8 ms &nbsp;&nbsp;&nbsp; RF-DETR Medium: ~15–25 ms |
| Postprocess (threshold, NMS, letterbox inverse) | < 1 ms |
| Mask decode (matmul + sigmoid) | 1–2 ms |
| Bbox-confined mask upsample (per detection) | ~1 ms × N |

Hot-swap latency after preload: ~2 seconds. First swap without preload: 5–8 seconds.

---

## When to Use

- **GPU available**: ONNX on CUDA is the default for production.
- **CPU fallback**: Works everywhere; consider [OpenVINO](openvino.md) on Intel hardware.
- **Cross-platform deployment**: ONNX graphs are framework-agnostic; the same `.onnx` file runs on any host with `onnxruntime-gpu` or `onnxruntime`.

For sub-millisecond inference on Ampere+ GPUs, export to [TensorRT](tensorrt.md) `.engine` format via `src/inference/export_engine.py` and route via the `.engine` extension instead.

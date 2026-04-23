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

:material-file-code: **Source**: `isidet/src/inference/openvino_inferencer.py`

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
    python isidet/scripts/run_live.py \
        --weights isidet/models/rfdetr/31-03-2026_1117/openvino/model.xml \
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

## Correctness Deep-Dive: Matching ONNX Output Bit-for-Bit

The OpenVINO engine must produce **identical detections** to the ONNX engine — operators do visual swaps between the two and any divergence looks like a bug. Two subtle correctness items bit us during deployment; this section explains what they are so you can spot them in future model additions.

### 1. Letterbox, not stretch-resize

**Problem observed:** On the web app with OpenVINO, polybags were repeatedly misclassified as cartons, confidence scores were low (~0.6), and masks rendered in the wrong region. Switching to ONNX on the same model made the problem disappear.

**Root cause:** The two inferencers were using different preprocessing.

=== "Wrong — stretch resize"

    ```python
    img = cv2.resize(frame, (self.model_w, self.model_h))   # distorts aspect ratio
    ```

    A 1280×720 (16:9) camera frame resized to 320×320 squashes objects horizontally. A polybag that was trained to look roughly square now appears wide and flat — and looks more like a carton to the model. Every prediction is being made on an input the model never saw during training.

=== "Correct — letterbox"

    ```python
    # Aspect-preserving resize + pad to (model_h, model_w) with gray (114)
    r = min(self.model_h / orig_h, self.model_w / orig_w)
    new_w, new_h = int(round(orig_w * r)), int(round(orig_h * r))
    resized = cv2.resize(frame, (new_w, new_h))
    padded = cv2.copyMakeBorder(
        resized,
        pad_h // 2, pad_h - pad_h // 2,
        pad_w // 2, pad_w - pad_w // 2,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    self._last_letterbox = (r, pad_x, pad_y)   # remember for inverse transform
    ```

    Object proportions are preserved. The model sees exactly what it saw during Ultralytics training (which uses letterbox by default).

**Follow-on fix — inverse transform on outputs.** Boxes come back in letterboxed model-pixel space. To get original-image coordinates:

```python
ratio, pad_x, pad_y = self._last_letterbox
boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
```

**Masks too.** Mask prototypes are emitted in letterboxed space (e.g. 80×80 for a 320-input model). To project a detection's bounding box onto the proto grid:

```
orig_coord → (orig_coord * ratio + pad) → proto_coord * (proto_size / model_size)
```

This is what `_process_masks` does. Skip the letterbox term and your masks land in the wrong region — particularly visible on non-square frames where `pad_y > 0`.

!!! warning "RF-DETR is the exception"
    RF-DETR (DINOv2 backbone) was trained on stretch-resized inputs (no letterbox). The `preprocess()` method branches on `self.is_rfdetr` and skips the letterbox for DETR-family models. Don't "fix" this by unifying the two paths — the model will silently break.

### 2. Post-NMS vs pre-NMS output layout (YOLO only)

Ultralytics YOLO exports to ONNX with one of two layouts, chosen at export time:

=== "Post-NMS (`nms=True` at export)"

    Output shape: `[1, 300, 6]` (detection) or `[1, 300, 38]` (segmentation).

    - col 0–3: `x1, y1, x2, y2`
    - col 4: **confidence** (single scalar, NMS already picked it)
    - col 5: **class_id** (argmaxed, integer)
    - col 6–37: 32 mask coefficients (segmentation only)

    The model emits at most 300 detections, already filtered and scored. Read col 4 as confidence, col 5 as class id, done.

=== "Pre-NMS (`nms=False` at export)"

    Output shape: `[1, 4+nc(+32), A]` where `A ≈ 3549` anchor positions for 416-input.

    - col 0–3: `cx, cy, w, h` (center-width format)
    - col 4:`4+nc`: per-class scores (need argmax to pick class)
    - col 4+nc:: mask coefficients

    You must run `cv2.dnn.NMSBoxes` yourself to dedupe overlapping anchors.

**Why this matters:** If you parse a post-NMS output as if it were pre-NMS, you take `argmax([conf, class_id])` across cols 4-5 as if they were per-class scores. The result is random garbage for both class and confidence — every one of the 300 slots passes the threshold (wrong confidence), and rendering 300 spurious boxes per frame kills FPS to 0.2.

The inferencer distinguishes the two layouts by the **anchor-axis magnitude** — pre-NMS exports always have a dimension > 1000 (the anchor count), post-NMS never does:

```python
is_raw = max(raw.shape[0], raw.shape[1]) > 1000
if is_raw:
    return self._postprocess_yolo_raw(...)      # pre-NMS path
# otherwise fall through to post-NMS path
```

Both paths are implemented in `OpenVINOInferencer` to match the ONNX engine.

### 3. RF-DETR on OpenVINO: known-broken (as of OpenVINO 2026.0)

This is a deployment reality check, not a code bug.

**Symptom:** OpenVINO RF-DETR returns zero detections, even though ONNX on the same `.onnx` file returns correct detections.

**Evidence:** Dumping the raw logits shows OpenVINO's transformer outputs diverge from ONNX by `|Δ| up to 8.8` on labels and `|Δ| up to 1.8` on normalised boxes — well past numerical noise. The divergence is identical with:

- FP16 weight compression (default)
- FP32 weights (`compress_to_fp16=False`)
- Simplified ONNX input (`.sim.onnx`)
- Un-simplified ONNX input

The likely culprit is OpenVINO's translation of the segmentation head's `Einsum` op (`bchw,bnc→bnhw`) or dynamic-shape resolution in DINOv2 attention — both historically weak spots in OpenVINO's transformer support.

**Recommendation:** Do not deploy RF-DETR via OpenVINO. For CPU hosts:

- **YOLO + OpenVINO** — works perfectly, 40+ FPS on AMD CPU, correct masks, stable classes. This is the recommended CPU path.
- **RF-DETR + ONNX CPU EP** — numerically correct but slow (~2 FPS on CPU); only useful as a fallback when GPU is unavailable *and* the accuracy advantage justifies the frame-rate hit.
- **RF-DETR is a GPU-only model in practice** — the transformer backbone needs CUDA kernels to be real-time.

The inferencer still includes the RF-DETR postprocess (top-K over queries×classes, background class skip, 1-indexed class emission) in case a future OpenVINO release fixes the conversion, but the IR it reads today produces zero detections.

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

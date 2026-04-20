# Working with OpenVINO Models for Computer Vision Inference

A practical walk-through of how to take an ONNX file, convert it to OpenVINO's Intermediate Representation, and build a correct, fast CPU inference engine around it. Illustrated with the real debugging story of `src/inference/openvino_inferencer.py` in this repo, including the bugs that produced "0.2 FPS with 300 garbage boxes" and the fixes that got us to "40+ FPS with stable detection and rendered masks."

You'll learn:

- What OpenVINO actually is and why it exists alongside ONNX
- The IR file format (`.xml` + `.bin`) and why it's two files
- How `ov.convert_model()` translates ONNX into IR — and what it silently mangles
- The `compress_to_fp16` trap that breaks transformer models
- How to compile a model for a specific device (CPU / iGPU / NPU / AUTO)
- The letterbox-vs-stretch preprocessing rule that decides whether your detections work
- The post-NMS vs pre-NMS YOLO output layouts and how to detect which you have
- The RF-DETR + Einsum compatibility wall (and why we route DETR-family models through ONNX instead)
- The full set of bugs we fixed, what each one teaches, and the runtime traces that flushed them out

Each section starts with the high-level idea, then drops into the actual code, math, or runtime trace.

---

## 1. What Is OpenVINO, Really?

**OpenVINO** (Open Visual Inference and Neural network Optimization) is Intel's inference toolkit. Three things matter for our purposes:

1. **A model format** — Intermediate Representation (IR), a pair of files: `model.xml` (graph topology) + `model.bin` (weights).
2. **An optimisation step** — `ov.convert_model()` reads an ONNX (or PyTorch, TF, PaddlePaddle) file, fuses ops, folds constants, picks Intel-friendly kernels.
3. **A runtime** — `core.compile_model(...)` JIT-compiles the IR into device-specific code (CPU AVX-512/VNNI, Intel iGPU OpenCL, Intel NPU). Then `compiled([input])` runs it.

It's the same shape as ONNX Runtime, but with a deeper Intel-hardware optimisation budget on the CPU side.

### Mental model

```
┌──────────────┐                ┌──────────────┐               ┌──────────────────┐
│  PyTorch     │  torch.export  │  inference_  │  ov.convert_  │  OpenVINO IR     │
│  / TF /      │  ────────────► │  model.onnx  │  model()      │  model.xml       │
│  PaddlePaddle│                │  (ONNX)      │  ───────────► │  model.bin       │
└──────────────┘                └──────────────┘               └────────┬─────────┘
                                                                        │
                                                                        ▼  core.compile_model()
                                                              ┌──────────────────┐
                                                              │  CompiledModel   │
                                                              │  CPU / GPU / NPU │
                                                              │  + InferRequest  │
                                                              └────────┬─────────┘
                                                                        │  compiled([np_input])
                                                                        ▼
                                                              ┌──────────────────┐
                                                              │  output tensors  │
                                                              └──────────────────┘
```

### When OpenVINO beats ONNX Runtime CPU

- **Intel CPUs** with AVX-512 or VNNI: 1.5–3× faster than ONNX CPU on the same model.
- **Intel iGPUs** (UHD, Iris, Arc): a free 1.5–2× speedup on top of CPU, no dedicated card needed.
- **CNN-heavy graphs** — Conv, MatMul fuse aggressively. YOLO is the canonical good fit.

### When it doesn't help (or actively hurts)

- **AMD CPUs** — falls back to generic AVX2 paths. Still beats ONNX CPU but the gap shrinks.
- **NVIDIA-only deployments** — TensorRT is faster; OpenVINO is irrelevant.
- **Transformer-heavy graphs with exotic ops** — RF-DETR is the cautionary tale (see §10).

### Tools you'll use

| Tool | Purpose |
|---|---|
| [Netron](https://netron.app) | Drop in a `.xml` to visualise the IR graph |
| `openvino` Python | Convert, save, compile, infer |
| `ovc` CLI | One-shot ONNX→IR converter (`ovc model.onnx --output_dir openvino/`) |
| `benchmark_app` | Bundled CLI tool for FPS/latency benchmarking |
| `nncf` | Post-training quantisation (FP16/INT8 with a calibration dataset) |

Install:
```bash
pip install openvino openvino-dev   # openvino-dev gives you ovc + benchmark_app
```

---

## 2. The IR File Format: `.xml` + `.bin`

OpenVINO splits a model into two files **on purpose**, not by accident:

| File | Format | Contents | Typical size |
|---|---|---|---|
| `model.xml` | XML text | Graph topology — layers, shapes, connections, op attributes | ~500 KB – 2 MB |
| `model.bin` | Raw binary | Weight tensors, packed back-to-back, byte-aligned | 5 MB – 500 MB |

### Why two files?

- **Editable topology** — `model.xml` is text. You can grep it, hand-edit input shapes, change layer attributes without re-converting.
- **mmap-friendly weights** — `model.bin` is one contiguous blob. The runtime can `mmap()` it instead of reading + parsing, which makes load times near-instant for large models.
- **Diff-able** — graph changes show up in `git diff` on the XML; weight churn is invisible (just a binary swap).

### What's actually in `model.xml`

```xml
<net name="best" version="11">
    <layers>
        <layer id="0" name="images" type="Parameter">
            <data shape="1, 3, 320, 320" element_type="f32"/>
            <output>
                <port id="0" precision="FP32" names="images">
                    <dim>1</dim><dim>3</dim><dim>320</dim><dim>320</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="/model/model.0/conv/Conv" type="Convolution">
            <data strides="2, 2" pads_begin="1, 1" pads_end="1, 1" kernel="3, 3" />
            ...
        </layer>
        ...
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        ...
    </edges>
</net>
```

It's a flat list of layers with weight tensors named (offsets into `model.bin` are stored in the XML — that's how the two files stay linked).

### The pairing rule

`model.xml` and `model.bin` **must live in the same directory with matching basenames**. When you call `core.read_model("path/to/model.xml")`, OpenVINO automatically resolves `path/to/model.bin`. You never reference the `.bin` directly.

This is why the model dropdown in the web app shows `.xml` files only — picking the `.bin` would be meaningless.

---

## 3. Conversion: ONNX → IR

The conversion is one Python call:

```python
import openvino as ov

ov_model = ov.convert_model("inference_model.sim.onnx")
ov.save_model(ov_model, "openvino/model.xml")
```

What this does in order:

1. **Parse** — read the ONNX protobuf, build an in-memory `ov.Model` (the runtime equivalent of the XML graph).
2. **Type / shape inference** — propagate dtypes and shapes through the graph. Resolves dynamic dims where possible.
3. **Op translation** — map ONNX ops to OpenVINO ops. Most have 1:1 equivalents (`Conv`, `MatMul`, `Sigmoid`); some need decomposition (`Resize` with bilinear), some are *partially* supported (`Einsum` — see §10).
4. **Constant folding** — pre-compute any subgraph whose inputs are all constants.
5. **Op fusion** — merge sequences like `Conv + BatchNorm + ReLU` into a single fused kernel call.
6. **(Default) Compress weights to FP16** — cast all FP32 weight initializers to FP16. **Reduces `model.bin` by 50% but breaks transformer attention.** See §9.

The result is an `ov.Model` ready to be saved or compiled.

### The pt2-archive warning you can safely ignore

When you convert, you'll see this scary stack trace:

```
W ... pt2_archive/_package.py: Unable to load package. f must be a buffer or
a file ending in .pt2. Instead got {.../model.onnx}
RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory
```

This is OpenVINO probing the input as a PyTorch pt2 archive *before* falling back to its ONNX frontend. The probe fails (because the file is ONNX, not pt2), then the ONNX path takes over and works. We silence it in `src/inference/export_engine.py:_silence_torch_export_pt2_probe()` because the noise scared people into thinking the conversion failed.

### CLI alternative: `ovc`

If you want a non-Python conversion path (for CI, Docker build steps, etc.):

```bash
ovc inference_model.sim.onnx --output_dir openvino/
ovc inference_model.sim.onnx --compress_to_fp16=False --output_dir openvino_fp32/
```

Same output, useful when Python isn't around.

---

## 4. Loading and Compiling for a Device

OpenVINO separates **reading** (parse the IR) from **compiling** (target a device). This is intentional — you compile once, run many times.

```python
import openvino as ov

core = ov.Core()                                  # singleton runtime
print("Devices:", core.available_devices)         # ['CPU', 'GPU', 'NPU', ...]

model = core.read_model("openvino/model.xml")     # 1. parse
compiled = core.compile_model(model, "CPU")       # 2. compile for target
infer_request = compiled.create_infer_request()   # 3. allocate run state
```

### Device choices

| Device string | Hardware | Notes |
|---|---|---|
| `"CPU"` | Intel x86 | AVX2 / AVX-512 / VNNI auto-selected |
| `"GPU"` | Intel iGPU/dGPU | Needs `intel-opencl-icd` installed |
| `"NPU"` | Intel NPU | Ultra/Lunar Lake laptops only |
| `"AUTO"` | Best available | OpenVINO picks; falls back if first choice fails |
| `"GPU.0"` / `"GPU.1"` | Specific iGPU/dGPU | Multi-GPU systems |

In `OpenVINOInferencer.__init__` we default to `"CPU"` and only try `"GPU"` if the user explicitly asked for it and `core.available_devices` shows it:

```python
ov_device = "CPU"
if self.device and self.device.upper() in ("GPU", "AUTO"):
    if "GPU" in core.available_devices:
        ov_device = "GPU"
    else:
        logger.info(f"Intel GPU not available, using CPU. Available: {core.available_devices}")
```

### Compile cost

`compile_model` is **slow** — 1-10 seconds depending on model size, because OpenVINO is JIT-compiling kernels for your exact hardware. Do this **once at startup**, never per-frame.

The compiled model is hardware-specific. An IR compiled on an i7-13700H won't be optimal on an i5-1135G7 (different AVX features). The `.xml` + `.bin` are portable; the compiled in-memory result is not.

### Inspecting model IO

```python
print(f"Input:  {compiled.input(0).get_any_name()}  shape={compiled.input(0).shape}")
for i in range(len(compiled.outputs)):
    o = compiled.output(i)
    print(f"Output: {o.get_any_name()}  shape={o.shape}")
```

For our YOLOv26 320×320 segmentation:
```
Input:  images   shape=[1, 3, 320, 320]
Output: output0  shape=[1, 300, 38]    ← post-NMS detections (300 slots, 38 cols)
Output: output1  shape=[1, 32, 80, 80] ← mask prototypes
```

For RF-DETR-Seg:
```
Input:  input    shape=[1, 3, 312, 312]
Output: dets     shape=[1, 100, 4]     ← normalised cxcywh
Output: labels   shape=[1, 100, 91]    ← per-query × per-class logits (COCO 91-head)
Output: masks    shape=[1, 100, 78, 78]← per-query masks
```

The output shapes tell you the model family before you write any postprocessing — see §6.

---

## 5. Preprocessing — The Single Biggest Source of Silent Wrongness

This is the section that, if you get wrong, gives you "model is loading and running but predictions are garbage" — the most demoralising failure mode.

**Rule:** preprocess at inference exactly the way the training pipeline did. No exceptions.

For our two model families that means two completely different recipes:

### 5.1 YOLO — letterbox + /255

Ultralytics YOLO trains on **letterboxed** inputs (aspect-preserving pad with grey 114). At inference you must do the same:

```python
def _letterbox(self, frame, pad_color=114):
    orig_h, orig_w = frame.shape[:2]
    r = min(self.model_h / orig_h, self.model_w / orig_w)
    new_w, new_h = int(round(orig_w * r)), int(round(orig_h * r))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = self.model_w - new_w
    pad_h = self.model_h - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(pad_color,) * 3,
    )
    return padded, r, left, top   # ratio + pads needed for inverse transform
```

Then:

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        # OpenCV reads BGR; YOLO trained on RGB
padded, ratio, pad_x, pad_y = self._letterbox(rgb)
self._last_letterbox = (ratio, pad_x, pad_y)        # remember for postproc inverse
img = padded.transpose((2, 0, 1)).astype(np.float32) / 255.0   # HWC→CHW, [0,255]→[0,1]
return np.ascontiguousarray(img[np.newaxis, ...])    # add batch dim
```

### 5.2 RF-DETR — stretch resize + ImageNet normalisation

RF-DETR's DINOv2 backbone trained on stretch-resized + ImageNet-normalised inputs:

```python
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

if self.is_rfdetr:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (self.model_w, self.model_h), interpolation=cv2.INTER_LINEAR)
    self._last_letterbox = None   # signal: no inverse transform needed
    img = resized.transpose((2, 0, 1)).astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
```

### 5.3 The bug we hit

The first version of `OpenVINOInferencer.preprocess()` used `cv2.resize(frame, (model_w, model_h))` for **everything** — both YOLO and RF-DETR. For YOLO this distorted objects (1280×720 camera squashed to 320×320) and the model — trained only on letterboxed inputs — confidently misclassified them. Symptoms:

- Polybags consistently labelled as cartons (then "corrected" later as the object moved through the frame and the distortion shifted)
- Confidence scores capped around 0.6 instead of the 0.95+ ONNX got on the same image
- Masks rendered in the wrong region of the frame

The fix is what's shown above. **The model didn't change. The preprocessing did.** And that recovered correctness *and* speed (no more rendering 300 spurious detections per frame).

### 5.4 How to verify your preprocessing is right

Run the same numpy input through both your OpenVINO inferencer and the ONNX inferencer. The output tensors should match within ~1e-3 absolute difference. If they don't, your preprocessing is the issue 9 times out of 10. Code:

```python
inp = preprocess(img, ...)   # your candidate preprocessing

ov_out = compiled([inp])
onnx_out = onnx_session.run(None, {input_name: inp})

print("Max abs diff:", np.abs(ov_out[0] - onnx_out[0]).max())
# Should be < 0.01. If 1.0+, preprocessing or conversion is broken.
```

---

## 6. Detecting Model Family from Outputs

`OpenVINOInferencer.__init__` has to know if it's looking at YOLO or RF-DETR before postprocessing. We use **output names** as the discriminator:

```python
self.output_names = [self.compiled.output(i).get_any_name()
                     for i in range(len(self.compiled.outputs))]
self.is_rfdetr = any(n in self.output_names
                      for n in ('dets', 'pred_logits', 'bboxes', 'labels'))
```

DETR-family exports use semantic names (`dets`, `labels`, `pred_logits`, `pred_boxes`, `pred_masks`). Ultralytics YOLO uses positional names (`output0`, `output1`). The presence of any DETR name flips the branch.

This is more robust than checking output **shapes** because:
- Both families can have 2 or 3 outputs depending on segmentation
- YOLO output shapes vary by export type (post-NMS vs raw)
- Naming is stable across export versions

---

## 7. YOLO Postprocessing — Two Layouts to Handle

Ultralytics YOLO has **two** ONNX export layouts, and you must detect which one you have at runtime. Misreading the layout produced our worst bug — see §13.

### 7.1 Post-NMS (`nms=True` at export time)

Output shape: `[1, N, 6]` (detection) or `[1, N, 6+M]` (segmentation, M = mask coeffs, usually 32).

`N` is at most 300 (or whatever `max_det` was set to). Columns:

| Col | Meaning |
|---|---|
| 0–3 | `x1, y1, x2, y2` (xyxy in letterboxed model space) |
| 4 | confidence (single scalar — NMS already picked the winner per detection) |
| 5 | class_id (already argmaxed, integer-valued float) |
| 6+ | mask coefficients (segmentation only) |

Parsing:

```python
boxes = preds[:, :4].copy()
confidences = preds[:, 4].astype(np.float32)
class_ids = preds[:, 5].astype(int)
mask_coeffs = preds[:, 6:] if preds.shape[1] > 6 else None

keep = confidences > self.conf_threshold
boxes = boxes[keep]; confidences = confidences[keep]; class_ids = class_ids[keep]
```

### 7.2 Pre-NMS (`nms=False` at export time)

Output shape: `[1, A, 4+nc(+M)]` or `[1, 4+nc(+M), A]` (Ultralytics has flipped this dim order across versions).

`A` is the **anchor count** — every grid position across all feature pyramid levels. For 416-input that's ~3549; for 320-input ~2100. The model emits raw scores at every anchor; you have to argmax + NMS yourself.

| Col | Meaning |
|---|---|
| 0–3 | `cx, cy, w, h` (centre-width in letterboxed model space) |
| 4:`4+nc` | per-class scores (sigmoid-activated, need argmax) |
| `4+nc:` | mask coefficients |

Parsing:

```python
class_ids = np.argmax(scores, axis=1)
confidences = np.max(scores, axis=1)
keep = confidences > self.conf_threshold
# ...

# Manual NMS — Ultralytics' embedded NMS is gone, do it yourself
idx = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(),
                       self.conf_threshold, 0.45)
boxes = boxes[idx]
```

### 7.3 Detecting which layout you have

The key insight: pre-NMS exports always have one dimension > 1000 (the anchor count). Post-NMS never does (≤300 detections).

```python
raw = outputs[0][0]   # drop batch dim
is_raw = max(raw.shape[0], raw.shape[1]) > 1000
if is_raw:
    preds = raw if raw.shape[0] > raw.shape[1] else raw.T   # normalise to [A, features]
    return self._postprocess_yolo_raw(preds, ...)
# otherwise post-NMS path
```

### 7.4 Inverting the letterbox on output boxes

Box coordinates come back in **letterboxed model space**. To get original-image coordinates you invert the preprocessing transform:

```
orig_x = (model_x - pad_x) / ratio
orig_y = (model_y - pad_y) / ratio
```

In code:

```python
ratio, pad_x, pad_y = self._last_letterbox
boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
```

If you skip this step, boxes will be drawn at fractional positions in the original frame — visible by eye on a non-square source.

---

## 8. RF-DETR Postprocessing — Three Subtleties

DETR-family postprocessing has gotchas that don't exist in YOLO. The OpenVINO inferencer's RF-DETR branch was wrong on all three before our fix.

### 8.1 The 91-class head with background at index 0

RF-DETR was pretrained on COCO (91 classes) and fine-tuned with our 2 classes. The classification head still emits **91 logits per query**, with index 0 reserved for "background / no-object."

Naive postprocessing does `argmax` over all 91 columns and gets class 0 (background) for almost every query, because background is always the highest-probability class for non-detected queries. You then emit `class_id=0` for everything.

Fix: slice off the background column before sigmoid:

```python
logits = logits[:, 1:1 + effective_nc]   # drop background, keep first `nc` real classes
```

After this slice, column 0 = first fine-tuned class, column 1 = second, etc.

### 8.2 Top-K over (queries × classes), not per-query argmax

The native PyTorch RF-DETR postprocess (`rfdetr/util/box_ops.py`) doesn't pick one class per query and rank queries by score. It picks the top 300 (query, class) **pairs** across the entire flattened score matrix.

```python
probs = sigmoid(logits)              # [N_queries, nc]
flat = probs.reshape(-1)             # flatten to [N_queries * nc]
topk_idx = np.argpartition(-flat, num_select - 1)[:num_select]
topk_idx = topk_idx[np.argsort(-flat[topk_idx])]   # sort descending
scores = flat[topk_idx]
query_idx = topk_idx // nc
class_ids = (topk_idx % nc).astype(int) + 1   # +1 to restore the COCO indexing convention
```

The `+1` is because the app's `class_names` dict for RF-DETR is `{1: "carton", 2: "polybag"}` — DETR-family models use 1-indexed class IDs by convention (palette slot 0 reserved for background).

### 8.3 Stretch-resize box mapping

RF-DETR's box outputs are **normalised** `[cx, cy, w, h]` in `[0, 1]`. Because preprocessing was stretch resize (no letterbox), mapping to original coordinates is a simple multiply:

```python
boxes_xyxy[:, [0, 2]] *= orig_w
boxes_xyxy[:, [1, 3]] *= orig_h
```

No `pad_x`, no `ratio` — the stretch transform was a pure scale.

---

## 9. Mask Decoding

### 9.1 YOLO masks — prototype factorisation

YOLOv8/v26-seg outputs **mask prototypes** plus per-detection coefficients, not full masks per detection. This is much smaller — instead of `[N, 320, 320]` you get `[32, 80, 80]` (proto) + `[N, 32]` (coefficients).

The mask for detection `i` is reconstructed as:

```
mask_i = sigmoid(coeffs[i] @ proto.reshape(32, -1)).reshape(80, 80)
```

In code, vectorised across all detections:

```python
masks_raw = coeffs @ proto.reshape(proto.shape[0], -1)   # [N, 80*80]
masks_raw = masks_raw.reshape(n, 80, 80)
masks_raw = sigmoid(masks_raw)
```

**Key correctness item:** the mask is in **letterboxed model space**, not original-image space. To project a detection's bounding box onto the proto grid:

```
orig_box → letterboxed_model_box → proto_box
   x1     →  x1 * ratio + pad_x   →  (x1 * ratio + pad_x) * (proto_w / model_w)
```

Then crop the proto-space mask, threshold at 0.5, and resize to the bounding box region in the original frame:

```python
ratio, pad_x, pad_y = self._last_letterbox
sx = proto_w / self.model_w
sy = proto_h / self.model_h

for i in range(n):
    x1, y1, x2, y2 = boxes[i]                              # original coords
    px1 = int(np.floor((x1 * ratio + pad_x) * sx))        # → proto coords
    py1 = int(np.floor((y1 * ratio + pad_y) * sy))
    px2 = int(np.ceil((x2 * ratio + pad_x) * sx))
    py2 = int(np.ceil((y2 * ratio + pad_y) * sy))

    mask_crop = masks_raw[i, py1:py2, px1:px2]
    mask_resized = cv2.resize(mask_crop, (x2-x1, y2-y1))
    masks[i, y1:y2, x1:x2] = mask_resized > 0.5
```

We resize **only the bounding box region** to the original frame dimensions, not the full 80×80 mask. On 1080p / 4K streams this is the difference between 0.5 ms and 50 ms per frame — full-frame upsample dominates wall-clock.

### 9.2 RF-DETR masks — per-query, no factorisation

RF-DETR emits `[N_queries, mask_h, mask_w]` directly — one full mask per query. No prototype reconstruction needed:

```python
gathered = mask_preds[query_idx]   # pull only the kept queries
gathered = sigmoid(gathered)
```

Because RF-DETR uses stretch resize (no letterbox), proto coordinates map to the original frame by a simple scale — no `pad` term:

```python
sx = mw / orig_w
sy = mh / orig_h
```

The bbox-confined upsample trick still applies for performance.

### 9.3 dtype: bool, not uint8

`supervision.MaskAnnotator` expects `mask.dtype == bool`. If you write `np.zeros((n, H, W), dtype=np.uint8)` and then assign 0/1, the annotator silently does nothing or renders incorrect overlays. Always:

```python
masks = np.zeros((n, orig_h, orig_w), dtype=bool)
masks[i, y1:y2, x1:x2] = mask_resized > 0.5
```

This was bug #4 in our debugging story — masks were "produced" (non-zero pixel count), but the annotator got dtype-confused and skipped them.

---

## 10. The `compress_to_fp16` Trap

**`ov.convert_model()` defaults to compressing all FP32 weights to FP16.** This is the silent default that breaks transformer models.

```python
ov_model = ov.convert_model("model.onnx")        # compress_to_fp16=True (default!)
ov.save_model(ov_model, "openvino/model.xml")    # bin file is half-size

# vs.
ov_model = ov.convert_model("model.onnx")
ov.save_model(ov_model, "openvino/model.xml", compress_to_fp16=False)   # bin file FP32
```

### Why FP16 is fine for CNNs

YOLO is convolution-heavy. Conv weights are statistically close to zero with small dynamic range — FP16 representation loses < 0.5% mAP. The 50% size reduction is worth it.

### Why FP16 breaks transformers

Attention layers compute `softmax(Q @ K^T / sqrt(d))`. The dot product accumulates over the entire feature dimension, so FP16 weights compound rounding error. Layer norms with very small or very large activations get distorted. Linear projections in transformer blocks are particularly sensitive.

For RF-DETR specifically, FP16 weight compression shifts the logit distribution by ~5-10 units, which after sigmoid produces near-zero probabilities everywhere → no detections.

### Diagnosing it

If the same `.onnx` file works through ONNX Runtime but the converted IR returns no detections (or very low confidences), suspect FP16 compression first. Re-export with `compress_to_fp16=False` and check.

For our RF-DETR investigation, even FP32 didn't fix it (see §11), so the issue ran deeper than just precision — but FP16 should always be your **first** suspect on transformer models.

---

## 11. RF-DETR + OpenVINO: A Real-World Compatibility Wall

This section documents a **negative result**. Save yourself days by reading it.

### What we observed

- `inference_model.sim.onnx` runs correctly via ONNX Runtime (CPU and CUDA): 0.96 confidence on a polybag image, mask renders correctly.
- The same `.onnx` converted to OpenVINO IR (any combination of FP16/FP32, simplified/unsimplified): zero detections, max sigmoid 0.12.

### What we tried

| Attempt | Result |
|---|---|
| `ov.convert_model(sim.onnx)` (default FP16) | 0 detections |
| `ov.convert_model(sim.onnx)` + `compress_to_fp16=False` | 0 detections |
| `ov.convert_model(unsim.onnx)` (raw, before onnx-simplifier) | 0 detections |
| Compare raw output tensors to ONNX | Logits differ by `|Δ| up to 8.8` — far past noise |

### Likely causes

1. **`Einsum("bchw,bnc->bnhw")`** in the segmentation head. OpenVINO has historically partial Einsum support — some patterns translate cleanly, exotic equation strings fall back to slow generic paths or get mistranslated.
2. **DINOv2 attention layers**. OpenVINO 2026 still has incomplete coverage for some MultiHeadAttention patterns when shapes are dynamic.
3. **Dynamic shape resolution** in the decoder when `num_queries=100` interacts with `nc=91`.

### What to do about it

- **Don't ship RF-DETR via OpenVINO.** Today, on OpenVINO 2026.0, it's broken at the conversion level. No amount of postprocess code can recover it.
- **For CPU deployment, use YOLO + OpenVINO.** Validated, fast, correct.
- **For RF-DETR on CPU (slow but correct), use ONNX Runtime CPU EP.** ~2 FPS on a modern laptop. Acceptable as a fallback, never as primary.
- **For RF-DETR with real-time needs, use a GPU host.** The transformer backbone wants CUDA kernels.

We left the (correct) RF-DETR postprocess in `openvino_inferencer.py` for the day OpenVINO fixes the conversion, but added a doc warning so operators don't waste a day chasing it.

---

## 12. Verifying Correctness Against ONNX

Treat your ONNX inferencer as **ground truth** and your OpenVINO inferencer as the candidate that needs to match it. The structural skeleton:

```python
img = cv2.imread('test_image.jpg')

ov_det = ov_inferencer.predict_frame(img)
onnx_det = onnx_inferencer.predict_frame(img)

assert len(ov_det) == len(onnx_det), \
    f"Detection count mismatch: OV={len(ov_det)} ONNX={len(onnx_det)}"
assert (ov_det.class_id == onnx_det.class_id).all(), \
    f"Class IDs differ: {ov_det.class_id} vs {onnx_det.class_id}"
assert np.allclose(ov_det.confidence, onnx_det.confidence, atol=0.01), \
    f"Confidences diverge: {ov_det.confidence} vs {onnx_det.confidence}"
if ov_det.mask is not None and onnx_det.mask is not None:
    iou = (ov_det.mask & onnx_det.mask).sum() / (ov_det.mask | onnx_det.mask).sum()
    assert iou > 0.95, f"Mask IoU = {iou:.3f}"
```

Run this on a handful of representative frames before declaring an OpenVINO build production-ready. If it fails, the diff between OV and ONNX outputs at the **raw tensor level** (before postprocessing) will tell you whether the issue is in conversion (logits diverge) or postprocessing (logits match, decoded results don't).

---

## 13. Case Studies: The Bugs We Squashed

Real bugs from the OpenVINO debugging session, with the lesson each one teaches.

### Bug #1 — Stretch resize instead of letterbox

**Symptom:** Polybags labelled as cartons, confidence ~0.6 (vs 0.95 on ONNX), masks in wrong region. FPS reasonable but detections useless.
**Cause:** `cv2.resize(frame, (model_w, model_h))` distorts aspect ratio; model never trained on stretch inputs.
**Fix:** Implement `_letterbox()` matching the ONNX inferencer's preprocessing (§5.1).
**Lesson:** Inference preprocessing must mirror training preprocessing **exactly**. If your inferencer files have different `preprocess()` methods for the same model family, one of them is wrong.

### Bug #2 — Post-NMS columns parsed as per-class scores

**Symptom:** 300 spurious bounding boxes per frame, FPS crater to 0.2 from rendering load.
**Cause:** Code did `argmax(preds[:, 4:4+nc], axis=1)` on a post-NMS export. Cols 4–5 were `[confidence, class_id]`, not per-class scores. `argmax([conf, class])` is garbage.
**Fix:** Detect post-NMS layout (anchor axis ≤ 1000) and parse `confidences = preds[:, 4]; class_ids = preds[:, 5]` directly (§7.1).
**Lesson:** Ultralytics has two ONNX export layouts. Always detect which one you have at runtime — don't assume.

### Bug #3 — RF-DETR background class included in argmax

**Symptom:** Every detection labelled class 0 (which app interprets as something nonsensical).
**Cause:** `np.argmax(probs, axis=1)` over all 91 logits picks index 0 (background) almost always, because background dominates for non-detected queries.
**Fix:** Slice `logits[:, 1:1+nc]` before argmax/topk (§8.1).
**Lesson:** DETR-family models reserve index 0 for background. Always slice it off before any class operation.

### Bug #4 — Mask dtype `uint8` instead of `bool`

**Symptom:** Inferencer reports masks with positive pixel counts, but `MaskAnnotator` renders nothing on the frame.
**Cause:** `supervision.MaskAnnotator.annotate()` checks `mask.dtype == bool` and silently skips other dtypes.
**Fix:** Allocate `np.zeros(..., dtype=bool)` and assign comparison result `mask_resized > 0.5`.
**Lesson:** Read your downstream library's expected dtypes. Silent skipping is the worst kind of failure mode.

### Bug #5 — Naive proto-grid mask scaling without letterbox inverse

**Symptom:** Even when masks were generated correctly, they appeared offset from the bounding boxes (especially on non-square sources where padding matters).
**Cause:** Mask post-processing used `sx = proto_w / orig_w` (stretch mapping) instead of `sx = proto_w / model_w` with letterbox inverse on box coords first.
**Fix:** Map original → letterboxed model → proto coords properly (§9.1).
**Lesson:** When two coordinate spaces are connected by a non-trivial transform, **always** route through the canonical intermediate (model space) — never compose with shortcuts.

### Bug #6 — Per-query argmax instead of top-K over (queries × classes) for RF-DETR

**Symptom:** RF-DETR detections were noisy and ranked oddly compared to the .pth path.
**Cause:** Naive `argmax(probs, axis=1)` per query. Native RF-DETR picks top 300 (query, class) pairs across the flat product.
**Fix:** Implement the flatten + argpartition + topk pattern (§8.2).
**Lesson:** Read the **native** post-processing in the upstream library before writing your own. Subtle differences ("each query picks its best class" vs "300 best query-class pairs") drastically change result distributions.

### Bug #7 — Default FP16 weight compression breaks transformers

**Symptom:** OpenVINO RF-DETR returned 0 detections despite ONNX working.
**Cause:** `ov.convert_model()` defaults to `compress_to_fp16=True`. RF-DETR's attention layers are too sensitive.
**Fix attempt:** Pass `compress_to_fp16=False`. (Didn't actually fix RF-DETR, but did diagnose the layer of the problem.)
**Lesson:** The OpenVINO conversion API has a sneaky default. Always pin it explicitly for transformer models even if you "want" FP16.

### Bug #8 — Stale Docker image after code edit

**Symptom:** Code change ostensibly applied, but `docker compose restart` showed no behaviour change.
**Cause:** `src/` is baked into the image at build time, not volume-mounted. `restart` reuses the old image.
**Fix:** `docker compose up --build -d` or add `./src:/opt/isitec/src` as a volume during dev.
**Lesson:** Know which files are volume-mounted vs baked. Production wants baked (reproducible); dev wants mounted (fast iteration).

### Bug #9 — Trying to fix RF-DETR + OpenVINO conversion in postprocess code

**Symptom:** Multiple attempts at fixing RF-DETR postprocessing didn't help.
**Cause:** The conversion itself was broken. Logits diverged from ONNX by `|Δ| up to 8.8` — far past anything postprocessing could repair.
**Fix:** Stop trying. Document the limitation. Route RF-DETR through ONNX RT instead of OpenVINO on CPU.
**Lesson:** **When the raw output tensors diverge, no amount of postprocessing will save you.** Always compare raw outputs first; only debug postprocessing if raw outputs match.

---

## 14. Performance Notes

### Where time goes per frame

| Stage | YOLO 320×320 OpenVINO CPU | RF-DETR 416×416 ONNX CPU |
|---|---|---|
| Preprocessing (letterbox + normalize + transpose) | 1–2 ms | 1–2 ms |
| OpenVINO `compiled([inp])` call | 12–18 ms | 200+ ms (transformer) |
| Postprocess + mask decode (1 detection) | 0.5–2 ms | 1–3 ms |
| Tracker + line-crossing logic | < 1 ms | < 1 ms |
| Annotate (mask + box + label + line) | 2–4 ms | 2–4 ms |
| JPEG encode for MJPEG | 3–8 ms | 3–8 ms |
| **Total wall-clock** | **20–35 ms (28–50 FPS)** | **210–250 ms (4–5 FPS)** |

### Wins (in priority order)

1. **Use OpenVINO instead of ONNX CPU on Intel hardware.** 1.4–2× free speedup.
2. **Smaller input size.** 320 vs 640 is 4× fewer pixels and roughly 4× faster — accuracy drop on large objects (cartons / polybags) is negligible.
3. **YOLO over RF-DETR on CPU.** Order-of-magnitude difference.
4. **INT8 quantisation via NNCF.** Another 1.5–2× on Intel CPUs (but adds calibration step). See `Compression/` tool.
5. **Bbox-region-only mask upsample.** 10–50× cheaper than full-frame upsample on high-res sources.

### Things that don't help much

- **Multi-threading at the Python level.** OpenVINO already parallelises internally. Wrapping `compiled()` in a thread pool just adds GIL contention.
- **Async inference** (`compiled.start_async`). Useful for batching, irrelevant for our 1-frame-at-a-time live stream.
- **`AUTO` device with iGPU on integrated graphics laptops.** The iGPU is usually slower than the CPU on small models; the kernel-launch overhead dominates.

---

## 15. A Minimal End-to-End Skeleton

```python
import cv2
import numpy as np
import supervision as sv
import openvino as ov

class MinimalOpenVINOInferencer:
    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    _IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __init__(self, xml_path, conf_threshold=0.5, num_classes=2, device="CPU"):
        core = ov.Core()
        self.compiled = core.compile_model(core.read_model(xml_path), device)
        self.input = self.compiled.input(0)
        self.model_h, self.model_w = self.input.shape[2], self.input.shape[3]
        self.output_names = [self.compiled.output(i).get_any_name()
                             for i in range(len(self.compiled.outputs))]
        self.is_rfdetr = any(n in self.output_names
                              for n in ('dets', 'pred_logits', 'bboxes', 'labels'))
        self.conf_threshold = conf_threshold
        self.nc = num_classes

    # ----- preprocessing -----
    def _letterbox(self, frame, pad=114):
        h, w = frame.shape[:2]
        r = min(self.model_h / h, self.model_w / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        ph, pw = self.model_h - nh, self.model_w - nw
        top, left = ph // 2, pw // 2
        padded = cv2.copyMakeBorder(resized, top, ph - top, left, pw - left,
                                     cv2.BORDER_CONSTANT, value=(pad,) * 3)
        return padded, r, left, top

    def preprocess(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.is_rfdetr:
            img = cv2.resize(rgb, (self.model_w, self.model_h)).transpose(2, 0, 1)
            img = img.astype(np.float32) / 255.0
            img = (img - self._IMAGENET_MEAN) / self._IMAGENET_STD
            self._lb = None
        else:
            padded, r, px, py = self._letterbox(rgb)
            img = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
            self._lb = (r, px, py)
        return np.ascontiguousarray(img[np.newaxis, ...])

    # ----- inference -----
    def predict_frame(self, frame):
        out = self.compiled([self.preprocess(frame)])
        outs = [out[self.compiled.output(i)] for i in range(len(self.compiled.outputs))]
        oh, ow = frame.shape[:2]
        return self._post_yolo(outs, ow, oh) if not self.is_rfdetr else self._post_detr(outs, ow, oh)

    # ----- YOLO post-NMS only (skeleton) -----
    def _post_yolo(self, outs, ow, oh):
        preds = outs[0][0]
        if preds.shape[0] == 0: return sv.Detections.empty()
        boxes, conf, cls = preds[:, :4].copy(), preds[:, 4], preds[:, 5].astype(int)
        keep = conf > self.conf_threshold
        boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
        r, px, py = self._lb
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / r
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh)
        return sv.Detections(xyxy=boxes, confidence=conf.astype(np.float32), class_id=cls)

    # ----- RF-DETR (skeleton, no masks) -----
    def _post_detr(self, outs, ow, oh):
        out_map = dict(zip(self.output_names, outs))
        bboxes = out_map.get('dets', out_map.get('pred_boxes'))[0]
        logits = out_map.get('labels', out_map.get('pred_logits'))[0]
        head_nc = logits.shape[-1]
        nc = min(self.nc, head_nc - 1)
        logits = logits[:, 1:1 + nc]
        probs = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        flat = probs.reshape(-1)
        k = min(300, flat.size)
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[np.argsort(-flat[idx])]
        scores = flat[idx]
        keep = scores > self.conf_threshold
        idx, scores = idx[keep], scores[keep]
        q, c = idx // nc, (idx % nc) + 1
        b = bboxes[q]
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        boxes = np.column_stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        boxes[:, [0, 2]] *= ow
        boxes[:, [1, 3]] *= oh
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh)
        return sv.Detections(xyxy=boxes, confidence=scores.astype(np.float32), class_id=c.astype(int))


# Usage:
inf = MinimalOpenVINOInferencer("openvino/model.xml", conf_threshold=0.5, num_classes=2)
img = cv2.imread("test.jpg")
det = inf.predict_frame(img)
print(f"{len(det)} detections, classes={det.class_id}, confs={det.confidence}")
```

This is the same architecture as the production `OpenVINOInferencer` minus mask decoding and error handling. Drop in mask code from §9.1 / §9.2 to add segmentation.

---

## 16. Debugging Checklist

### Tier 1 — does the IR even load?

```python
core = ov.Core()
model = core.read_model("model.xml")     # exception → bin file missing or xml malformed
print("Inputs:",  [(i.get_any_name(), i.shape) for i in model.inputs])
print("Outputs:", [(o.get_any_name(), o.shape) for o in model.outputs])
```

If this fails: regenerate the IR via `ov.convert_model(...)`, check `.xml` and `.bin` are in the same directory.

### Tier 2 — do raw outputs match ONNX?

```python
# Same numpy input through both runtimes
ov_out = compiled([inp])[compiled.output(0)]
onnx_out = onnx_session.run(None, {input_name: inp})[0]
print("Max abs diff:", np.abs(ov_out - onnx_out).max())
```

If diff > 0.1: conversion is broken. Try `compress_to_fp16=False`. If still bad, the model has ops OpenVINO mishandles (transformer attention, exotic Einsum) — see §11.

### Tier 3 — does postprocessing produce valid detections?

```python
det = inferencer.predict_frame(img)
assert len(det) > 0, "no detections — preprocessing or threshold issue"
assert (det.confidence > 0).all(), "garbage confidences — wrong column parsed"
assert (det.class_id >= 0).all() and (det.class_id < num_classes + 1).all(), \
    "class IDs out of range — background not sliced (RF-DETR) or wrong column (YOLO)"
```

### Tier 4 — do detections look right on the frame?

Render boxes manually and visually check:

```python
for box in det.xyxy:
    cv2.rectangle(img, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), (0, 255, 0), 2)
cv2.imwrite("debug.jpg", img)
```

If boxes are in wrong positions: letterbox inverse is wrong (§7.4) or box format mis-decoded (xyxy vs cxcywh).

### Tier 5 — do masks render?

```python
print(f"Mask is None: {det.mask is None}")
if det.mask is not None:
    print(f"  shape: {det.mask.shape}, dtype: {det.mask.dtype}, nonzero: {det.mask.sum()}")
```

If mask is `None`: prototype factorisation failed silently. Wrap `_process_masks` in try/except and log.
If mask has nonzero pixels but doesn't render: dtype isn't `bool` (Bug #4).
If mask renders in wrong place: proto-coordinate mapping is wrong (§9.1).

---

## 17. Further Reading

- [OpenVINO docs — model conversion](https://docs.openvino.ai/2026/openvino-workflow/model-preparation.html)
- [OpenVINO docs — operation set support](https://docs.openvino.ai/2026/documentation/openvino-ir-format/operation-sets.html) — check before converting models with exotic ops
- [Ultralytics export reference](https://docs.ultralytics.com/modes/export/) — what `nms=True` actually does
- [RF-DETR repo](https://github.com/roboflow/rf-detr) — for native postprocessing reference
- [NNCF](https://github.com/openvinotoolkit/nncf) — for INT8 post-training quantisation when you need more speed
- The repo's `Compression/stages/int8_qdq.py` — for examples of OpenVINO-targeted quantisation already wired into our tooling

---

## Summary

OpenVINO is a powerful CPU inference path with one main pitfall: silent correctness failures from preprocessing mismatches and conversion-time defaults. The mental model to keep:

- **`ov.convert_model` is doing more than you think** — graph optimisation, op translation, FP16 compression, fusion. Verify raw outputs against ONNX before trusting it.
- **Preprocessing is part of the model.** Letterbox for YOLO, stretch+ImageNet for RF-DETR. There is no "generic" preprocessing.
- **Detect output layouts at runtime** — Ultralytics has two YOLO formats; DETR has 91-class heads with background reserved.
- **Compare to ONNX as ground truth.** When raw tensors diverge, you have a conversion bug. When they match but final detections don't, you have a postprocessing bug.
- **Some models just don't work.** RF-DETR + OpenVINO 2026 is one of them. Document the limit, ship YOLO + OpenVINO, route RF-DETR through ONNX RT or GPU.

If a future you (or teammate) hits the "0.2 FPS with 300 garbage boxes" symptom — read §13 Bug #2 first. That's the same mistake we made.

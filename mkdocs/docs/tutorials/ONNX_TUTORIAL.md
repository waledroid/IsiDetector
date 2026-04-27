# Working with ONNX Models for Computer Vision Inference

A practical walk-through of how to take a generated ONNX file, understand what's inside it, and build a correct, fast inference engine around it. Illustrated with the real debugging story of `isidet/src/inference/onnx_inferencer.py` in this repo.

You'll learn:
- What an ONNX file actually is
- How to dissect one and what to look for
- How to classify the model family from the graph alone
- How to build a session, preprocess, postprocess, and decode masks
- The coordinate-system math that makes boxes line up
- How to read the training code to match the exact preprocessing recipe
- The nine bugs we fixed to get yolov26 and RF-DETR working, and what each one teaches

Each section starts with the high-level idea, then drops into the actual code, math, or runtime trace.

---

## 1. What Is an ONNX File, Really?

**ONNX** (Open Neural Network Exchange) is a framework-agnostic container for three things:

1. **A computational graph** — a DAG of operators (Conv, MatMul, Sigmoid, Reshape, etc.) linked by named tensors.
2. **Weights** — the numeric parameters baked into those operators.
3. **Metadata** — producer name, opset version, optional custom key/value pairs the exporter stamps.

It is a `.onnx` file written in Protocol Buffers format. You hand it to an *execution engine* (we use ONNX Runtime / `onnxruntime`) and it runs. The engine decides *how* to execute — on CPU, CUDA, TensorRT, OpenVINO — while the file itself is pure declaration.

Mental model:

```
┌───────────────────────────────────────────────┐
│  .onnx file                                   │
│  ┌────────────────────────────────────────┐   │
│  │  graph:  input → [Conv → BN → ReLU]*N  │   │
│  │          → ... → output tensors        │   │
│  │  weights:  W_1, W_2, ..., W_k          │   │
│  │  metadata: {producer: "pytorch", ...}  │   │
│  └────────────────────────────────────────┘   │
└───────────────────────────────────────────────┘
         │
         ▼ loaded by
┌───────────────────────────────────────────────┐
│  ONNX Runtime InferenceSession                │
│  chooses execution provider (CUDA / CPU / ...)│
│  optimises graph                              │
│  allocates tensors                            │
└───────────────────────────────────────────────┘
```

### Tools you'll use

| Tool | Purpose |
|---|---|
| [Netron](https://netron.app) | Visual graph browser — drag a `.onnx` file in a browser |
| `onnxruntime.InferenceSession` | Runtime introspection + execution |
| `onnx` Python package | Static graph manipulation (read, modify, stamp metadata, save) |
| `onnxsim` / `onnxslim` | Graph simplification (constant folding, dead-op removal) |

Install what's needed:
```bash
pip install onnxruntime-gpu onnx onnxsim   # or onnxruntime for CPU-only
```

---

## 2. Dissecting an ONNX File: The 30-Second Interrogation

Before you write a single line of inference code, load the model and ask it what it wants.

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

print("Providers chosen:", session.get_providers())

for inp in session.get_inputs():
    print(f"INPUT  name={inp.name!r}  shape={inp.shape}  dtype={inp.type}")

for out in session.get_outputs():
    print(f"OUTPUT name={out.name!r}  shape={out.shape}  dtype={out.type}")

meta = session.get_modelmeta()
print("Producer:", meta.producer_name)
print("Custom metadata:", meta.custom_metadata_map)
```

Typical output from our yolov26-seg:

```
INPUT  name='images'   shape=[1, 3, 416, 416]  dtype=tensor(float)
OUTPUT name='output0'  shape=[1, 300, 38]       dtype=tensor(float)
OUTPUT name='output1'  shape=[1, 32, 104, 104]  dtype=tensor(float)
```

From RF-DETR:

```
INPUT  name='input'        shape=[1, 3, 432, 432]  dtype=tensor(float)
OUTPUT name='dets'         shape=[1, 200, 4]       dtype=tensor(float)
OUTPUT name='labels'       shape=[1, 200, 91]      dtype=tensor(float)
OUTPUT name='masks'        shape=[1, 200, 108, 108] dtype=tensor(float)
```

Four things to read out of this every time:

### 2.1 Input name
You must pass exactly this key in `session.run()`. If the exporter called it `"images"` and you pass `{"input": ...}`, you get a KeyError.

### 2.2 Input shape
- **Static dims** (concrete integers like `416`) are fixed — your preprocess must produce that exact shape.
- **Dynamic dims** (`None`, `-1`, or strings like `"N"`, `"height"`) are placeholders you can set at runtime. Most exports pin H/W and leave batch dynamic.

### 2.3 Input dtype
- `tensor(float)` = float32 — most common.
- `tensor(float16)` = FP16 — for half-precision exports; your feed tensor must also be FP16 or you'll get a type error.
- `tensor(uint8)` = raw pixel input; model does its own normalisation inside.

### 2.4 Output names, shapes, dtypes
These are your Rosetta Stone for identifying the model family. More in §4.

### 2.5 Metadata (often empty — sadly)
Ultralytics, PyTorch, and most exporters don't stamp anything useful. RF-DETR doesn't either. If you *own* the export pipeline, this is the single cheapest upgrade you can make — stamp `model_family`, `imgsz`, `class_names`, and the inferencer never has to guess.

Example stamp (do this once, at export time):
```python
import onnx, json
model = onnx.load("model.onnx")
for k, v in {
    "model_family": "yolo_nms",
    "imgsz": "416",
    "class_names": json.dumps(["carton", "polybag"]),
}.items():
    entry = model.metadata_props.add()
    entry.key = k
    entry.value = v
onnx.save(model, "model.onnx")
```

Read back at load time: `session.get_modelmeta().custom_metadata_map`.

---

## 3. Classifying the Model Family from the Graph Alone

When the exporter stamps nothing, you have to reverse-engineer the family from the output layout. This is the single most important judgment call — get it wrong and every downstream step fails silently.

### 3.1 YOLO with NMS baked in (post-NMS)
Exported with Ultralytics `yolo export ... nms=True`.

```
outputs: 1 (det)  or  2 (seg)
output[0] shape: [1, N, 6]      detection
                 [1, N, 6+32]   segmentation
                 N is typically 300 (top-K after NMS)
output[1] shape: [1, 32, Ph, Pw]  only for segmentation (mask prototypes)
```

Columns of `output[0]`: `[x1, y1, x2, y2, score, class_id, (mask_coeffs...)]`.

### 3.2 YOLO raw / NMS-free during inference (pre-NMS)
Exported with `nms=False`, or older exports.

```
output[0] shape: [1, 4+nc, A]       (detection)
                 [1, 4+nc+32, A]    (segmentation)
                 or transposed: [1, A, 4+nc(+32)]
                 A is the anchor/proposal count: 8400 for 640 input, ~3549 for 416
```

Columns: `[cx, cy, w, h, score_0, score_1, ..., score_{nc-1}, (mask_coeffs...)]`. No single `class` column — per-class scores.

### 3.3 DETR family (including RF-DETR)
Exported from a transformer-based architecture.

```
outputs: 2 (det)  or  3 (seg)
names vary by exporter:
  - {"dets", "labels", ("masks")}              (RF-DETR)
  - {"pred_boxes", "pred_logits", "pred_masks"} (canonical DETR)
boxes shape:  [1, Q, 4]   where Q is the fixed number of object queries
logits shape: [1, Q, C]   C is class count (often 91 from COCO pretraining)
masks shape:  [1, Q, Mh, Mw]
```

### 3.4 Heuristic used in our code (src/inference/onnx_inferencer.py)

```python
self.is_rfdetr = any(n in self.output_names for n in
    ('dets', 'pred_logits', 'pred_boxes', 'bboxes', 'labels'))
```

And at postprocess time for YOLO:
```python
raw = outputs[0][0]
s0, s1 = raw.shape[0], raw.shape[1]
is_raw = max(s0, s1) > 1000    # >1000 = anchor axis, pre-NMS
```

The ">1000" threshold distinguishes post-NMS (max 300) from raw anchors (thousands).

### 3.5 Distinguisher summary

| Signal | Post-NMS YOLO | Pre-NMS YOLO | DETR |
|---|---|---|---|
| Number of outputs | 1 or 2 | 1 or 2 | 2 or 3 |
| Output names | `output0`, `output1` | `output0` | `dets`/`pred_boxes`, `labels`/`pred_logits`, `masks` |
| Detection axis size | ≤300 | Thousands (anchors) | Fixed queries (100–500) |
| Score column | Single | Per-class | Per-class (logits) |
| Needs NMS in postproc | No | Yes | No (top-K) |
| Box format | xyxy in model pixels | cxcywh in model pixels | cxcywh normalised to 0–1 |

---

## 4. Building the Session

### 4.1 Provider selection

Execution providers (EPs) run the ops. Order matters — ONNX Runtime tries each in turn and uses the first that can execute an op. Put the fastest first, CPU last.

```python
# The robust recipe
available = ort.get_available_providers()
if 'CUDAExecutionProvider' in available:
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'HEURISTIC',
        }),
        'CPUExecutionProvider',      # fallback
    ]
else:
    providers = ['CPUExecutionProvider']

session = ort.InferenceSession("model.onnx", providers=providers)

# ALWAYS verify what actually got chosen — don't trust the request.
print("Actually using:", session.get_providers()[0])
```

### 4.2 The `onnxruntime` vs `onnxruntime-gpu` trap
If both packages are installed, the CPU one shadows the GPU one and CUDA silently disappears. See the auto-detect block at `isidet/src/inference/onnx_inferencer.py:24-49` for the workaround — it detects the conflict and uninstalls the CPU package.

### 4.3 Session options worth setting

```python
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.enable_mem_pattern = True          # reuse tensor buffers across runs
opts.enable_cpu_mem_arena = False       # avoid large upfront allocation
opts.log_severity_level = 3             # suppress runtime chatter
```

### 4.4 Warmup
First inference after session creation is slow (kernel compilation, autotuning). Burn a few dummy runs so real frames don't spike.

```python
dummy = np.random.randn(1, 3, 416, 416).astype(np.float32)
for _ in range(3):
    session.run(None, {input_name: dummy})
```

---

## 5. Preprocessing — Getting Pixels to Match the Training Recipe

This is where most silent bugs live. A model trained on RGB + letterbox + [0,1] normalization will happily accept BGR + stretch-resize + [0,255] input and return plausible-looking but wrong detections.

### 5.0 High-level: preprocessing is model-family-specific

Every trainer has an opinion about how pixels should arrive at the backbone. **You must replicate that opinion exactly**, or detections degrade silently (boxes still render, accuracy drops; or flat zero activations and no boxes at all).

| Family | Color | Resize | Range | Normalisation |
|---|---|---|---|---|
| Ultralytics YOLO | RGB | **letterbox** pad (grey 114) to `(imgsz, imgsz)` | `/255` → [0,1] | none |
| RF-DETR / DINOv2 | RGB | **stretch** to `(imgsz, imgsz)` (no pad) | `/255` → [0,1] | `(x - mean) / std`, ImageNet stats |
| Canonical DETR (ResNet) | RGB | longest-side resize w/ max cap | `/255` → [0,1] | ImageNet |
| Raw-pixel exports | depends | depends | none (uint8 → uint8) | baked into graph |

Why the variation? Each backbone has a different prior. YOLO's CSPDarknet was trained with letterbox so seeing grey pad is normal. DINOv2 (RF-DETR's backbone) was pretrained by self-supervision on naturally-resized ImageNet crops, so it expects full-frame content with ImageNet statistics. Feeding letterboxed grey to DINOv2 makes 44% of the pixels statistically look like "no information" — the transformer attention heads barely activate.

The rule: **read the training code once, and cache the recipe.** Section 5.4 shows how to extract it from a library in 30 seconds.

### 5.1 Color space (BGR ↔ RGB)

**Opencv returns BGR. Training frameworks use RGB.** Always check.

| Source | Channel order |
|---|---|
| `cv2.imread`, `cv2.VideoCapture`, RTSP via OpenCV | BGR |
| PIL `Image.open` | RGB |
| Ultralytics, Torchvision, RF-DETR training | RGB |

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

This single missing line was a bug in our code — it caused degraded accuracy for every ONNX model even when the rest was correct.

### 5.2 Letterbox — the aspect-preserving pad

The model wants a fixed input (e.g. 416×416). The camera gives you 1920×1080. You have two options:
- **Stretch** — `cv2.resize(frame, (416, 416))`. Fast, one line, wrong. Distorts objects; boxes come back skewed.
- **Letterbox** — scale by the smaller ratio, pad the rest. Preserves aspect ratio.

The math:
```
r = min(model_h / orig_h, model_w / orig_w)
new_w = round(orig_w * r)
new_h = round(orig_h * r)
pad_w = model_w - new_w
pad_h = model_h - new_h
pad_left = pad_w // 2
pad_top  = pad_h // 2
```

Implementation:
```python
resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
padded  = cv2.copyMakeBorder(
    resized,
    top=pad_top, bottom=pad_h-pad_top, left=pad_left, right=pad_w-pad_left,
    borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114),    # YOLO's canonical grey
)
```

Visual:
```
  original (1920x1080)             letterboxed (416x416)
  ┌──────────────────┐            ┌───────────────┐
  │                  │            │░░░pad░░░░░░░░│
  │                  │            ├───────────────┤
  │    content       │   ────►    │   content     │
  │                  │            ├───────────────┤
  │                  │            │░░░pad░░░░░░░░│
  └──────────────────┘            └───────────────┘
```

**Crucially, save `(ratio, pad_left, pad_top)`** — you need them to invert the transform on the output boxes.

### 5.3 Normalisation + layout

For most YOLO/DETR exports the layout is the same — `NCHW` float32, contiguous:

```python
img = padded.astype(np.float32) / 255.0         # 0-255 uint8 → 0-1 float
img = img.transpose(2, 0, 1)                    # HWC → CHW
img = img[np.newaxis, ...]                      # add batch dim → NCHW
img = np.ascontiguousarray(img)                 # ensure stride-contiguous memory
```

What differs is whether to *also* subtract a per-channel mean and divide by std.

**ImageNet normalisation** (required by DETR family, DINOv2 backbones, almost any transformer-based CV model trained with torchvision's standard transforms):
```python
mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
std  = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
img = (img - mean) / std        # after /255 and transpose to CHW
```

The result: `img.min() ≈ -2.1`, `img.max() ≈ 2.6` — centred near zero, with unit-ish variance. This is what the backbone's first Conv layer was trained to see.

**How to tell which case you're in:**
1. Look at `session.get_inputs()[0].type`. If `tensor(uint8)` → normalisation is baked into the graph, feed raw pixels. If `tensor(float)` → you apply it externally.
2. Read the training code (next section). Grep for `mean`, `std`, `Normalize`.
3. A/B test: run the model on one frame with and without ImageNet norm. The right one gives a sigmoid > 0.5 somewhere. The wrong one flatlines at < 0.1. See `a.py` in the repo for the exact diagnostic script we used.

### 5.4 Extracting the recipe from the training library

When docs are thin — and for DETR-family projects they usually are — the fastest way to learn the preprocessing is to grep the library source inside the container that has it installed:

```bash
docker compose exec <service> python -c "
import rfdetr, os, glob    # substitute your lib
for p in glob.glob(os.path.dirname(rfdetr.__file__) + '/**/*.py', recursive=True):
    src = open(p).read()
    if '0.485' in src or 'IMAGENET' in src.upper() or 'Normalize' in src:
        print('---', p)
        for ln in src.splitlines():
            if any(k in ln for k in ('0.485','mean','std','Normalize','Resize','to_tensor')):
                print(ln.strip()[:160])
"
```

What we discovered doing this for rfdetr (`detr.py:620-700`):
```python
img = F.to_tensor(img)                                    # PIL → [0,1] CHW
img_tensor = F.resize(img_tensor, (res, res))             # stretch, bilinear
img_tensor = F.normalize(img_tensor, self.means, self.stds)   # ImageNet
```

Three lines. Reproducing them in numpy + OpenCV is mechanical. The trick is *finding* them — once you see them, the rest is typing.

### 5.5 The A/B methodology for preprocessing doubts

When you suspect preprocessing is wrong but don't know which step, don't guess — run a deterministic comparison:

1. Grab one real frame from your actual input source.
2. Build variant A (your current preprocessing), variant B (candidate fix), etc.
3. Feed each through `session.run()`.
4. Print the **max sigmoid score** for each variant.
5. The correct preprocessing gives a clearly higher top score than the others.

A variant hitting `0.97+` is correct. A variant stuck at `0.08` is wrong. A variant at `0.3` is partially right (often one channel ordered wrong, or a small scale mismatch).

Script pattern is in `a.py`. Takes 15 seconds per iteration once set up.

---

## 6. Running Inference

```python
outputs = session.run(None, {input_name: img})
```

- First arg `None` means "return all outputs". You can pass a list of names to restrict.
- Second arg is the feed dict keyed by input tensor names.
- `outputs` is a list of numpy arrays, one per graph output, in the same order as `session.get_outputs()`.

That's it — the hardest part of ONNX inference is never the `run()` call. It's everything either side of it.

---

## 7. Postprocessing — YOLO Post-NMS (yolov26, Ultralytics `nms=True`)

### 7.1 Input to this step

```
outputs[0].shape = (1, 300, 38)    # 4 box + 1 score + 1 class + 32 mask_coeffs
outputs[1].shape = (1, 32, 104, 104)   # mask prototypes
```

### 7.2 Steps
```python
preds = outputs[0][0]               # drop batch → (300, 38)

boxes       = preds[:, :4].copy()   # xyxy in MODEL pixel space (letterboxed)
confidences = preds[:, 4]
class_ids   = preds[:, 5].astype(int)
mask_coeffs = preds[:, 6:]          # (300, 32)

# 1. Confidence threshold (post-NMS output still has padding rows with score≈0)
keep = confidences > self.conf_threshold
boxes, confidences, class_ids, mask_coeffs = \
    boxes[keep], confidences[keep], class_ids[keep], mask_coeffs[keep]

# 2. Invert letterbox → put boxes back in original frame coordinates
#    Forward was:  model_x = orig_x * r + pad_x
#    Inverse is:   orig_x  = (model_x - pad_x) / r
boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio

# 3. Clip to frame (detections partially outside the image are common)
boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

# 4. Drop degenerate boxes (width or height ≤ 1)
valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
boxes, confidences, class_ids, mask_coeffs = \
    boxes[valid], confidences[valid], class_ids[valid], mask_coeffs[valid]
```

### 7.3 The bug we had
The original code treated columns `4:4+nc` as *per-class scores* and took `argmax` over them. For `nc=2`, that meant argmaxing `[score, class_id]` as if they were competing probabilities — `class_id=0` happened to survive; `class_id=1` produced nonsense confidences. The fix was recognising that post-NMS is single-score + single-class, not per-class.

**Rule of thumb**: if you see ≤300 detections and 2 suspicious columns after the 4 box columns, it's post-NMS. Treat them as `score, class`.

---

## 8. Postprocessing — YOLO Pre-NMS / Raw

### 8.1 Input
```
outputs[0].shape = (1, 4+nc+32, 3549)   # or transposed: (1, 3549, 4+nc+32)
outputs[1].shape = (1, 32, proto_h, proto_w)   # seg only
```

### 8.2 Steps
```python
raw = outputs[0][0]
# Normalise layout to (A, features) where A = anchor count
if raw.shape[0] < raw.shape[1]:
    raw = raw.T                 # was (features, A) → (A, features)

boxes_cxcywh = raw[:, :4]
class_scores = raw[:, 4:4+nc]
mask_coeffs  = raw[:, 4+nc:]    # seg only

# Per-anchor best class and score
confidences = class_scores.max(axis=1)
class_ids   = class_scores.argmax(axis=1)

# Confidence threshold first (kills 99% of anchors)
keep = confidences > conf_threshold
...

# cxcywh → xyxy
cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
xyxy = np.column_stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

# Class-aware NMS (cv2 is fine)
keep_idx = []
for cid in np.unique(class_ids):
    cls_mask = class_ids == cid
    kept = cv2.dnn.NMSBoxes(
        xywh_list, conf_list, conf_threshold, iou_threshold=0.45)
    keep_idx.extend(global_indices[kept])

# Then letterbox inverse + clip, same as post-NMS path.
```

Our code routes between post-NMS and raw by the heuristic `max(shape[0], shape[1]) > 1000`. It's heuristic because an exporter in theory could emit anything; in practice thousands vs hundreds is a clean split.

---

## 9. Postprocessing — DETR / RF-DETR

Fundamentally different from YOLO. No anchors, no NMS. Instead, a fixed set of *object queries* (200 in our RF-DETR export), each emitting one box and one class distribution. You pick the best (query, class) pairs.

### 9.0 High-level: the class-index gotcha

Before the math — there is **one trap** in DETR-family postprocessing that will silently kill every detection if you get it wrong: **class index 0 is not your first class**.

DETR heads inherit COCO's convention where **index 0 means "background" / "no object"**. The model is trained to drive that column's sigmoid near zero for every query. Your real fine-tuned classes begin at **index 1**.

Example from our model: a 91-slot head with `nc=2` fine-tuned classes lays out like:

```
column 0:    background              → sigmoid always ~ 0.001
column 1:    carton (your class 0)   → can reach 0.98 on a real carton
column 2:    polybag (your class 1)  → can reach 0.98 on a real polybag
columns 3-90: dead COCO slots        → residual pretraining noise
```

If you slice the logits as `logits[:, :nc]` you capture columns 0 and 1 — but column 0 is always dead. You are essentially only looking at class 0 signal, and missing class 1 entirely.

Correct slice: `logits[:, 1 : 1+nc]`. After slicing, column 0 of the view is carton (app class 0), column 1 is polybag (app class 1). Class IDs you emit are 0-indexed, so downstream code maps straight to `class_names`.

**How to verify on your own model** without guessing: run the A/B script (§5.5) but also print per-column max sigmoid across queries. Whatever columns light up to 0.9+ are your fine-tuned classes. If only column 0 is hot, training put classes at 0-indexed slots (rare but possible). If columns 1..nc are hot, standard COCO offset — use the `[:, 1:1+nc]` slice.

### 9.1 Input
```
outputs = {
    "dets":   (1, 200, 4)       # normalised cxcywh in model space
    "labels": (1, 200, 91)      # raw logits across 91 COCO-pretrained classes
    "masks":  (1, 200, 108, 108)   # per-query mask logits
}
```

Critical detail: RF-DETR inherits a 91-class head from COCO pretraining, but only `1..nc` are fine-tuned (see §9.0). Indices outside that range hold residual signal from pretraining and will fire as "person", "bicycle", etc. if you don't suppress them.

### 9.2 Steps — replicating the native `.pth` postproc

```python
bboxes = outputs["dets"][0]            # (200, 4)   normalised cxcywh
logits = outputs["labels"][0]          # (200, 91)
masks_raw = outputs["masks"][0]        # (200, 108, 108)

# 1. Skip background (index 0), slice out the fine-tuned classes
#    After the slice, col 0 == carton, col 1 == polybag (app-indexed).
logits = logits[:, 1 : 1 + self.nc]    # (200, 2)

# 2. Sigmoid (NOT softmax — DETR treats classes as independent)
probs = 1 / (1 + np.exp(-logits))      # (200, 2)

# 3. Top-K over flattened (query, class) matrix
#    This mirrors rfdetr's torch.topk in the native postproc:
flat = probs.reshape(-1)               # (200 * 2,)
num_select = min(300, flat.size)
topk_idx = np.argpartition(-flat, num_select - 1)[:num_select]
topk_idx = topk_idx[np.argsort(-flat[topk_idx])]   # sort desc

scores    = flat[topk_idx]
query_idx = topk_idx // self.nc        # which query emitted this
class_ids = topk_idx %  self.nc        # already 0-indexed to app classes

# 4. Threshold
keep = scores > self.conf_threshold
scores, query_idx, class_ids = scores[keep], query_idx[keep], class_ids[keep]

# 5. Gather boxes by query index
chosen = bboxes[query_idx]             # (K, 4) still normalised cxcywh
```

Why top-K across the full (query, class) matrix instead of `argmax` per query? Because in DETR a single query may carry two plausible interpretations ("this thing is either a carton or a polybag"), and we want both pairs considered on their own merits. Argmax-per-query throws away the second-best class score, even if it is above threshold. Native rfdetr uses top-K; we match.

### 9.3 Coordinate mapping — where DETR math differs from YOLO

DETR outputs **normalised** coordinates (range 0–1). The question is: normalised relative to *what*? That depends on what the model saw during training.

- **If training stretch-resizes** (rfdetr's case), normalised coords live in the stretched space, which shares fractional coordinates with the original frame. Multiplying by `(orig_w, orig_h)` lands directly in original pixels. **No letterbox inverse needed.**
- **If training letterboxes**, normalised coords are relative to the letterboxed model input. You must scale by `(model_w, model_h)` then subtract pad and divide by ratio — same as YOLO.

rfdetr's native postproc (`detr.py:699`) uses `target_sizes = (orig_h, orig_w)` and scales normalised boxes directly: `boxes = normalised * [orig_w, orig_h, orig_w, orig_h]`. So the model outputs are in stretched-space normalised coordinates → multiplying by original dimensions gives original pixels:

```python
# cxcywh → xyxy (still normalised)
cx, cy, w, h = chosen[:, 0], chosen[:, 1], chosen[:, 2], chosen[:, 3]
boxes = np.column_stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

# Scale directly to original frame — no letterbox inverse for rfdetr
boxes[:, [0, 2]] *= orig_w
boxes[:, [1, 3]] *= orig_h
boxes = boxes.clip(0, [orig_w, orig_h, orig_w, orig_h])
```

### 9.4 The bugs we fixed in this path

- **Argmax over all 91 classes** surfaced nonsense labels (bottle, dog) from COCO pretraining noise. Fixed by top-K over the sliced head.
- **Slicing `[:, :nc]` instead of `[:, 1:1+nc]`** stripped the real signal and kept the dead background column. Fixed once we printed per-column max sigmoid and saw column 0 was always ~0.001 while columns 1 and 2 were the hot ones.
- **Applying letterbox preprocess to RF-DETR** fed 44% grey pixels to a backbone that expects full-frame content → near-zero activations everywhere. Fixed by switching to stretch-resize and dropping the letterbox inverse from the postproc.
- **Missing ImageNet normalisation** made the backbone see `[0, 1]` input when it was trained on `(x - mean)/std`. Activations were muted, not dead — a partial bug masking the bigger one.

---

## 10. Mask Decoding

### 10.1 YOLO — prototype factorisation

Segmentation YOLO outputs two things:
- A per-detection *coefficient* vector (length 32).
- A single *prototype* tensor (32 spatial masks).

Decoding is a matrix multiply:

```
mask_i = sigmoid( Σ_j coeff_i[j] * proto[j] )
```

In numpy:
```python
coeffs = mask_coeffs         # (N, 32)
proto  = outputs[1][0]       # (32, Ph, Pw)

# (N, 32) @ (32, Ph*Pw) → (N, Ph*Pw) → (N, Ph, Pw)
masks_raw = coeffs @ proto.reshape(32, -1)
masks_raw = masks_raw.reshape(N, Ph, Pw)
masks_raw = 1 / (1 + np.exp(-masks_raw))    # sigmoid → 0..1 soft masks
```

This compresses per-detection mask storage from `H * W * N` down to `32 * N + 32 * Ph * Pw` — the trick that makes high-resolution YOLO-seg tractable.

### 10.2 Mapping mask space → original frame

The prototype is in **letterboxed model space, at reduced resolution** (e.g. 104×104 for a 416 model — a 4× downsample). To draw it on the original 1920×1080 frame:

```
proto space  →  letterboxed model space  →  original frame
  (Ph, Pw)        (model_h, model_w)          (orig_h, orig_w)
     │                     │                         │
  upscale             crop the pad            upscale to orig
```

Our code does this per detection, **bbox-confined**, because upscaling a full 104×104 mask to 1920×1080 for every one of 20 detections costs real time:

```python
sx = Pw / model_w          # orig→proto scale: model_x * sx
sy = Ph / model_h

for i in range(N):
    # Box i (already in original-frame coords)
    x1, y1, x2, y2 = boxes[i]

    # Map box to proto space
    px1 = int((x1 * ratio + pad_x) * sx)
    py1 = int((y1 * ratio + pad_y) * sy)
    px2 = int((x2 * ratio + pad_x) * sx)
    py2 = int((y2 * ratio + pad_y) * sy)

    # Crop mask to that region in proto space
    mask_crop = masks_raw[i, py1:py2, px1:px2]

    # Resize only that crop to the original bbox size
    bw = x2 - x1;  bh = y2 - y1
    mask_resized = cv2.resize(mask_crop, (bw, bh), interpolation=cv2.INTER_LINEAR)

    # Paste into a bool mask at original resolution
    full_mask[y1:y2, x1:x2] = mask_resized > 0.5
```

**Why bbox-confined.** A naive approach resizes the *whole* proto mask (104×104) to the original frame size (1920×1080), then crops to the bbox. That's `1920*1080 = 2M pixels` per mask × N detections. Confined: you only resize `bw * bh`, typically under 200×200 — 50× cheaper.

### 10.3 DETR — per-query masks (no factorisation)

Each query emits its own full mask directly. Simpler, bigger.

```python
gathered = mask_logits[query_idx]         # (K, Mh, Mw), selected queries only
probs    = 1 / (1 + np.exp(-gathered))    # sigmoid
```

Same letterbox-aware bbox-confined upscaling as §10.2 applies — the formulas `sx = Mw / model_w`, etc. — just with the DETR mask resolution (108×108 in our case) instead of the YOLO proto resolution.

### 10.4 dtype: bool, not uint8
`supervision`'s `MaskAnnotator` (and `sv.Detections.from_ultralytics`) expects masks as `np.bool_`. uint8 0/1 arrays will parse as `sv.Detections.mask` but silently fail to render. Bug #4 in our debugging saga.

```python
masks[i, y1:y2, x1:x2] = mask_resized > 0.5      # bool on RHS
masks = np.zeros((N, H, W), dtype=bool)          # bool on LHS
```

---

## 11. Coordinate System Transformations — Reference Sheet

Boxes live in at most three coordinate systems depending on the model. Know which you're in at every step.

### 11.1 Spaces

| Space | Units | Range |
|---|---|---|
| Proto / mask | Pixels of the mask prototype | [0, Ph) × [0, Pw) |
| Letterboxed model | Pixels of the model input (with grey pad) | [0, model_h) × [0, model_w) |
| Unpadded model | Pixels of model input minus pad | [0, new_h) × [0, new_w) |
| Original frame | Pixels of the camera frame | [0, orig_h) × [0, orig_w) |
| Normalised | Fraction of letterboxed model space | [0, 1] |

### 11.2 Transforms

```
Forward (preprocess):
    orig     --resize by r--> unpadded  --add pad--> letterboxed  --÷(W,H)--> normalised
    (ox, oy)               (ox*r, oy*r)           (ox*r+px, oy*r+py)     (.../W, .../H)

Inverse (postprocess):
    letterboxed            unpadded              orig
    (mx, my)  --sub pad-> (mx-px, my-py)  --÷r--> ((mx-px)/r, (my-py)/r)
```

### 11.3 Proto ↔ anywhere

Use scale factors `sx = Ph / model_h`, `sy = Pw / model_w`. Then:
```
proto_x = model_x * sx = (orig_x * r + px) * sx
```

This composition is what the bbox-confined mask code computes in §10.2.

---

## 12. Packaging the Output

We use `supervision.Detections` as the lingua franca:

```python
return sv.Detections(
    xyxy       = boxes,            # float32, shape (K, 4), in ORIGINAL pixel coords
    confidence = confidences,      # float32, shape (K,)
    class_id   = class_ids,        # int, shape (K,)
    mask       = masks,            # bool, shape (K, orig_h, orig_w) or None
)
```

- No `tracker_id` — ByteTrack sets that downstream.
- No class names — the annotator gets those from `self.inferencer.class_names` dict.
- For detection-only models, pass `mask=None`.

---

## 13. Case Studies: The Eleven Bugs We Squashed

Each bug teaches a distinct lesson. Cross-referenced to the lines in `isidet/src/inference/onnx_inferencer.py`. The last three (7, 8, 9) are RF-DETR specific and together produce the most interesting lesson in the whole exercise: **preprocessing is recipe-specific, and "close enough" is not a thing in transformer backbones**.

### Bug #1 — Post-NMS columns misread as per-class scores
**Symptom**: detection classes random, confidence values in range [0, class_id].
**Cause**: code treated columns `4:4+nc` as probabilities and argmaxed them.
**Fix**: `conf = preds[:, 4]; cls = preds[:, 5].astype(int)`.
**Lesson**: Know whether your YOLO export has NMS baked in. Test it.

### Bug #2 — Naive resize instead of letterbox
**Symptom**: boxes skewed horizontally on 16:9 RTSP feeds; vertical-only sources look fine.
**Cause**: `cv2.resize(frame, (model_w, model_h))` stretches non-square frames.
**Fix**: letterbox + save `(ratio, pad_x, pad_y)` + inverse transform at output.
**Lesson**: aspect-ratio-sensitive preprocess matters the moment your source isn't square.

### Bug #3 — Missing BGR→RGB conversion
**Symptom**: detection accuracy clearly worse than `.pt` equivalent; same boxes, lower confidences, some missed objects.
**Cause**: cv2 gives BGR, model trained on RGB, channels swapped.
**Fix**: `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before letterbox.
**Lesson**: always verify channel order of the source vs training. This bug is silent — detections still appear, just weaker.

### Bug #4 — Mask dtype uint8 instead of bool
**Symptom**: boxes render, masks don't, no errors in logs.
**Cause**: `supervision.MaskAnnotator` needs `np.bool_` masks; uint8 silently ignored.
**Fix**: `masks = np.zeros(..., dtype=bool); masks[...] = mask_resized > 0.5`.
**Lesson**: when a downstream consumer mysteriously ignores your output, check dtypes. Read the library source if the docs are thin.

### Bug #5 — NMS-free export pattern not handled
**Symptom**: hundreds of ghost boxes, FPS drops to 1 on GPU.
**Cause**: yolov26 with `nms=True` was fine, but earlier tests had exports where raw `[1, 4+nc+32, 3549]` was dumped through the post-NMS path.
**Fix**: detect layout by `max(shape[0], shape[1]) > 1000`, branch to `_postprocess_yolo_raw`.
**Lesson**: export format is not uniform even within "YOLO". Shape-based dispatch, not filename-based.

### Bug #6 — Stale Docker image
**Symptom**: diagnostic logs I added didn't show up in the container logs.
**Cause**: `isidet/src/` isn't volume-mounted in `docker-compose.yml`; only data dirs are. Code changes require `--build`.
**Fix**: either rebuild (`docker compose up -d --build`) or add `- ./src:/opt/isitec/src` to volumes for dev hot-reload.
**Lesson**: before debugging "why isn't my fix working", verify the fix is actually running. Log a banner line at import time if unsure.

### Bug #7 — Missing ImageNet normalisation for DINOv2 backbone
**Symptom**: RF-DETR ONNX: zero detections. Logits max `-2.45`, top sigmoid `0.08` — basically dead.
**Cause**: DINOv2 (RF-DETR's backbone) was trained with `(x/255 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]`. We fed raw `[0,1]` pixels. Off by a mean shift of ~0.45 and variance of ~5× — small numerically, catastrophic for transformer attention.
**Fix**: apply ImageNet mean/std in `preprocess()` when `is_rfdetr`. YOLO doesn't need it because its CNN was trained on raw `[0,1]`.
**Lesson**: transformer-based vision models have tight statistical expectations. YOLO's CNN backbones are more forgiving. If the family changes, audit the normalisation.
**How we found it**: standalone script (`a.py`) ran the ONNX with two preprocessing variants — normalised gave top sigmoid `0.977`, un-normalised gave `0.976` on a clean test frame. (Turned out not to be the dominant factor on that frame — but *was* the fix needed to give us the right baseline for later diagnostics.)

### Bug #8 — Letterbox applied to a stretch-trained model
**Symptom**: after fixing #7, RF-DETR still scored ~0.08 top sigmoid on the actual stream, even though the standalone test was `0.97`.
**Cause**: my preprocess letterboxed the frame (like YOLO) before feeding the model. rfdetr trains with `F.resize(img, (res, res))` — pure stretch, no pad. Result: ~44% of the 432×432 input was grey padding the model had never seen. Transformer attention interpreted "grey everywhere" as "this image has almost no content", and all queries shrank toward the background class.
**Fix**: branched `preprocess()` on `is_rfdetr`: stretch-resize for DETR, letterbox for YOLO. Matching branch in `_postprocess_rfdetr`: scale normalised boxes directly by `(orig_w, orig_h)` — no letterbox inverse.
**Lesson**: the resize strategy used during *training* determines what the model accepts at *inference*. Check the library source (`detr.py:659` in rfdetr) — one line of truth outranks a thousand lines of guessing.
**How we found it**: read `rfdetr/detr.py:620-700` (the `predict` method). Three lines: `F.to_tensor`, `F.resize(img, (res, res))`, `F.normalize`. The `(res, res)` tuple is diagnostic — if it were aspect-preserving it would be a different API call.

### Bug #9 — DETR class-index offset
**Symptom**: after #7 and #8, standalone test at `0.98` but live stream still at `0.08`. Same frame, same preprocessing.
**Cause**: DETR's COCO-91 head reserves **index 0 for background**. Fine-tuned classes start at index 1. My code sliced `logits[:, :nc]` which took columns 0 and 1 — column 0 being the dead background, column 1 being only *one* of the two fine-tuned classes (the other sat at column 2, which we discarded).
**Fix**: `logits = logits[:, 1 : 1+nc]`. After the slice, view-column 0 is app-class 0, view-column 1 is app-class 1.
**Lesson**: DETR-family class indices aren't zero-indexed at the fine-tuned classes. They inherit whatever convention the pretraining used (COCO starts categories at 1). Always print per-column max sigmoid on a known-good frame to identify which columns actually light up.
**How we found it**: added `col_max = probs.max(axis=0)` to the diagnostic. Output was `[(2, 0.977), (1, 0.044), (35, 0.019), ...]` — column 2 dominant, column 0 dead. Subtract 1 from every lit column and you have the fine-tuned indices.

### Bug #10 — Cross-thread CUDA session sharing causes multi-second stalls

**Symptom**: operators noticed hot-swap to a *preloaded* RF-DETR ONNX was **slower** than a cold swap. Server log showed `POST /api/start` hanging for 30–80 s before the `Model switched` line appeared. `cache HIT` was logged — so the session wasn't being rebuilt — yet the handler stalled afterwards anyway. Cold start ran the same postproc path in ~1 s.

**Cause**: the preload thread (spawned at app startup) built an `ort.InferenceSession` and stored it in a shared cache dict keyed by model path. Later, a Flask request thread pulled the cached session and called `session.run()` on it. `onnxruntime` *technically* supports this (sessions are documented as thread-safe for inference), but with the **CUDA execution provider** each session binds its compiled kernels and memory arenas to the CUDA stream of the thread that built it. Cross-thread `run()` triggers stream re-synchronisation, context re-binding, and in some driver versions a full reallocation of per-session GPU buffers — which can cost tens of seconds.

**Fix**: stop sharing session objects across threads. Keep the *preload itself* (it still warms the process-wide CUDNN kernel cache, which persists after the session is destroyed), but discard the session immediately after one dummy inference. Each real swap then builds a fresh session in the Flask request thread, reusing the cached compiled kernels from the driver layer. Swap latency dropped from 30–80 s to a consistent ~2 s.

```python
def preload_onnx(model_path):
    # Build, run one inference to force CUDNN algo selection, then discard.
    session = ort.InferenceSession(model_path, providers=[...CUDA...])
    dummy = np.random.randn(*shape).astype(np.float32)
    session.run(None, {inp.name: dummy})
    del session     # <- key line. Kernels stay cached at driver level.
```

**Lesson**: "thread-safe" in library docs means "won't crash", not "will perform the same from any thread". For CUDA-backed libraries, always ask **which thread owns the GPU context**. If the answer isn't the same thread that will consume the object, you have two options:

1. **Share only driver-level caches** (CUDNN kernel compilations, cuBLAS heuristics). These are process-wide and survive session destruction. Preload → run once → discard.
2. **Hand the object off to the consumer thread via a queue**, and build it there in the first place.

Anti-pattern: building an object in a worker thread and passing the reference for a different thread to use. You pay thread-affinity costs every call.

**How we found it**: stopwatched the hang. Noticed `cache HIT` fired at T=0 but the handler didn't return for T+80. Nothing between those two log lines except `_warmup()` → 3 × `session.run()`. If `run()` itself is slow on a *fresh* cached session (that was just built seconds ago), cross-thread sync is the top suspect. Confirmed by removing the cache entirely: swap cost became consistent 1–3 s across every invocation, matching the fresh-build baseline — which would be impossible if session construction were the bottleneck.

**Bonus corollary**: this is why subsequent swaps after the first cold swap are fast even without a cache. The *process* already has CUDNN kernels compiled for that graph; only the first swap in a given process pays the autotune cost. Cold swap #1: 3–5 s. Cold swap #2+: 1–2 s. No caching needed.

### Bug #11 — Orphan FP32 Cast nodes left behind by `convert_float_to_float16`

**Symptom**: an FP16-converted YOLO ONNX (produced via `compression/stages/fp16.py` calling `onnxconverter_common.float16.convert_float_to_float16`) refused to load in `OptimizedONNXInferencer`. Session creation died with::

    Type Error: Type parameter (T) of Optype (Concat) bound to different
    types (tensor(float16) and tensor(float)) in node (/model.23/Concat_7).

The FP32 source loaded fine; the FP16 file halved size as expected (10 MB → 5.3 MB) and 240 of 242 weight tensors were correctly converted to FP16. So the conversion *looked* right — the failure was a single typed-tensor mismatch deep in the graph.

**Cause**: `convert_float_to_float16` rewrites weight initializers and inserts boundary `Cast(FP32↔FP16)` pairs around blocklisted ops (`Resize`, `TopK`, …), but **it does not touch the `to` attribute of pre-existing Cast nodes that were baked into the original export**. Ultralytics' YOLO graph contains `/model.23/Cast_2` — a Cast(INT64→FP32) the export inserts so an `Unsqueeze`-derived index can be concatenated with float coordinates. After conversion, `Cast_2` still emits FP32, but the three other inputs to its consumer `/model.23/Concat_7` are now FP16. The Concat op requires all inputs to share dtype, and ORT's loader checks types at session creation — hence the immediate `Type Error` before a single inference frame runs.

This is *not* a bug in the converter's blocklist or in `keep_io_types=True` — those work as designed. It's a gap: the converter has no way to know whether an *original* Cast(to=FP32) is "really FP32 because the consumer wants FP32" or "incidentally FP32 because the export was FP32 throughout". When the rest of the graph drops to FP16, the orphan stands out.

**Fix**: a post-conversion sweep that walks the graph and rewrites any `Cast(to=FP32)` whose consumer is no longer FP32:

```python
def _fix_orphan_fp32_casts(self, model) -> int:
    FP32, FP16 = 1, 10
    consumers: dict[str, list] = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)
    output_names = {o.name for o in model.graph.output}
    blocklisted = set(self.fp16_op_block_list)

    fixed = 0
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        to_attr = next((a for a in node.attribute if a.name == "to"), None)
        if to_attr is None or to_attr.i != FP32:
            continue
        out_name = node.output[0]
        # Two cases where FP32 is correct and we leave it alone:
        if out_name in output_names:
            continue                              # keep_io_types boundary
        if any(c.op_type in blocklisted for c in consumers.get(out_name, [])):
            continue                              # converter's own boundary cast
        to_attr.i = FP16
        fixed += 1
    return fixed
```

Call it immediately after `convert_float_to_float16` and before the existing `del value_info / shape_inference.infer_shapes` step — the re-inference picks up the corrected types automatically. On the YOLO graph the sweep rewrites exactly **one** node (`/model.23/Cast_2`); the six other `Cast(to=FP32)` survivors are legitimate boundary casts feeding `Resize`/`TopK` and stay alone.

**Lesson**: when a graph-rewriting tool advertises "halves your weights to FP16", it operates on weight tensors and inserts compatibility scaffolding around known-hostile ops. It does **not** introspect every existing Cast. The places this leaks through are graphs whose original export already contained Cast nodes that *coincidentally* targeted FP32 — at which point those Casts become silent FP32 islands in an otherwise FP16 graph. The two safe categories (boundary toward a blocklisted op, boundary toward the model output under `keep_io_types`) are easy to enumerate; everything else can be rewritten.

**How we found it**: ORT's `Type Error` message names the offending Concat node, which is the consumer, not the producer. Tracing each of `Concat_7`'s four inputs by walking the graph produced one FP32 input among three FP16 — and the producer of that FP32 input was an unmodified `Cast(to=1)`. From there it was clear the converter had skipped that one `to` attribute. Validation: top-5 confidence scores from the patched FP16 model matched the FP32 baseline within `2×10⁻⁴` on a realistic (uniform `[0,1]`) input, confirming the fix didn't subtly break inference math.

---

## 14. Performance Playbook

### 14.1 Where time goes

| Stage | Typical cost (yolov26n @ 416, 1 detection) |
|---|---|
| Preprocess (BGR→RGB, letterbox, transpose) | 1–3 ms |
| `session.run()` on CUDA | 5–10 ms |
| Postprocess (threshold, letterbox inverse) | <0.5 ms |
| Mask decode (matmul + sigmoid) | 1–2 ms |
| Mask bbox-confined upsample × N | 1 ms × N |
| `supervision` mask/box/label annotation | 1–3 ms × N |

### 14.2 Biggest wins (in order)

1. **Execution provider** — CPU → CUDA is 10-50×. Verify `session.get_providers()[0]` is what you expect.
2. **Warmup** — avoid paying for first-run kernel compile on a real frame.
3. **Bbox-confined mask upsample** — 50× vs full-frame when sources are 1080p+.
4. **Batch input channel conversion** — combining `cv2.cvtColor` + letterbox + transpose in the fewest copies.
5. **TensorRT provider** — for static-shape production deployment, another 1.5–3× over CUDA.
6. **FP16 export** — cuts memory + often adds 1.3–2× on Ampere+ GPUs. Verify accuracy.

### 14.3 Things that don't help as much as you'd think

- Multi-threading `session.run()` — ONNX Runtime already parallelises op-level.
- Running mask sigmoid on GPU separately — the per-detection overhead eats the gains unless you batch.
- Switching `INTER_LINEAR` → `INTER_NEAREST` for mask resize — ~2× on that call but masks get noticeably blockier.
- **Caching ONNX session objects for reuse across threads.** Counter-intuitive, but worse than rebuilding. See §13 Bug #10.

### 14.4 Thread affinity for GPU sessions

**High-level:** a CUDA-backed `ort.InferenceSession` wants to be built *and used* on the same thread.

**Low-level:** when the CUDA EP initialises a session, it binds:
- Compiled CUDNN kernels to a specific CUDA stream.
- Memory arena to the thread's context handle.
- `cudnn_conv_algo_search='HEURISTIC'` results to this session's lookup table.

When a different thread calls `session.run()`, onnxruntime must synchronise across streams, potentially reallocate buffers, and in some CUDA driver versions the handover itself stalls until the original stream idles. This is the "works but slow from another thread" trap.

**Good patterns:**

| Goal | Pattern |
|---|---|
| Pre-warm kernel cache at boot | Build, run one dummy inference, **`del session`**. CUDNN's kernel cache is process-wide and survives. |
| Fast first inference in request thread | Build the session *in* the request thread. CUDNN cache (from preload above) makes construction ~2× faster. |
| Sub-second hot-swap | Refactor: build in the inference thread (already owns the stream), atomic swap of `engine.inferencer` after next frame boundary. The Flask request just queues the build. |

**Bad patterns:**

- Global session cache shared across Flask workers and inference thread.
- Building session in `main()` then handing to worker threads.
- Assuming "thread-safe" in library docs means "performance-safe".

---

## 15. A Minimal End-to-End Skeleton

Putting it all together, stripped of project-specific cruft. Adapt to your exporter.

```python
import cv2, numpy as np, onnxruntime as ort

class MinimalYOLOInferencer:
    def __init__(self, model_path, conf=0.5):
        self.session = ort.InferenceSession(
            model_path,
            providers=[('CUDAExecutionProvider', {}), 'CPUExecutionProvider'],
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        _, _, self.H, self.W = inp.shape
        self.conf = conf

    def _letterbox(self, frame):
        oh, ow = frame.shape[:2]
        r = min(self.H / oh, self.W / ow)
        nw, nh = int(round(ow * r)), int(round(oh * r))
        resized = cv2.resize(frame, (nw, nh))
        pad_w, pad_h = self.W - nw, self.H - nh
        left, top = pad_w // 2, pad_h // 2
        padded = cv2.copyMakeBorder(
            resized, top, pad_h - top, left, pad_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        return padded, r, left, top

    def __call__(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        padded, r, px, py = self._letterbox(rgb)
        x = padded.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]
        x = np.ascontiguousarray(x)

        out = self.session.run(None, {self.input_name: x})[0][0]  # (N, 6+...)

        keep = out[:, 4] > self.conf
        boxes = out[keep, :4].copy()
        scores = out[keep, 4]
        classes = out[keep, 5].astype(int)

        # Inverse letterbox
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / r
        oh, ow = frame_bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh)

        return boxes, scores, classes
```

Run it:
```python
inf = MinimalYOLOInferencer("best.onnx")
frame = cv2.imread("sample.jpg")
boxes, scores, classes = inf(frame)
for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
cv2.imwrite("out.jpg", frame)
```

That's the whole game for a detection model. Everything else is a delta on top: swap the postprocessor for DETR, add mask decode for segmentation, stamp metadata at export to make classification automatic.

---

## 16. Debugging Checklist

When a new ONNX model refuses to cooperate, walk this list in order. Stop the first time a print tells you something surprising — that's usually the bug.

### Tier 1 — is the pipe even open?
1. **Does the session load?** If not: opset mismatch or provider issue. `ort.get_available_providers()`.
2. **What are the output names and shapes?** Always print them. Most bugs start here.
3. **What family is this?** Post-NMS YOLO / pre-NMS YOLO / DETR. See §3.
4. **Is the input dtype float32?** Check `session.get_inputs()[0].type`. If `uint8`, normalisation is baked in — don't apply it externally.
5. **Is the container running the code you think?** Print a banner line on import. Bind-mount `isidet/src/` during dev so edits are hot.

### Tier 2 — does the model see anything?
6. **Print max logit and top-5 sigmoid on a known-good frame.** This one check distinguishes "model is dormant" (max < 0) from "model is confident but filtered" (max > 2, count_above = 0 because of threshold/slice/coord bug).
7. **Run the A/B script (§5.5).** Two preprocessing variants side by side, top sigmoid score as the verdict. Which wins tells you what the model was trained with.
8. **Is your frame RGB?** cv2 gives BGR, training uses RGB. Compare scores with and without the swap.
9. **Is normalisation right for the family?** YOLO: no norm. DETR/transformers: ImageNet `(x/255 - mean)/std`. Uint8-input graph: no external norm at all.

### Tier 3 — does the model see the right thing?
10. **Letterbox vs stretch?** Read the library's training code (§5.4). Grep for `Resize`, `Normalize`, `to_tensor`. One line in the right file beats a week of guessing.
11. **Is the class-index slice correct?** For DETR: print `probs.max(axis=0)` to find which columns light up. If column 0 is always dead and columns `1..nc` are hot, use `logits[:, 1:1+nc]`. Never assume 0-indexed.
12. **Is your letterbox math reversible?** Print `(ratio, pad_x, pad_y)` once. Apply forward then inverse on a synthetic box — should land at the start.

### Tier 4 — does the model tell you, but you ignore it?
13. **Is your confidence threshold too strict?** Lower to 0.1, see if any detections emerge. Separates "no detections" from "all filtered out".
14. **Are masks bool?** Not uint8. Print `masks.dtype` and `np.count_nonzero(masks)`. `supervision.MaskAnnotator` silently skips uint8.
15. **Does the output agree with the `.pt`/`.pth` path** on the same frame? If not, diff box counts and positions — the discrepancy tells you which stage is wrong.

### Tier 5 — is the model fast but "feels" slow?
16. **Which thread built the session? Which thread is calling `run()`?** If they differ and the EP is CUDA, expect multi-second stalls. See §14.4. Rule of thumb: build in the thread that will consume. Preload only the driver-level kernel cache via build → run once → `del`.
17. **Hot-swap looks like a server hang, but `/api/stats` still updates?** It's a browser MJPEG reconnect, not a backend stall. Time the POST with `curl` to confirm server-side latency.
18. **First swap slow, later swaps fast?** Normal — CUDNN autotuning runs once per (graph, input-shape) pair per process. Preload at boot to hide that first-time cost behind app start.

### The "flatlined top score" flowchart
When your DETR top sigmoid is below `0.2` on a real frame:
```
top_sigmoid < 0.2 on known-good frame
├── logit max < 0   → model is dormant
│   ├── run A/B script (§5.5)
│   ├── try with/without ImageNet norm       (bug #7)
│   ├── try with/without letterbox           (bug #8)
│   └── try with/without BGR→RGB swap        (bug #3)
└── logit max > 2   → model is alive, you're filtering wrong
    ├── check class-index slice              (bug #9)
    ├── check confidence threshold
    └── check box clip / degenerate filter
```

---

## 17. Further Reading

- [ONNX spec](https://github.com/onnx/onnx/blob/main/docs/IR.md) — the file format itself.
- [ONNX Runtime API](https://onnxruntime.ai/docs/api/python/api_summary.html) — `InferenceSession`, providers, graph optimization.
- [Netron](https://netron.app) — drag-and-drop graph viewer; essential.
- [Ultralytics export docs](https://docs.ultralytics.com/modes/export/) — flags that change output shape.
- [supervision.Detections](https://supervision.roboflow.com/detection/core/) — the data class we use.
- In this repo:
  - `isidet/src/inference/onnx_inferencer.py` — the implementation this tutorial explains.
  - `isidet/src/inference/base_inferencer.py` — abstract parent, class-name loading, rescale helper.
  - `isidet/src/training/trainers/rfdetr.py:92-140` — the native RF-DETR postprocess we replicated for ONNX.

---

## Summary

Working with ONNX is not magic. It's four questions asked repeatedly:

1. **What goes in?** Input name, shape, dtype. Match them. For the *values*: color space, resize strategy (letterbox vs stretch), normalisation (none vs ImageNet), range (`/255` vs uint8). The training code is authoritative — `F.resize(img, (res, res))` and `F.normalize(img, mean, std)` are the kind of lines that make or break you.
2. **What comes out?** Output names and shapes tell you the family. Column semantics within each output tell you the decoder. For DETR: **class index 0 is background**, not your first class.
3. **Does the model agree with the frame?** Print max logit and top sigmoid on a known-good frame *before* you trust the pipeline. If the numbers are dead, preprocessing is wrong. If they are high but no detections survive, filtering or slicing is wrong.
4. **How do coordinates flow?** Trace every transform: original → (letterboxed or stretched) model → (maybe proto/mask) → back to original. Invert at every boundary. For letterbox: store `(ratio, pad_x, pad_y)` once, reuse everywhere.

Get those four right and the rest is numpy.

The meta-lesson: **don't guess — instrument.** Every bug in this repo was found by printing something. Every fix was validated by printing something else. The A/B diagnostic in `a.py` took 20 lines and replaced several hours of speculation. Keep it nearby; it scales to every new model family you'll meet.

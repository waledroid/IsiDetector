# Compression — `./compress.sh`

Interactive and scriptable tool for shrinking models, converting between formats, and benchmarking the result. Lives in `compression/` at the repo root, invoked via the `./compress.sh` root wrapper (which `exec`s `deploy/_impl/compress.sh`, sets `PYTHONPATH=isidet/`, then calls `python -m compression`).

Runs on the **office GPU workstation**, not on site PCs — the heavy conversion + validation work needs the full dep stack (`onnx`, `onnxconverter-common`, `onnxruntime`, `openvino`, `onnxsim`). Site PCs only consume the compressed artefacts.

---

## Two modes

### 1. Interactive menu (no arguments)

```bash
./compress.sh
```

Opens a four-level back-navigable picker:

```
❯ What would you like to do?
  🧩   Convert a model format
  🗜️   Compress a model
  📊   Benchmark variants
  🧪   Validate accuracy
```

Discovery walks the repo for every `.onnx` it can find (grouped by training run) and shows them in a coloured table. `.pt` / `.pth` / `.xml` files are discovered on-demand inside the `Convert` flow.

### 2. Scripted / one-shot (--model + --stage or --convert)

```bash
./compress.sh --help                            # full argparse usage

./compress.sh --model PATH --stage  NAME        # compression stage
./compress.sh --model PATH --convert NAME       # format conversion
./compress.sh --model PATH --stage NAME --output /custom/path.onnx
```

`PATH` accepts any absolute or relative path — inside or outside the repo.

---

## Compression stages (`--stage`)

All operate on ONNX input and produce ONNX output, except `openvino_fp16` which takes `.xml` → `.xml`.

| Stage | What it does | Input | Typical size delta |
|---|---|---|---|
| `fp16` | Cast every FP32 weight tensor to FP16. Adds `Cast` ops around a short blocklist of ops that lose precision at half precision (`Resize`, `Upsample`, `ReduceSum`, `TopK`). `keep_io_types=True` holds model inputs/outputs at FP32. | `.onnx` | **~½ size**, negligible accuracy drop |
| `int8` | Dynamic INT8 quantisation with synthetic calibration — weights quantised offline, activations quantised on the fly at inference. Fast to run, CPU-only. | `.onnx` | ~¼ size, 1–3 % mAP drop |
| `int8_qdq` | Static INT8 (QDQ format) with real image calibration — activation ranges measured on samples from `isidet/data/…`. Cross-EP (ORT CUDA / CPU / OpenVINO). | `.onnx` | ~¼ size, smaller mAP drop than dynamic INT8 |
| `sim` | Runs `onnxsim` — constant folding, dead-op removal, shape-inference cleanup. Not a size play; unblocks downstream stages that choke on redundant ops. | `.onnx` (refuses if already `.sim.onnx`) | usually unchanged or slightly smaller |
| `openvino_fp16` | Re-save an OpenVINO IR with `compress_to_fp16=True`. Halves the `.bin` weight file. | `.xml` (FP32 IR) | ~½ `.bin` size |

Every stage prints phase-by-phase progress with elapsed seconds and emits a heartbeat line every 10 s during long calls so a slow conversion looks alive instead of hung.

---

## Format conversions (`--convert`)

Wrappers around `isidet/src/inference/export_engine.py` with the same phase-print + heartbeat UX.

| Mode | Pipeline | Input |
|---|---|---|
| `pt-onnx` | `.pt`/`.pth` → `.onnx` (Ultralytics YOLO or `rfdetr` export) | `.pt` / `.pth` |
| `pt-sim` | `.pt`/`.pth` → `.onnx` → `.sim.onnx` | `.pt` / `.pth` |
| `pt-openvino` | `.pt`/`.pth` → `.onnx` → `.sim.onnx` → OpenVINO IR (full deploy pipeline) | `.pt` / `.pth` |
| `onnx-sim` | `.onnx` → `.sim.onnx` | `.onnx` |
| `onnx-openvino` | `.onnx` → OpenVINO IR (`.xml` + `.bin`) | `.onnx` |
| `openvino-fp16` | FP32 `.xml` → FP16 `.xml` (equivalent to `--stage openvino_fp16`) | `.xml` |

---

## Examples

```bash
# Shrink a YOLO ONNX to FP16 (most common case)
./compress.sh --model isidet/models/yolo/2026-04-01/weights/best.onnx --stage fp16

# Full pipeline from raw .pt to a deployable OpenVINO IR
./compress.sh --model isidet/models/yolo/2026-04-01/weights/best.pt --convert pt-openvino

# Convert a single ONNX to OpenVINO IR (skipping sim)
./compress.sh --model some/external/foo.onnx --convert onnx-openvino

# Compress an existing FP32 OpenVINO IR in place
./compress.sh --model isidet/models/yolo/2026-04-01/weights/openvino/model.xml --stage openvino_fp16

# Static INT8 with real calibration (RF-DETR)
./compress.sh --model isidet/models/rfdetr/2026-04-01/inference_model.sim.onnx --stage int8_qdq
```

---

## What it does **not** do

- **No in-place compression of `.pt` / `.pth`.** Export to `.onnx` first (use `--convert pt-onnx`), then compress.
- **No RF-DETR → OpenVINO.** `src/inference/openvino_inferencer.py` hard-rejects RF-DETR IR at load time because OpenVINO 2026 mistranslates the transformer's `Einsum` ops. Stick to ONNX for RF-DETR.
- **No TensorRT compilation.** The TensorRT path lives in `isidet/src/inference/export_engine.py --format tensorrt`. It's GPU-per-device and intentionally not in the cross-backend compression tool.

---

## Architecture

```
compression/
├── __main__.py         # argparse front-door + one-shot dispatch
├── cli.py              # Interactive questionary menu + PROJECT_ROOT anchor
├── discovery.py        # *.onnx walk, run-directory grouping, variant detection
├── inspect.py          # ONNXProperties (dtype / op count / size)
├── convert_ops.py      # pt_to_onnx, onnx_to_sim, onnx_to_openvino, openvino_fp16
├── benchmark.py        # FPS + latency across every variant of a model
├── validate.py         # Accuracy diff vs baseline on real frames
├── stages/
│   ├── base.py         # Stage ABC + register() decorator
│   ├── fp16.py         # FP16Stage — narrow op blocklist + heartbeat
│   ├── int8.py         # INT8Stage — dynamic, synthetic calibration
│   ├── int8_qdq.py     # INT8QDQStage — static, real-image calibration
│   └── sim.py          # SimplifyStage — onnxsim wrapper
├── calibration/
│   └── image_reader.py # Letterbox + ImageNet normalisation for calibration
├── ui.py               # Rich banner, capabilities table, models table
└── requirements.txt    # rich + questionary + onnxconverter-common + onnxsim
```

Every stage is a `Stage` subclass registered into `STAGES` via `@register`. Adding a new compression strategy = add a new file + the decorator + nothing else. The interactive menu, CLI flags, and discovery auto-pick it up.

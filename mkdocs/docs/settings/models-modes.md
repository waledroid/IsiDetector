# Models & Modes (GPU/CPU)

Drop a model file into the models folder, pick it in Settings, and the engine runs it on the right backend for your hardware.

## Drop in a model

Copy the model file the office workstation sent you into the models folder, then select it in **Settings → Model 1**. You do not choose a backend — the engine picks one from the file extension.

| Extension | Backend | Runs on |
|---|---|---|
| `.engine` | TensorRT | GPU only |
| `.pt` | YOLO (Ultralytics) | GPU |
| `.pth` | RF-DETR | GPU |
| `.xml` | OpenVINO | CPU (Intel) |
| `.onnx` | ONNX Runtime | GPU or CPU |

!!! tip
    The mode badge in the live view shows what loaded, e.g. `OpenVINO INT8 • CPU` or `TensorRT • GPU`. Use it to confirm the file you dropped in is the one running.

## CPU mode vs GPU mode

The site PC runs in **CPU** or **GPU** mode, fixed at startup by `COMPOSE_MODE` (set by `up.sh` / `run_start.sh`). The mode decides which model files are accepted.

| | CPU mode | GPU mode |
|---|---|---|
| Allowed files | `.xml`, `.onnx` | `.engine`, `.pt`, `.pth`, `.xml`, `.onnx` |
| Model families | YOLO only | YOLO + RF-DETR |
| YOLO image size | 320 | 640 |
| Masks + traces | off (boxes only) | on |

On CPU, smaller image size and box-only rendering keep the belt real-time. On GPU there is headroom for full masks and per-track traces.

!!! warning "Wrong file for the mode"
    Select a file the mode does not allow and the model **refuses to load** with a clear message. On CPU mode, a `.pt`, `.pth`, or `.engine` file is rejected — use a `.xml` or `.onnx` export instead. Ask the office workstation to re-export via `compress.sh` if you only have a `.pt`.

## OpenVINO will not run RF-DETR

A `.xml` file is only valid for **YOLO** models. RF-DETR `.xml` files are hard-refused at load time: OpenVINO mistranslates RF-DETR's transformer ops and would return zero detections. The engine raises an error instead of silently counting nothing.

For RF-DETR, use an `.onnx` export (CPU or GPU) or a `.pth` (GPU only).

## TensorRT engines are per-GPU

A `.engine` file is compiled for **one specific GPU**. It will not load on a different GPU model.

!!! warning "New or swapped GPU"
    If the site PC's graphics card is replaced, the old `.engine` file stops working. Switch Settings to the matching `.onnx` model and ask the office workstation to rebuild the `.engine` for the new hardware. See [Reference](../reference/index.md) for the export/build steps.

## Confidence threshold

Each detection must score above the confidence threshold to be counted. The shipped default is **0.3**.

- Raise it (e.g. 0.5) if you see false detections on an empty belt.
- Lower it if real parcels are being missed.

Set it in Settings. It applies to whichever model is loaded, on either mode.

!!! note
    Backend, mode, and image size are not operator dials — they follow the file and the PC's `COMPOSE_MODE`. Confidence is the one detection knob you tune on site.

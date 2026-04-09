# TensorRT Engine

The `TensorRTInferencer` delivers maximum inference throughput on NVIDIA GPUs by loading pre-compiled `.engine` files.

---

## When to Use

- **NVIDIA GPU** with TensorRT installed
- **Maximum FPS** required (10-30% faster than ONNX CUDA)
- **Fixed deployment hardware** — engines are compiled for a specific GPU architecture

!!! warning
    TensorRT engines are **not portable** across GPU architectures. An engine built on an RTX 3080 will not work on an RTX 4090. Rebuild with the [Export Engine](export.md) on the target machine.

---

## Technical Overview

:material-file-code: **Source**: `src/inference/tensorrt_inferencer.py`

The inferencer loads a serialized `.engine` file, allocates CUDA input/output buffers, and runs inference via the TensorRT Python API.

### Requirements

```bash
pip install tensorrt pycuda
```

If TensorRT is not installed, the module still imports but raises a clear error at instantiation.

---

## Usage

=== "CLI"

    ```bash
    python scripts/run_live.py \
        --weights models/yolo/tensorrt/model.engine \
        --source "rtsp://192.168.1.100:554/stream"
    ```

=== "Python"

    ```python
    from src.inference.tensorrt_inferencer import TensorRTInferencer

    engine = TensorRTInferencer(
        model_path="tensorrt/model.engine",
        conf_threshold=0.5
    )
    detections = engine.predict_frame(frame)
    ```

!!! note
    TensorRT engines are **rejected on CPU-only machines**. The web app returns a clear error message guiding the user to use OpenVINO or ONNX instead.

---

## How to Generate TensorRT Engines

```bash
python -m src.inference.export_engine \
    --model-dir runs/segment/models/yolo/yolo26m_640_200/weights \
    --format tensorrt
```

Requires `trtexec` on PATH. The export engine uses FP16 precision by default for optimal speed/accuracy balance.

# Platform Functionalities

Deep-dive into the core features of the IsiDetector Web App.

---

## 1. Dual-Mode Perception

The platform supports two distinct perception modes reachable via the "Modes" dropdown:

- **Mode 1 (YOLO)**: Speed-optimized detection using the **YOLOv26-seg** architecture (NMS-free via one-to-one label assignment). Best for high-speed conveyor belts.
- **Mode 2 (RF-DETR)**: Transformer-based detection. Highly accurate for complex, overlapping objects or varying light conditions.

---

## 2. Intelligent Source Switching

The app automatically adjusts its processing logic based on the input source:

### Local Processing (Inspection Mode)
- **Images**: When you upload a JPG or PNG, the system processes it once and "freezes" the result on the screen. This is perfect for quality control sub-stations.
- **Videos**: Processes the MP4 file frame-by-frame, allowing you to watch the "replay" of an event with AI overlays.

### Live Telemetry (Streaming Mode)
- **USB/Integrated Cameras**: Connects directly to local hardware with near-zero latency.
- **RTSP Streams**: Connects to industrial IP cameras (e.g., Hikvision, Amcrest) using the `rtsp://` protocol.

---

## 3. Dynamic Configuration

Unlike static applications, the web app is **"Config-Live"**. It reads the `isidet/configs/train.yaml` file every time a stream starts.

This allows you to change:
- **Confidence Cutoffs**: `conf_threshold`
- **Line Positions**: Logic-based positioning.
- **Tracker Memory**: `track_buffer` (How many frames to "remember" a parcel if it's hidden).

---

## 4. Real-Time Analytics

The **Detection Analytics** chart updates dynamically as parcels cross the line.
- **Live Filter**: Shows the current session's productivity.
- **Historical (24h/7d/30d)**: Aggregates CSV logs to show long-term productivity trends.

---

## 5. Global Language Switching

The platform supports real-time translation of the user interface. By hitting the `/api/language` endpoint (or toggling the UI switch), the `StreamHandler` adjusts its internal reporting and the frontend updates labels instantly without a page refresh.

Supported locales include:
- English (EN)
- French (FR)
- German (DE)

---

## 6. Persistent Model Hot-Swap

Operators can switch between model backends mid-stream without losing session data. Clicking "Start" in the UI with a different weight file while a stream is already running triggers a hot-swap path (`StreamHandler.start() → can_hot_swap` branch).

### Accepted weight formats

The hot-swap route accepts five weight formats. `StreamHandler._build_engine()` dispatches by file extension — no config flag needed:

| Extension | Backend | Device | Used when |
|---|---|---|---|
| `.pt` | **Ultralytics YOLO** (PyTorch) | CUDA if available, else CPU | You have the native YOLO checkpoint from training |
| `.pth` | **RF-DETR** (PyTorch, via the `rfdetr` library) | CUDA required | Native RF-DETR checkpoint. In Docker this routes to the `rfdetr` sidecar over HTTP (`http://rfdetr:9510`); bare-metal uses the local `RFDETRInferencer` |
| `.onnx` | **ONNX Runtime** | CUDA if available, else CPU | Portable cross-framework export. Works for both YOLO and RF-DETR — the engine auto-detects the family from the graph (see [ONNX Engine](../inference/onnx.md)) |
| `.xml` | **OpenVINO** | Intel CPU | Fastest CPU backend on Intel hardware. Requires the companion `.bin` file alongside the `.xml` |
| `.engine` | **TensorRT** | NVIDIA GPU only (rejects with a clear error on CPU-only hosts) | Production-grade GPU inference, typically 1.5–3× faster than ONNX-CUDA. Static-shape only — recompile per input size |

The resulting `mode_text` displayed in the UI footer reflects the chosen backend and device:

- `YOLO • GPU`, `YOLO • CPU`
- `RF-DETR • GPU`, `RF-DETR Remote • GPU` (via sidecar)
- `ONNX • GPU`, `ONNX • CPU`
- `OpenVINO • CPU`
- `TensorRT • GPU`

### Auto-discovery

If the UI sends `weights: null`, the backend falls back to `_resolve_default_weights(model_type)` — which reads `webapp/isitec_app/settings.json` for a user-chosen default (`yolo_weights` / `rfdetr_weights`), and if empty, auto-discovers the newest weight file under `isidet/models/yolo/` or `isidet/models/rfdetr/` respecting a per-device extension priority list:

- GPU + YOLO: prefer `.pt` → `.onnx` → `.xml`
- GPU + RF-DETR: prefer `.pth` → `.onnx` → `.xml`
- CPU: prefer `.xml` → `.onnx` → `.pth` → `.pt`

So an operator without credentials can click "Start" and the backend picks the fastest weight format available on the current hardware.

### State preserved across the swap

| State | Behaviour |
|---|---|
| Running counts (`class_totals`) | Preserved — not reset to zero |
| Tracker IDs (`ByteTrack`) | Preserved — same instance keeps running |
| Counted-crossing set (`counted_ids`) | Preserved — no double-count on mid-crossing objects |
| Line zone position | Preserved |
| Daily CSV logger | Preserved — keeps writing to the same file |
| Stream reader + RTSP connection | Preserved — no reconnect |

What changes:

- The inferencer (`self.engine.inferencer`) points to the new model.
- Palette-dependent annotators are rebuilt so colours reflect the new model's class-ID convention — operators **see colours flip** as visual confirmation the swap actually took effect.

Implementation: `VisionEngine.swap_inferencer(new_inferencer)` does the in-place replacement. `StreamHandler` keeps the `class_totals` dict alive and uses `setdefault` to backfill any new class names the new model exposes, never zeroing existing buckets.

!!! tip "Why colours flip on YOLO ↔ RF-DETR swap"
    YOLO emits `class_id ∈ {0, 1}` (carton, polybag). RF-DETR emits `class_id ∈ {1, 2}` (background class 0 reserved by DETR convention). `supervision`'s annotators colour by `class_id` via a palette lookup, so the two families land on different palette slots. This divergence is intentional — operators get an instant visual signal that the swap worked without reading logs.

Hot-swap latency:

- YOLO `.pt`/`.onnx` ↔ YOLO or RF-DETR ONNX: **~2 seconds** (CUDA session construction for the new model).
- First swap to RF-DETR ONNX after cold start: same 2 s if the default `rfdetr_weights` was preloaded (see `preload_onnx` in [ONNX Engine §5](../inference/onnx.md)); 5–8 s otherwise.
- Swap to a different **source** (file / RTSP URL) falls through to the full-restart path — counts reset intentionally because it's a new session.

---

## 7. Deployment Time Alignment

Both container images set `TZ=Europe/Paris` and include the `tzdata` package. Timestamps produced inside the containers — CSV analytics rows, UDP event `ts` fields, session summaries in `isidet/logs/sessions.json`, the `/api/stats` "last detected" field — all match the host's wall-clock time.

The same `TZ` value is also exported via `docker-compose.yml → services.*.environment` for belt-and-braces overrides. Change `Europe/Paris` in three places (both `Dockerfile`s and `docker-compose.yml`) if the rig is deployed outside France, or externalise via `TZ=${TZ:-Europe/Paris}` in the compose file.

---

## 8. File-Source Loop Without Stutter

When an uploaded MP4 reaches its last frame, the `LiveReader` silently reopens the file and keeps delivering frames from the beginning. There is no 1-second pause, no `Stream disconnected` warning in the logs, and no visible stutter in the MJPEG feed at the loop boundary.

RTSP and webcam sources still log a warning and back off 1 second on genuine disconnects — the silent-loop path only applies when `self.is_file` is True.

---

## 9. Backend API (Fast Integration)

For industrial control systems (SCADA/PLC) that need to interact with IsiDetector externally, the following REST APIs are available:

| Endpoint | Method | Payload / Params | Description |
|---|---|---|---|
| `/video_feed` | `GET` | — | MJPEG stream (`multipart/x-mixed-replace`). Embed in `<img src="/video_feed">`. |
| `/api/start` | `POST` | `{"source": "...", "model_type": "yolo\|rfdetr", "weights": "path/to/weights"}` | Starts the inference loop. `weights` is optional — falls back to the latest found checkpoint. |
| `/api/stop` | `POST` | — | Gracefully terminates the stream and flushes the session CSV. |
| `/api/upload` | `POST` | `Multipart/Form-Data` (field: `file`) | Uploads a JPG/PNG/MP4 to `webapp/isitec_app/uploads/` for local inspection. Returns `{"filepath": "..."}`. |
| `/api/language`| `POST` | `{"language": "en\|fr\|de"}` | Switches the StreamHandler locale in real-time without a page reload. |
| `/api/stats` | `GET` | — | Returns `{"is_running": bool, "counts": {...}, "last_detected": {...}}`. |
| `/api/chart` | `GET` | `?period=live\|24h\|7d\|30d` | `live` mirrors `/api/stats` counts. Historical periods aggregate CSV logs from `webapp/isitec_app/logs/`. |

!!! info "Chart period detail"
    - `live` — mirrors the current in-memory session counts (no disk read)
    - `24h` / `7d` / `30d` — scans all `.csv` files in `webapp/isitec_app/logs/` and counts rows whose timestamp falls within the cutoff window

---

## 10. Long-Run Stability (12-Hour Shifts)

The web app is designed to run uninterrupted across full production shifts. Several mechanisms work together to prevent the common failure modes of long-running inference loops.

### FPS Pacing for Uploaded Video

When the source is an MP4 file, the `LiveReader` thread reads frames as fast as possible — much faster than real time. Without pacing, the inference queue floods and the video appears to play at 2× or 3× speed.

**Fix**: The reader reads the native FPS from OpenCV and sleeps `1/fps` seconds after each frame:

```python
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_delay = 1.0 / fps          # e.g. 0.033s at 30fps
# ... after q.put(frame):
time.sleep(frame_delay)           # paces frame production to real-time
```

This is only applied for file sources — RTSP and webcam streams are self-pacing.

### Exponential Backoff on Inference Errors

A GPU error (CUDA OOM, device reset) that persists will cause the inference loop to retry continuously. Without backoff, this generates 10 error log lines per second and wastes CPU.

**Fix**: The loop tracks `consecutive_errors` and sleeps `min(0.1 × 2^n, 5.0)` seconds between retries — capping at 5 seconds. The counter resets to 0 on the next successful frame.

| Consecutive Errors | Sleep |
|---|---|
| 1 | 0.1s |
| 2 | 0.2s |
| 3 | 0.4s |
| 5 | 1.6s |
| 7+ | 5.0s (cap) |

### Periodic CUDA Memory Flush

PyTorch does not automatically return freed GPU tensors to the OS — it holds them in an internal cache. Over thousands of frames this cache can consume hundreds of MB that could otherwise be used for inference.

**Fix**: Every 1800 frames (~60s at 30fps), the loop forces a GC cycle and calls `torch.cuda.empty_cache()`:

```python
if frame_count % 1800 == 0:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### Heartbeat Logging

If the inference thread stalls silently (e.g., waiting on an empty queue, or a lock contention), there is no indication from the outside — the web UI just shows a frozen frame. A heartbeat log line makes this detectable.

**Fix**: Every 9000 frames (~5 minutes at 30fps), the loop logs a status line:

```text
INFO: 💓 Inference heartbeat — frame 9000, counts: {'carton': 142, 'polybag': 87}
```

If you stop seeing heartbeats in the log, the inference thread has stalled.

### MJPEG Generator Safety

If a client disconnects mid-stream (browser tab closed, network drop), Flask's generator can raise when trying to `yield` the next frame. Without handling, this silently stops the generator — the next client sees no stream.

**Fix**: Each `yield` in `generate_frames()` is wrapped in a `try/except`. On exception the generator returns cleanly, freeing the connection.

---

## API Reference

::: isitec_app.stream_handler.StreamHandler

# Platform Functionalities

Deep-dive into the core features of the IsiDetector Web App.

---

## 1. Dual-Mode Perception

The platform supports two distinct perception modes reachable via the "Modes" dropdown:

- **Mode 1 (YOLO)**: Speed-optimized detection using the **YOLOv12-seg** architecture. Best for high-speed conveyor belts.
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

Unlike static applications, the web app is **"Config-Live"**. It reads the `configs/train.yaml` file every time a stream starts.

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

## 6. Backend API (Fast Integration)

For industrial control systems (SCADA/PLC) that need to interact with IsiDetector externally, the following REST APIs are available:

| Endpoint | Method | Payload / Params | Description |
|---|---|---|---|
| `/video_feed` | `GET` | — | MJPEG stream (`multipart/x-mixed-replace`). Embed in `<img src="/video_feed">`. |
| `/api/start` | `POST` | `{"source": "...", "model_type": "yolo\|rfdetr", "weights": "path/to/weights"}` | Starts the inference loop. `weights` is optional — falls back to the latest found checkpoint. |
| `/api/stop` | `POST` | — | Gracefully terminates the stream and flushes the session CSV. |
| `/api/upload` | `POST` | `Multipart/Form-Data` (field: `file`) | Uploads a JPG/PNG/MP4 to `isitec_app/uploads/` for local inspection. Returns `{"filepath": "..."}`. |
| `/api/language`| `POST` | `{"language": "en\|fr\|de"}` | Switches the StreamHandler locale in real-time without a page reload. |
| `/api/stats` | `GET` | — | Returns `{"is_running": bool, "counts": {...}, "last_detected": {...}}`. |
| `/api/chart` | `GET` | `?period=live\|24h\|7d\|30d` | `live` mirrors `/api/stats` counts. Historical periods aggregate CSV logs from `isitec_app/logs/`. |

!!! info "Chart period detail"
    - `live` — mirrors the current in-memory session counts (no disk read)
    - `24h` / `7d` / `30d` — scans all `.csv` files in `isitec_app/logs/` and counts rows whose timestamp falls within the cutoff window

---

## 7. Long-Run Stability (12-Hour Shifts)

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

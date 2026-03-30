# IsiDetector VisionAI Platform

The Web Application provides a modern, responsive interface for managing your Industrial VisionAI pipeline without using the command line.

---

## System Overview

The platform is built using:
- **Backend**: Python Flask (Robust, lightweight).
- **Inference Engine**: Asynchronous `StreamHandler` for non-blocking AI processing.
- **Communication**: MJPEG Streaming for low-latency visual feedback.
- **Frontend**: Clean Material Design with responsive analytics.

### Main Navigation

| Section | Purpose |
|---|---|
| **Live Inference** | Real-time visual monitoring and control. |
| **Analytics** | Historical data exploration and chart views. |
| **Models** | High-level overview of active perception weights. |
| **Settings** | Configuration of thresholds and line triggers. |

---

## Internal Architecture

The web app is split into two decoupled Python modules:

```text
isitec_app/
├── app.py              # Flask routes — thin HTTP layer only
└── stream_handler.py   # All AI and streaming logic
    ├── StreamHandler   # Session manager (start/stop, stats, language)
    ├── LiveReader      # Background video-capture thread (zero-lag queue)
    └── TelemetryPublisher  # Simulated MQTT publisher to isitec/sorting
```

**Key design principle:** `app.py` never touches OpenCV or model weights directly — it delegates everything to `StreamHandler`. This keeps Flask routes clean and thread-safe.

### Async Processing Pipeline

```mermaid
graph LR
    A[Camera / RTSP / File] --> B[LiveReader<br/>Background Thread]
    B -->|Latest frame| C[_inference_loop<br/>Daemon Thread]
    C --> D[VisionEngine<br/>ByteTrack + Line Count]
    D --> E[latest_annotated<br/>JPEG buffer]
    E --> F[/video_feed<br/>MJPEG stream]
    D -->|line_crossed event| G[TelemetryPublisher<br/>MQTT log]
```

The `LiveReader` uses a `queue.Queue(maxsize=1)` — always discarding stale frames — so the inference thread always processes the most recent camera frame, not a backlog.

### Heartbeat

A lightweight daemon thread runs every 10 seconds to keep the process alive on platforms that may reclaim idle worker processes. It performs no I/O and has no observable side effects.

---

## Deployment
The web app is accessible internally on port **9501**.

```bash
# To start the platform
python isitec_app/app.py
```

It can also be deployed via Docker for industrial servers:
```bash
docker build -t isitec-platform ./isitec_app
docker run -p 9501:9501 isitec-platform
```

# Performance Monitor

The `PerformanceMonitor` collects real-time metrics for the Performance dashboard. It tracks hardware usage, inference throughput, detection quality, and tracking health across inference sessions.

---

## Architecture

:material-file-code: **Source**: `webapp/isitec_app/performance_monitor.py`

The monitor is instantiated once in `StreamHandler.__init__()` and lives for the lifetime of the Flask process. Each new inference session calls `start_session()` to reset counters.

### Metric Hooks

Call these from the inference loop in `stream_handler.py`:

| Hook | When to Call | What It Records |
|---|---|---|
| `start_session(model_type)` | Stream start | Baseline hardware snapshot, reset counters |
| `track_frame(latency, stages, dets)` | Every frame | FPS, latency, confidence, mask coverage |
| `track_frame_drop()` | Reader queue empty | Frame drop counter |
| `track_error(is_oom)` | Inference exception | Error/OOM counters |
| `track_crossing()` | Line-crossing event | Crossing counter |
| `track_udp_publish()` | After UDP send | UDP counter |
| `track_csv_write(success)` | After CSV save | CSV success/failure |
| `heartbeat()` | Every N frames | Liveness timestamp |

---

## Hardware Detection

The monitor **adapts to the available hardware**:

### GPU Mode (NVIDIA GPU detected)

Shows VRAM usage, GPU utilization, GPU temperature, and deltas from baseline.

### CPU-Only Mode (no GPU)

Shows CPU utilization, CPU frequency, CPU cores, and CPU temperature (Linux only via `psutil.sensors_temperatures()`).

System RAM is shown in both modes.

### Baseline Deltas

At `start_session()`, the monitor snapshots VRAM, RAM, GPU temp, and GPU util **before the model loads**. The dashboard then shows the increase caused by inference:

```
VRAM: 3.2 / 6.0 GB  [+1.8 GB]
GPU Temp: 68°C       [+12°C]
```

---

## Status Thresholds

Each metric group has a traffic-light status (green / yellow / red):

### Hardware

| Metric | Yellow | Red |
|---|---|---|
| VRAM % | > 70% | > 85% |
| GPU Util | > 80% | > 90% |
| GPU Temp | > 75°C | > 85°C |
| CPU Util | > 80% | > 95% |
| CPU Temp | > 75°C | > 90°C |
| System RAM | > 85% | > 95% |

### Throughput

| Metric | Yellow | Red |
|---|---|---|
| FPS | < 20 | < 10 |
| Latency | > 50ms | > 100ms |

### Detection Quality

| Metric | Yellow | Red |
|---|---|---|
| Avg confidence | < 0.75 | < 0.55 |
| Low-conf rate | > 10% | > 25% |

### Tracking

| Metric | Yellow | Red |
|---|---|---|
| ID ratio (IDs / crossings) | > 2.0 | > 5.0 |

### Counting

| Condition | Status |
|---|---|
| Counts incrementing | Green |
| > 5 min with zero counts | Yellow |
| > 10 min with zero counts | Red |

---

## Session History

End-of-session summaries are saved to `webapp/isitec_app/logs/sessions.json` (last 10 sessions). Each entry records model type, duration, FPS, average confidence, ID ratio, class counts, and the baseline hardware snapshot.

---

## Dev-Only Access

The Performance and Settings tabs are hidden from operators. Dev team members double-click the logo to enter the password and unlock these pages. See the [Web Platform](../web-app/index.md) docs for details.

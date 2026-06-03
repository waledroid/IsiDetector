# The Dashboard

Open the browser at `http://localhost:9501` to reach the Live Inference page — the operator's main screen.

---

## Layout at a glance

The page has two areas side-by-side:

- **Left panel** — live counts, analytics chart, mode selector, source selector.
- **Right panel** — annotated video, Start / Stop buttons, optional ROI buttons.

A header bar holds the logo, navigation tabs, and the language toggle.

---

## Live counts and stats feed

Three count cards update in real time while a stream is running:

| Card | What it shows |
|---|---|
| **Cartons** | Running total of cartons that crossed the counting line this session |
| **Polybags** | Running total of polybags |
| **Last Detected** | Class name, ByteTrack ID, and timestamp of the most recent crossing |

The footer bar shows the last UDP event dispatched — `UDP → CARTON #42 @ 14:23:45` — or `UDP: idle` when no stream is active.

### How stats arrive

On FastAPI (default), the browser opens a `ws://…/ws/stats` WebSocket on Start. The server pushes a JSON tick every 500 ms. The cards update from each tick — no page polling.

If WebSocket is unavailable, the page falls back to polling `/api/stats` directly.

!!! note "Session restore on page reload"
    If the stream is already running when the browser loads (e.g., after the 30-minute auto-reload), the page reconnects to the running session automatically and repopulates the count cards.

---

## Video feed

The annotated video appears on a canvas element on the right.

On FastAPI, the browser opens `ws://…/ws/video`. The server sends binary JPEG frames capped at ~30 fps. Frames are silently dropped if the browser can't keep up — the stream never stalls.

If WebSocket is unavailable, the page falls back to the MJPEG stream at `/video_feed`.

The canvas resizes to the frame's aspect ratio (longest side fills 640 px). If an ROI crop is active, the canvas shows only the cropped area.

---

## Starting and stopping a stream

1. Select a **Mode** (see below).
2. Select an **Input Source** (see below).
3. Click **Start**. The stream opens, the video appears, counts begin updating.
4. Click **Stop** to halt the stream. Counts are preserved until the next Start.

Clicking Start while a stream is already running with the same source triggers a **hot-swap** — the model changes without dropping the RTSP connection or resetting counts.

Clicking Start with a different source restarts the session fully (counts reset).

---

## Mode selection

The **Modes** dropdown selects the inference backend:

| Option | Backend | When available |
|---|---|---|
| Mode 1 | YOLO (CNN) | Always |
| Mode 2 | RF-DETR (Transformer) | GPU hosts only |

On CPU-only site PCs, Mode 2 is hidden. The weights file used for each mode is set in [Settings](../settings/index.md). If no weights are saved, the backend auto-discovers the newest compatible file on disk.

---

## Input source selection

Four source buttons appear below the mode dropdown:

| Button | Source type | Notes |
|---|---|---|
| **Site Camera** | Saved RTSP URL | Default. Uses the URL saved in Settings → Camera. |
| **Image** | Uploaded JPG / PNG | File-picker or drag-and-drop. Processes once, freezes the result. |
| **Video** | Uploaded MP4 | Plays frame-by-frame in a loop with AI overlays. |
| **USB Camera** | Local device index | Enter the device index (e.g., `0`). |

!!! tip "Site Camera is the normal daily choice"
    The **Site Camera** button uses the RTSP URL saved in Settings. The operator never types an address on the Live page — just click Site Camera → Start.

---

## Analytics chart

A small bar chart is embedded in the left panel below the count cards. Use the filter buttons to switch views:

| Filter | What it shows |
|---|---|
| **Live** | Current session's per-class counts, updated every 500 ms from `/ws/stats` |
| **24h** | Hourly buckets for the last 24 hours, auto-refreshed every 5 s |
| **7 Days** | Daily buckets for the last 7 days |
| **30 Days** | Daily buckets for the last 30 days |

For production summaries, exports, and session comparisons see [Analytics](analytics.md).

---

## Language switch

Two flag buttons in the header (FR / EN) switch the UI language instantly without a page reload. The selection is also sent to the backend so log and stream messages match.

French is the default on load.

---

## ROI buttons (crop / clear)

The **Set ROI** (crop icon) and **Clear ROI** (scissors icon) buttons are hidden by default. They appear only when **Show "Set ROI" button** is enabled in Settings → Camera.

To draw an ROI:

1. Start the stream.
2. Click the crop button. The stream pauses and a snapshot appears.
3. Click and drag a rectangle over the conveyor belt area.
4. Click **Save ROI**. The crop takes effect on the next Start.

To remove the crop, click the scissors button. Full-frame mode is restored on the next Start.

!!! warning "Stop and Start to apply"
    Saving or clearing an ROI does not affect a stream that is currently running. Stop and Start to apply the change.

---

## Kiosk 30-minute auto-reload

The page reloads itself every 30 minutes via an HTTP `meta refresh` tag. This is a browser-level reload — it fires even if the JavaScript event loop is sluggish.

The backend stream, counts, event log, and UDP publishing continue uninterrupted during the reload. On return, the page reconnects and re-syncs automatically.

No operator action is needed. The reload is a memory-hygiene measure for long kiosk shifts (Chrome video buffers and JS heap can grow over several hours).

---

## Settings and tuning

The **Settings** and **Performance** tabs are hidden by default and require a developer password to unlock (double-click the logo). Operators do not need them for daily operation.

For model weights, confidence thresholds, tracking line position, UDP target, and CLAHE preprocess, see [Settings & Tuning](../settings/index.md).

# Settings Reference

Complete reference for every operator-tunable setting in `settings.json`.

---

## How the Settings page works

The Settings page is **developer-gated**. Click the lock icon in the header, enter the dev password, and the Settings panel unlocks. Without the token, `POST /api/settings` returns HTTP 403.

Settings are persisted to `webapp/isitec_api/settings.json` (FastAPI) inside the Docker image layer. Every `POST /api/settings` saves the current values to that file so they survive a container restart.

!!! tip "Protect your settings across git pulls"
    Set the skip-worktree marker once after first clone so `git pull` never overwrites your camera URL, ROI, or UDP target:

    ```bash
    git update-index --skip-worktree webapp/isitec_api/settings.json
    ```

    To revert to upstream defaults:

    ```bash
    git update-index --no-skip-worktree webapp/isitec_api/settings.json
    git checkout -- webapp/isitec_api/settings.json
    ```

### Server-written keys

`last_model_type` and `last_weights` are **written by the server** on every successful Start. The client cannot set them via the Settings UI — any attempt is silently stripped. They are used only by the [`auto_start`](#auto_start) path.

### Live-retarget keys

`udp_host` and `udp_port` take effect **immediately** on save — no stream restart needed. `dedup_time_enabled` / `dedup_interval_ms` and `count_interpolate` also apply live if the stream is already running.

---

## Settings table

### Camera

| Key | Type | Default | Meaning |
|---|---|---|---|
| `rtsp_url` | string | *(per site)* | RTSP stream URL used by the Site Camera button. Must start with `rtsp://` or `rtspt://`, max 512 chars. |
| `auto_start` | bool | `false` | If `true`, container boot replays the last successful Start automatically. See [Auto-start](#auto_start) below. |

### Model weights

| Key | Type | Default | Meaning |
|---|---|---|---|
| `yolo_weights` | string (path) | *(per build)* | Path to the YOLO model file selected by the operator; used on next Start. |
| `rfdetr_weights` | string (path) | `""` | Path to the RF-DETR model file. RF-DETR `.xml` (OpenVINO) is rejected at save time — use `.onnx` or `.pth`. |

See [Models & Modes](models-modes.md) for extension-to-backend mapping and per-mode allowed formats.

### Confidence thresholds

| Key | Type | Default | Meaning |
|---|---|---|---|
| `yolo_conf` | float | `0.3` | Detection confidence threshold for YOLO inference (0.0–1.0). |
| `detr_conf` | float | `0.3` | Detection confidence threshold for RF-DETR inference (0.0–1.0). |

### Counting line

| Key | Type | Default | Meaning |
|---|---|---|---|
| `line_orientation` | `vertical` / `horizontal` | `vertical` | Direction of the counting line across the frame. |
| `line_position` | float [0.0–1.0] | `0.5` | Position of the line as a fraction of frame width (vertical) or height (horizontal). After ROI crop, relative to the cropped frame. |
| `belt_direction` | `left_to_right` / `right_to_left` / `top_to_bottom` / `bottom_to_top` | `left_to_right` | Belt travel direction; selects the bbox leading-edge anchor for the trigger. |

See [Counting & Line Setup](counting.md) for trigger semantics and anchor mapping.

### UDP output

| Key | Type | Default | Meaning |
|---|---|---|---|
| `udp_host` | string | `10.0.0.1` | IP address or hostname of the sorter controller. Live-retargeted on save. |
| `udp_port` | int [1–65535] | `9502` | UDP port on the sorter controller. Live-retargeted on save. |

The UDP payload on every line-crossing event is:

```json
{"class": "carton", "seq": 42, "id": 7, "ts": "2026-06-03T08:14:22.341500"}
```

`seq` is a gap-free monotonic sequence number per session. `id` is the ByteTrack tracker ID. See [Counting & Line Setup](counting.md) for consumer-side implementation.

### Deduplication

| Key | Type | Default | Meaning |
|---|---|---|---|
| `dedup_time_enabled` | bool | `false` | Enable time-window deduplication: suppresses a second UDP event for the same class within `dedup_interval_ms` ms. Track-ID deduplication is always active regardless of this flag. |
| `dedup_interval_ms` | int [0–60000] | `300` | Minimum gap in ms between two UDP events for the same class (only used when `dedup_time_enabled` is `true`). |

See [Counting & Line Setup](counting.md) for full dedup behaviour.

### Counting interpolation

| Key | Type | Default | Meaning |
|---|---|---|---|
| `count_interpolate` | bool | `true` | Interpolate detections across frames where the tracker loses a track briefly. Reduces missed counts on fast belts or low-FPS streams. |

See [Counting & Line Setup](counting.md) for details.

### Tracker

| Key | Type | Default | Meaning |
|---|---|---|---|
| `tracker_fps_auto` | bool | `true` | Auto-calibrate ByteTrack's internal frame-rate assumption to the measured camera FPS. Recommended: leave enabled. |
| `tracker_fps` | float [0–120] | `0` | Override value used when `tracker_fps_auto` is `false`. `0` means "use auto". |
| `track_buffer` | int [1–600] | `60` | Number of frames ByteTrack holds a lost track before discarding it. At 25 FPS, `60` = 2.4 s of track memory. |

!!! note "tracker_fps_auto"
    When enabled, ByteTrack's velocity model matches the actual camera frame rate. Disabling it and setting `tracker_fps` manually is only needed if the auto-measured FPS is unstable (e.g. heavily compressed RTSP streams with variable frame delivery).

### ROI (Region of Interest)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `roi_enabled` | bool | `true` | Show the crop / clear ROI buttons on the Live Inference page. The crop button (rectangle icon) sets the ROI; the scissors button clears it. |
| `roi_points` | list of 4 `[x, y]` pairs, or `[]` | *(per site)* | Four corner points in original camera-frame pixel coordinates. Empty list = no crop applied. Must be exactly 0 or 4 entries; coordinates 0–8192. |

See [ROI & Image Pre-processing](image-roi.md) for the 4-click configurator workflow and performance impact.

### CLAHE / SpecularGuard

| Key | Type | Default | Meaning |
|---|---|---|---|
| `clahe_enabled` | bool | `false` | Enable adaptive contrast enhancement (CLAHE on the LAB L-channel) to reduce specular glare on polybag surfaces. Has a small CPU cost per frame. |

See [ROI & Image Pre-processing](image-roi.md) for guidance on when to enable CLAHE.

---

## Auto-start {#auto_start}

When `auto_start` is `true`, the container replays the last successful Start automatically ~5 s after the HTTP server binds.

**One-time setup:**

1. Unlock Settings and tick **Auto-start stream on boot**, then Save.
2. On the Live Inference page, click **Start** once and confirm the stream is running.
   The server records `last_model_type` and `last_weights` into `settings.json`.
3. From then on, every container restart resumes without operator action.

**Fail modes degrade cleanly** — the stream is never forced into a bad state:

- `last_weights` recorded but file deleted → log and skip; operator clicks Start manually.
- Camera unreachable at boot → existing TCP/UDP retry in the stream reader.
- `last_model_type` not yet set → log and skip.

!!! warning "Rebuild the image after a code-changing pull"
    Python source and templates are baked into the image at build time, not bind-mounted. After `git pull`, running `./up.sh` (which rebuilds) picks up new code. Running `./up.sh --no-build` keeps the old image — new `settings.json` keys will be present but the corresponding feature code will be missing. The rebuild is fast and offline-safe; only the small COPY layer re-runs.

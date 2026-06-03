# First Run & Camera Setup

Connect a camera, load a model file, and get the stream running for the first time.

---

## 1. Find the camera's RTSP URL

IP cameras on site do not advertise their URL automatically. Use one of the methods below to find it.

**Option A — check the camera label or commissioning sheet.**
Most Hikvision and Dahua units ship with a sticker listing the default IP and admin credentials.

**Option B — scan the camera subnet.**

```bash
# Replace 192.168.1.0/24 with the actual camera subnet
nmap -p 554 --open 192.168.1.0/24
```

Any host with port 554 open is a likely RTSP camera. Note its IP.

**Option C — use `./net.sh show`** to confirm which NIC is on the camera subnet, then scan that range.

!!! tip "Default admin credentials"
    Many site cameras ship with `admin` / `admin123` or `admin` / `12345`. Change them after commissioning.

### RTSP URL shape

The typical Hikvision URL pattern is:

```
rtsp://admin:PASSWORD@CAMERA_IP:554/user=admin&password=PASSWORD&channel=1&stream=0.sdp?
```

| Parameter | Values | Notes |
|---|---|---|
| `stream=0` | Main stream | Full resolution (e.g. 1920×1080). Higher bandwidth. |
| `stream=1` | Sub-stream | Lower resolution (e.g. 640×480). Useful if CPU cannot keep up. |
| `channel=1` | Channel index | Usually 1 for single-lens cameras; multi-head units may use 2, 3… |

!!! note "Main vs sub-stream"
    Start with `stream=0`. If you see frame drops or the inference FPS stays below 10 on a CPU host, switch to `stream=1` — the model runs at 320×320 internally regardless, so the resolution loss rarely affects accuracy.

**Test the URL before entering it in the dashboard:**

```bash
# Requires ffmpeg installed on the host (outside Docker)
ffplay "rtsp://admin:PASSWORD@192.168.1.108:554/user=admin&password=PASSWORD&channel=1&stream=0.sdp?"
```

You should see live video within 3–5 seconds. A black screen or immediate error means the URL or credentials are wrong.

---

## 2. Enter the URL in Settings

1. Open the dashboard at `http://localhost:9501`.
2. Navigate to **Settings → Camera**.
3. Paste the RTSP URL into the **Camera URL** field.
4. Click **Save**.

The URL is written to `webapp/isitec_api/settings.json` (`rtsp_url` key) and persists across container restarts.

!!! warning "Keep the URL out of the landing page"
    The **Site Camera** button on the Live Inference page sends an empty source — the saved URL is picked up automatically. Do not type the URL on the landing page; edit it only in Settings → Camera to avoid transcription errors.

---

## 3. Drop in a model file

Model files are expected under `isidet/models/yolo/` (YOLO) or `isidet/models/rfdetr/` (RF-DETR). They are mounted into the container at the same relative path. Discovery walks `isidet/models/yolo/**/weights/`, so a YOLO model dropped anywhere else will not be found.

**Copy the model file to the site PC** (USB stick, `scp`, or `rsync`):

```bash
# Example — copy a YOLO OpenVINO export
scp -r office:/path/to/yolo26n_320_200/weights/openvino \
    ~/fps/isidet/models/yolo/yolo26n_320_200/weights/
```

The container does **not** need to be restarted after a file copy — the file browser in **Settings → Model** reads the filesystem live.

| Host mode | Preferred format | Why |
|---|---|---|
| CPU (OpenVINO) | `.xml` + `bin` pair | Fastest on Intel; `.pt` and `.pth` are rejected in CPU mode |
| GPU (CUDA) | `.pt` or `.onnx` | TensorRT `.engine` also accepted if pre-compiled for this GPU |

!!! warning "CPU mode rejects `.pt` and `.pth`"
    The container detects its mode from the `COMPOSE_MODE` env var set by `up.sh`. If you see "Model extension not supported in CPU mode", export the weights via `compress.sh` on the office GPU workstation and copy the resulting `.xml` + `.bin` pair. See [Model formats & modes](../settings/models-modes.md).

---

## 4. Select the model in Settings

1. **Settings → Model 1** — choose model type (YOLO or RF-DETR) and select the weight file from the dropdown.
2. Click **Save**.

The dropdown groups files by format (OpenVINO / ONNX / PyTorch / TensorRT). The file selected here is recorded as `yolo_weights` or `rfdetr_weights` in `settings.json`.

---

## 5. Start the stream

1. Go to **Live Inference**.
2. Click the **Site Camera** button (the camera-icon button, not the URL field).
3. Click **Start**.

The inference engine loads, ByteTrack initialises, and the video feed appears within a few seconds. The mode badge in the top-right corner of the feed shows the active backend (e.g. `OpenVINO • CPU` or `YOLO • GPU`).

!!! note "First start is slower"
    On GPU hosts the ONNX CUDA kernel cache is empty on the first run — expect 5–15 s before the stream appears. Subsequent starts reuse the primed cache and load in ~2 s.

**If the feed stays black or shows STANDBY:**

- Check `docker compose logs -f web` for connection errors.
- The reader retries automatically on disconnect — wait 5 s before concluding the URL is wrong.
- Verify the camera is reachable: `ping CAMERA_IP` from the host.

---

## 6. Verify everything is working

| What to check | Where |
|---|---|
| Frame rate (target ≥ 15 fps on CPU) | Mode badge or **Settings → Performance** |
| Detections visible on belt | Live feed overlay |
| UDP datagrams reaching the PLC | **Settings → UDP** — `seq` counter increments on each crossing |

Once the stream is stable, configure the counting line and ROI crop to match the belt geometry — see [Counting line & triggers](../settings/counting.md) and [Image & ROI](../settings/image-roi.md).

To make the stream resume automatically after a reboot without any operator click, enable **Auto-start** — see [Starting & stopping](../daily-ops/start-stop.md).

---
hide:
  - navigation
---

# IsiDetector

Counts cartons and polybags crossing a conveyor line, and fires a UDP sort trigger to the PLC on every crossing.

---

## What it does

A camera watches the conveyor belt. IsiDetector detects each **carton** (green) and **polybag** (orange) in real time, assigns it a track ID, and fires a UDP datagram the instant the parcel's leading edge crosses the counting line. The sorter PLC receives the event and actuates the gate — no polling, one packet per parcel.

Each datagram looks like this:

```json
{"class": "carton", "seq": 17, "id": 42, "ts": "2026-06-03T08:14:22.481203"}
```

| Field | Meaning |
|---|---|
| `class` | `"carton"` or `"polybag"` |
| `seq` | Gap-free monotonic counter. Any gap means a lost packet. Resets on stream restart. |
| `id` | ByteTrack ID — not sequential; gaps are normal and do not indicate loss. |
| `ts` | Microsecond ISO timestamp. |

Default UDP target: **`10.0.0.1:9502`**. Change it live from Settings without restarting the stream.

!!! note "Trigger timing"
    The datagram fires on the **leading edge** of the bounding box — the side that enters the line first given belt direction. This maximises the PLC's reaction window.

---

## Site-PC workflow at a glance

| Step | What to do |
|---|---|
| **Install** (once per host) | Clone the `fps` branch and run `./run_start.sh` — see [Getting Started](getting-started/install.md) |
| **Daily start** | `cd ~/fps && ./up.sh` — stack starts, browser opens at `http://localhost:9501` |
| **Stop** | `docker compose down` |
| **Update** | `cd ~/fps && git pull && ./up.sh` |

The web UI runs in Docker. Two containers:

- **`web`** — Flask or FastAPI + ONNX Runtime, Ultralytics YOLO, OpenVINO. Ports `9501` (HTTP) and `9502` (UDP out).
- **`rfdetr` sidecar** — GPU hosts only; skipped on CPU-only hardware.

!!! tip "Hands-free kiosk"
    Enable autostart so the stack and browser come up on power-on with zero operator clicks. See [Daily Operation](daily-ops/start-stop.md) for the `autostart.sh` one-command setup.

---

## Hardware support

| Platform | Backend | RF-DETR |
|---|---|---|
| Ubuntu 22/24 — NVIDIA GPU | TensorRT / ONNX-CUDA | Available |
| Ubuntu 22/24 — CPU only | OpenVINO (fastest on Intel) | Not available |
| Windows 10 (build 19041+) / 11 — CPU only | OpenVINO | Not available |

CPU mode is **YOLO-only**. RF-DETR and TensorRT engines are skipped automatically — no configuration needed.

---

## Where to go next

| I want to… | Go to |
|---|---|
| Install on a new site PC | [Getting Started — Install](getting-started/install.md) |
| Start the stack and run inference | [Daily Operation](daily-ops/start-stop.md) |
| Configure the line, camera, model, or UDP target | [Settings](settings/index.md) |
| Set the static IP and verify UDP reaches the PLC | [Network & UDP](network/udp.md) |

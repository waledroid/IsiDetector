# `fps` branch — site-PC operability + perf features

> **Branch status note**: this content is drafted on `fps`; the canonical
> mkdocs source lives on `main`. Copy this file under
> `mkdocs/docs/operations/` (or wherever your operations section is) when
> updating main. The same content is also referenced from
> `CLAUDE.md` on the `deploy` and `fps` branches.

This page summarises everything the `fps` branch adds on top of `deploy`,
in operator-facing order. Each subsection is independent and can be
adopted or rolled back without affecting the others.

---

## 1. Network setup — `./net.sh setup`

Interactive multi-NIC freeze for the typical site PC layout: two LAN
NICs (e.g. `enp1s0` for the camera subnet, `enp2s0` for the automate
subnet), no default gateway, no internet uplink.

```bash
sudo ./net.sh setup
```

Walks every physical NIC the host knows about (including down/unplugged
ones via `ip -o link`), prompts per NIC for `static / dhcp / skip`, then
writes via NetworkManager with `autoconnect=yes` + `autoconnect-priority=100`
so the config survives reboot. Gateway and DNS are optional in the static
path — blank is fine on a no-uplink site.

The legacy `./net.sh apply` (single-NIC freeze) still exists for normal
LANs with a default gateway. `./net.sh show` and `./net.sh test` are
**offline-clean** — internet ping in `test` is a yellow `skip` instead
of a red fail when the WAN is unplugged, and Docker-egress checks gate
on the web container actually running.

Reverse:

```bash
sudo ./net.sh setup          # re-run and pick "dhcp" or "skip" per NIC
```

---

## 2. Standalone-mode helpers — `./autostart.sh`

Three independent layers turn a fresh site PC into a hands-free kiosk.

```bash
sudo ./autostart.sh enable-autologin $USER   # Layer 1 — OS skips login screen
sudo ./autostart.sh enable-systemd           # Layer 2 — docker compose at boot
./autostart.sh enable                        # Layer 3 — kiosk Chrome on login

./autostart.sh status                        # confirm all three
sudo reboot                                  # apply Layer 1
```

| Layer | Where it writes | What it does | Reverse |
|---|---|---|---|
| 1 — auto-login | `/etc/gdm3/custom.conf` (or LightDM/SDDM) | OS auto-logs the operator in at boot, no password prompt. Display manager is auto-detected. | `sudo ./autostart.sh disable-autologin` |
| 2 — systemd | `/etc/systemd/system/isidetector.service` | `docker compose up -d` runs at boot, after `docker.service` + `network-online.target`. Stack is up before the desktop session even loads (~30 s saved on cold boot). | `sudo ./autostart.sh disable-systemd` |
| 3 — desktop autostart | `~/.config/autostart/isidetector.desktop` | Desktop session opens kiosk Chrome on `http://localhost:9501` ~10 s after login. Auto-rewrites itself to use `up.sh --open-only` when Layer 2 is also enabled (no compose race). | `./autostart.sh disable` (also removes Layer 2) |

Combined with the in-app **Auto-start stream on boot** toggle (Section
3), the full hands-free path is **~30–40 s from power-on to the stream
running, zero clicks**.

`up.sh --open-only` is a new flag that skips compose entirely, waits
briefly for `tcp/9501`, and opens Chrome — used by Layer 3 when
Layer 2 owns the compose lifecycle.

---

## 3. Auto-start stream on container boot

Settings-driven toggle that makes the container replay the last
successful Start every time it boots. No operator click on the
dashboard, no need to remember model selection.

**One-time setup** (do once after deploying the feature):

1. **Settings → Camera** → tick **"Auto-start stream on boot"** → Save
2. **Live Inference** → click **Start** once
   (this records `last_model_type` + `last_weights` into `settings.json`)
3. Confirm the stream is running normally

From then on, every container restart auto-resumes within ~10 s of
the HTTP server binding.

**Behind the scenes**: `stream_handler.py` spawns a daemon thread at
init. After a 5 s grace period (lets the WSGI/ASGI server bind), it
reads `settings.json` and — if `auto_start=true` and all three of
`rtsp_url` / `last_model_type` / `last_weights` are set — calls
`self.start(source="", model_type=..., weights=...)`. Source empty
falls back to the saved `rtsp_url` via the existing 📡 Site Camera
button machinery.

**Fail modes degrade cleanly** (no crash, no UDP spam):

- Box ticked but `last_model_type` not yet recorded → log + skip
- Model file deleted between boots → log + skip; operator clicks Start manually
- Camera unreachable → existing TCP-first/UDP-fallback retry in `LiveReader`

---

## 4. ROI crop — Live-page 4-click belt configurator

Region-of-interest crop applied **before** the pre-engine resize. Same
model input size (320×320 with the shipped OpenVINO `.xml`) but the
pixels going into it are now the belt area instead of the full camera
view.

### Why

At `imgsz=320` with a 1920×1080 RTSP stream, a parcel becomes ~8 px wide
in the model input → carton/polybag class flips at the trigger line.
After cropping to e.g. 1100×500, that same parcel becomes ~14 px wide →
class stays stable. **Density gain ≈ original_width / cropped_width.**

| Original | Cropped to | Gain | Parcel pixel width (was 8 px) |
|---|---|---|---|
| 1920×1080 | 1500×800 | 1.28× | ~10 px |
| 1920×1080 | 1100×500 | 1.75× | ~14 px |
| 1920×1080 | 800×400 | 2.4× | ~19 px |

Bonus: the `line_position` slider operates relative to the cropped
frame, so `0.5` finally means "middle of the belt" instead of
"middle of the camera frame."

### Operator flow

1. **Settings → Camera** → tick **"Show 'Set ROI' button on landing page"** → Save
2. **Live Inference** — the **📐 Set ROI** button appears next to Start/Stop
3. With a stream running, click **📐 Set ROI**:
    - Snapshot freezes under the click area
    - Click 4 corners of the belt region (any order)
    - System computes the axis-aligned bounding rectangle
    - **Save ROI** → POST to `/api/settings`
4. **Stop and Start the stream** to apply the crop

The kiosk preview now shows just the cropped belt region (much
sharper); detections fire only inside it; line crossings still trigger
UDP exactly as before.

### Performance

ROI is **net-cheaper on CPU**, not heavier:

- Numpy slice `frame[y1:y2, x1:x2]` is a view, ~0 ms cost
- The existing `cv2.resize` then operates on a smaller source (e.g.
  1100×500 → 320×145 instead of 1920×1080 → 320×180) — saves ~2–3 ms/frame
- Inference cost unchanged (always 320×320)
- JPEG encode slightly cheaper (smaller annotated frame)

### Fail-safe

Any error reading or applying the ROI (off-frame coords, empty crop,
JSON malformed, frame dimensions changed) → log once, set
`self.roi = None` for the rest of the session, pipeline continues
with full frame. The stream is **never broken** by a bad config.

### `/api/snapshot`

New endpoint: `GET /api/snapshot` returns one full-resolution raw
camera frame as JPEG (latest from `LiveReader.get_frame()`). Used by
the ROI configurator. Returns 404 when the stream isn't running.

---

## 5. OpenVINO YOLO preprocess — resize-first

Internal optimisation, no operator-facing change. Mentioned for
completeness because it shows up in the per-frame timing breakdown.

`openvino_inferencer.preprocess()` was reordered so `cvtColor` runs on
the model-sized canvas (320×320), not the raw 1080p frame. Plus the
old `transpose + astype + /255 + add-batch` numpy chain is replaced by
`cv2.dnn.blobFromImage` (a single fused C++ SIMD pass with `swapRB=True`).

Saves ~2 ms/frame on i7-10710U at 1080p input. Output shape is
unchanged (NCHW float32 with batch dim) so no other code paths needed
to change.

---

## 6. Settings keys reference

All operator-tunable parameters live in `webapp/isitec_app/settings.json`
(and the symmetric `webapp/isitec_api/settings.json` for the FastAPI
backend). The Settings UI reads/writes via `GET`/`POST /api/settings`.

| Key | Type | Default | Purpose |
|---|---|---|---|
| `yolo_weights` / `rfdetr_weights` | str (path) | per-build | Model file the operator selected; respected on next Start. |
| `yolo_imgsz` / `detr_imgsz` | int | 320 / 416 | Inference input size hint. **OpenVINO `.xml` ignores this** (input shape is baked in at export); Ultralytics `.pt` and dynamic ONNX honor it. |
| `yolo_conf` / `detr_conf` | float | 0.55 / 0.35 | Confidence threshold. |
| `line_orientation` | `vertical` / `horizontal` | vertical | LineZone orientation. |
| `line_position` | float [0..1] | 0.5 | Fraction of frame width (vertical) or height (horizontal). After ROI, relative to the cropped frame. |
| `belt_direction` | `left_to_right` / `right_to_left` / `top_to_bottom` / `bottom_to_top` | left_to_right | Picks the bbox leading-edge anchor. |
| `cpu_threads` | int [1..64] | 8 | OpenVINO `INFERENCE_NUM_THREADS`. |
| `skip_masks` / `skip_traces` | bool | false / false | Render shortcuts; significant FPS bump on busy belts. |
| `rtsp_url` | str | per-build | Saved camera URL used by the **📡 Site Camera** landing-page button. |
| `udp_host` / `udp_port` | str / int | 10.0.0.1 / 9502 | Sorter target. **Live-retargets** on save. |
| `auto_start` | bool | false | If true, container boot replays the last successful Start. |
| `last_model_type` / `last_weights` | str / str | "" | **Server-written** (rejected from client POST). Used by `auto_start`. |
| `roi_enabled` | bool | false | If true, exposes the **📐 Set ROI** button on the Live Inference page. |
| `roi_points` | list of 0 or 4 `[x,y]` pairs | [] | Operator-drawn corner points in original camera-frame pixel coords. |

---

## 7. `settings.json` and `git pull`

The on-site `settings.json` diverges from upstream every time we add a
new key here. To avoid `git pull` conflicts on the operator's
customised file, set the skip-worktree marker once after first clone:

```bash
git update-index --skip-worktree \
    webapp/isitec_app/settings.json \
    webapp/isitec_api/settings.json
```

After that, `git pull` leaves the operator's `settings.json` alone
forever. New upstream keys appear in their file the first time they
**Save** the Settings panel — the backend writes the merged set to disk.

To undo (rare — accept upstream's settings.json instead):

```bash
git update-index --no-skip-worktree \
    webapp/isitec_app/settings.json \
    webapp/isitec_api/settings.json
git checkout -- webapp/isitec_app/settings.json webapp/isitec_api/settings.json
```

---

## 8. Daily ops sequence on `fps`

```bash
cd ~/fps
git pull                                     # safe — settings.json is skip-worktree
sudo systemctl restart isidetector.service   # if Layer 2 is enabled
# (or: sudo docker compose down && ./up.sh --force-cpu)
```

That's it. The autostart layers + auto-resume + ROI persist across
pulls and reboots — they're configured once per site PC.

!!! warning "Don't use `--no-build` after a code-changing pull"
    The Python source (`stream_handler.py`, `app.py`,
    `openvino_inferencer.py`), templates, and JS are **baked into the
    image at build time** via `COPY` — they are NOT bind-mounted volumes.
    Running `./up.sh --no-build` after a pull keeps the **old image with
    the new `settings.json`** → silent feature failures (new JSON keys
    present, no UI, no `/api/snapshot`, no ROI crop, no auto-start).

    The rebuild is **fast and offline-safe**: `requirements-deploy.txt`
    rarely changes, so the dep layer cache hits and only the small COPY
    layer re-runs. No internet needed for the rebuild itself once the
    image has been built once on the host.

    `--no-build` IS the correct flag for the **boot-time autostart**
    path (`autostart.sh enable` writes it into the .desktop file) — at
    that point the operator just wants to bring up whatever image is on
    disk, with no network dependency. The `autostart.sh` defaults handle
    this for you; don't pass `--no-build` from a manual update flow.

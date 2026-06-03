# Image & ROI

Control which part of the camera frame the model sees and whether contrast enhancement is active.

---

## ROI — Region of Interest

### Why set an ROI?

The camera sees more than the conveyor belt: side walls, floor, ceiling lights. Cropping to just the belt gives the model a cleaner view with parcels at higher pixel density, reducing false detections and improving counting accuracy.

ROI is applied before any downscaling. The model never sees pixels outside the crop box.

### Enable the buttons

The crop and cut buttons are hidden by default. Reveal them once during site setup:

1. Open **Settings → Image** (top-right gear icon).
2. Tick **Show "Set ROI" button on landing page**.
3. Click **Save Settings**.

Two icon-only buttons appear in the control bar on the Live Inference page:

| Button | Icon | Action |
|---|---|---|
| **Set ROI** (blue) | `crop` | Open the ROI drawing tool |
| **Clear ROI** (orange) | `content_cut` | Wipe the saved crop, revert to full frame |

### Draw the crop box

The stream must be running before you draw an ROI — the tool takes a live snapshot as the canvas.

1. Start the stream.
2. Click the blue **crop** button.
3. A frozen snapshot of the full camera frame appears with a blue instruction banner.
4. Click and drag a rectangle over the conveyor belt area.
5. Click **Save ROI**.

The current bbox is shown in **Settings → Image** as `x=[x1,x2] y=[y1,y2] (W×H)` so you can verify the saved coordinates.

!!! note "Stop and Start to apply"
    The ROI is read once at stream start (`_load_roi`). After saving, click **Stop** then **Start** — the new crop takes effect immediately.

!!! tip "Redrawn any time"
    Click the blue **crop** button again to draw a new box. The old box is replaced when you click Save ROI. You do not need to clear first.

### Clear the ROI

Click the orange **content_cut** button. The saved bbox is wiped (`roi_points: []`). Then click **Stop** and **Start** — the stream runs full-frame.

---

## CLAHE / SpecularGuard

### What it does

SpecularGuard applies **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) to the lightness channel of the image — shadows are lifted and polybag glare is reduced, while parcel colours are left unchanged.

Fixed parameters (not operator-tunable): `clip_limit=2.5`, `tile_grid=8×8`.

### Default: OFF

CLAHE is **disabled by default** (`clahe_enabled = false` in `settings.json`).

Enable it in **Settings → Image**: tick **Apply CLAHE preprocess (glare / low-light correction)**.

!!! warning "CLAHE can suppress carton detections"
    On some conveyor lighting setups, the contrast boost causes the model to see textured carton surfaces differently, reducing carton confidence scores. If you turn on CLAHE and notice carton counts drop, turn it off and click Stop + Start.

!!! note "Stop and Start to apply"
    CLAHE is loaded once per session in `_build_preprocess_chain`. Toggle the checkbox, save, then click **Stop** and **Start** for the change to take effect.

### When to try it

Try CLAHE when:

- Polybag detections are inconsistent and the camera shows bright specular hotspots on plastic film.
- The conveyor is dark and parcels in shadow are being missed.

Leave it off when lighting is even and both classes are counting correctly.

---

## Settings reference

| Setting | Location | Default | Notes |
|---|---|---|---|
| Show Set/Clear ROI buttons | Settings → Image | Off | One-time installer setup; hides the buttons from daily operators when off |
| ROI crop box | Drawn on snapshot | None (full frame) | Persisted as `roi_points` in `settings.json`; Stop + Start to apply |
| CLAHE preprocess | Settings → Image | Off | Stop + Start to apply; monitor carton counts after enabling |

---

For counting line position and belt direction, see [Counting settings](counting.md).

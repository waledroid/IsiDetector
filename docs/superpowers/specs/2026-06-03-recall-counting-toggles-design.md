# Recall-focused line-crossing accuracy toggles

**Date:** 2026-06-03
**Branch:** `fps`
**Status:** design approved, pending implementation plan

## Problem

On the conveyor, **detection is good but the count is low** — some parcels physically
cross the line without ever being counted. The operator confirms the failure mode is
**under-count, not over-count**. Most misses happen at **low effective FPS** (~20, dipping
lower). The fix must therefore raise **recall** (catch every crossing) without disturbing
the automaticien's existing UDP/PLC integration.

## Root causes (evidence)

1. **Global time-dedup guard drops distinct parcels.** `DedupGate` (`isidet/src/shared/dedup_gate.py`)
   suppresses any crossing when `(now_ms - self._last_emit_ms) < interval_ms`, where
   `_last_emit_ms` is a **single timestamp shared across all tracks**. With the live config
   (`dedup_time_enabled=true, interval_ms=300`), a second parcel — *a different track* —
   crossing within 300 ms of the previous one is silently dropped. Parcels routinely cross
   <300 ms apart on a moving belt. The guard exists only to absorb ID-churn double-counts,
   which is the opposite of this site's problem. **It is pure downside here.**

2. **ByteTrack runs with the wrong frame rate.** The engine builds
   `sv.ByteTrack(track_activation_threshold=..., lost_track_buffer=..., minimum_matching_threshold=...)`
   at `isidet/src/shared/vision_engine.py:64` **without `frame_rate`**, so it defaults to
   30 fps while the stream runs ~20. The Kalman motion model under-predicts the real
   per-frame displacement of a fast parcel.

3. **ID switch across the line = missed crossing.** supervision's `LineZone` counts an
   object that "jumps" the line between frames *as long as the track keeps the same ID on
   both sides*. A fast parcel can move more than its own size between frames → its boxes
   don't overlap → IoU match fails → ByteTrack assigns a **new ID** → neither the old nor
   the new ID is ever seen on both sides → no crossing is registered. Root cause 2 makes
   this worse (under-predicted motion → more ID switches).

## Design — recall only

A new **"Counting accuracy"** group on the dev-gated Settings page (sibling to the existing
dedup controls), persisted in `settings.json`, applied **live** (no restart), in **both**
backends (Flask `isitec_app`, FastAPI `isitec_api`). Five knobs total.

### New behaviour (2 toggles, both default ON, both latency-neutral)

| Toggle | settings key | Default | Effect |
|---|---|---|---|
| Tracker FPS calibration | `tracker_fps_auto` (bool) + `tracker_fps` (float, 0 = auto) | **ON / auto** | Feed the measured stream FPS to `sv.ByteTrack(frame_rate=...)` so the Kalman predicts the true per-frame displacement → fast parcels keep their ID across the line. The core low-FPS fix. |
| Predicted-path crossing | `count_interpolate` (bool) | **ON** | Decide the crossing from the track's trajectory across consecutive updates (including the Kalman-predicted position) so a one-sided or gap detection at the line still registers, instead of requiring a clean detection on both sides. |

### Existing knobs retuned for recall (no new engine code — surfaced in the same UI block)

| Knob | settings key | New default | Why |
|---|---|---|---|
| Time-dedup guard | `dedup_time_enabled` | **OFF** | Removes the global suppressor in root cause 1. Track-ID dedup (`counted_ids`) still prevents a single track counting twice. |
| Lost-track buffer | `track_buffer` (int) | **60** (unchanged; try 90) | Exposed for A/B — raising it keeps a track alive through a 1–2 frame detection gap at the line so the ID survives the fast move. Default stays at the just-restored baseline (60, commit `7ec8df1`); 90 is the recommended recall experiment, flipped live on-site rather than forced. |
| Detection confidence | `yolo_conf` / `detr_conf` | operator-tunable, no forced change | Lower it to catch motion-blurred fast parcels; safe to lower because over-count isn't a concern. Already in Settings; cross-referenced from the new block. |

### Dropped (and why)

min-hits, min-displacement, crossing hysteresis (`minimum_crossing_threshold`), box
smoothing (`DetectionsSmoother`), dual-line band. **Every one of these *reduces* counts or
*delays* the trigger** — the wrong direction for an under-count problem. Hysteresis would
actively suppress single-frame fast crossings → more misses. They remain documented in the
SOTA list for a future over-count scenario but are out of scope here.

## Honoring the automaticien's requirements

- **`seq` contract intact.** Each emitted crossing still produces exactly **one gap-free
  `seq`** and **one UDP datagram**; recovered crossings get the next `seq`, suppressed ones
  never consume one. Monotonic, gap-free per stream — unchanged.
- **Leading-edge trigger untouched.** Filters wrap the count *decision*, never the anchor
  (`_ANCHOR_MAP` / `triggering_anchors`). His "début de l'objet / même position" requirement
  is preserved.
- **Timing window protected.** Both new toggles are latency-neutral — the crossing fires on
  (or interpolated to) the same instant it does today, so the PLC's Mini/Maxi_Camera window
  does not shift. (Turning the time-dedup guard OFF can only *add* counts that were being
  dropped; it never moves an existing trigger's timing.)

## Architecture

- **`isidet/src/shared/vision_engine.py`** — pass `frame_rate` to `sv.ByteTrack` (from the
  measured FPS supplied by the stream loop); add predicted-path crossing logic to the
  per-track crossing decision; add `engine.configure_counting(**opts)` for live apply
  (mirrors `dedup.configure`). Track-ID dedup and `seq` emission unchanged.
- **`isidet/src/shared/dedup_gate.py`** — unchanged logic; `dedup_time_enabled` simply
  defaults OFF via config.
- **`webapp/{isitec_app,isitec_api}/stream_handler.py`** — read the new `settings.json`
  keys, measure live FPS and pass it to the engine, inject into `ve_config`, live-apply.
- **`webapp/{isitec_app,isitec_api}/app.py`** — `/api/settings` validates + live-applies the
  new fields (same path the dedup fields already use).
- **`webapp/{isitec_app,isitec_api}/templates/index.html` + `static/js/main.js`** — the
  "Counting accuracy" UI block, dev-gated, both backends in parity.
- **`isidet/configs/inference/*.yaml`** — defaults (`tracker_fps_auto: true`,
  `count_interpolate: true`, `dedup_time_enabled: false`, `track_buffer: 60` unchanged).
- **`webapp/{isitec_app,isitec_api}/settings.json`** — default `dedup_time_enabled: false`.

## Data flow (unchanged except the count decision)

```
process_frame(frame):
  detections = inferencer(frame)
  detections = tracker.update_with_detections(detections)   # ByteTrack now frame_rate-calibrated
  crossings  = line_zone.trigger(detections)                # + predicted-path recovery
  for each crossed track id:
    if dedup.should_emit(id):       # track-ID dedup; time guard OFF by default
        seq += 1 ; emit UDP(class,id,seq) ; log CSV(class,id,seq) ; count++
```

## Testing & rollout

- **Eval harness (`tools/`):** hand-label true counts on the 3 site clips + `testvid`; run
  the stream and report **miss rate / under-count per class** for any toggle combination.
  This is the gate that judges each toggle and decides whether the deferred tracker swap
  (BoT-SORT/OC-SORT) is ever worth it.
- **Invariants:** `seq` stays gap-free; exactly one datagram per count; the two new toggles
  must not move the crossing frame index for parcels already counted today (assert vs.
  baseline) — they may only *add* recovered crossings.
- **Rollout:** all behind flags with the new defaults; commit prefixed `accuracy:` for a
  one-command revert; A/B on-site live via the Settings UI, no redeploy.

## Reversibility

Every change is a flag with a config default. Setting `dedup_time_enabled` back ON,
`count_interpolate` OFF, and `tracker_fps_auto` OFF restores exact current behaviour.

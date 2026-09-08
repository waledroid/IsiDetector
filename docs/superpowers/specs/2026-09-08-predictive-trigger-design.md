# Predictive line trigger — design (2026-09-08)

## Problem

The sorter datagram fires on the first processed frame after a parcel's
leading edge passes the counting line. On site (25 fps, CPU/OpenVINO) the
automaticien measures the lead between datagram and parcel-at-gate as
850–1351 ms at line 0.85 — a 500 ms spread against a 200 ms target window.
Frame quantisation is only ~40 ms; the spread comes from detection gaps at
the line (datagram late) and per-parcel geometry. Moving the line or adding a
fixed delay shifts the range, it does not narrow it.

## Goal

Fire the datagram at *predicted crossing time + offset_ms* from a timer
thread, so a dropped frame at the line no longer moves it and the lead is a
setting in milliseconds. Fully revertible: setting default off, code in one
module, commits prefixed `pred:`.

## Module — `isidet/src/shared/predictive_trigger.py`

`PredictiveTrigger(line_coord, after_is_greater, offset_ms, on_fire, clock)`

- `observe(track_ids, positions, classes, t_ms)` — per frame. Keeps the last
  12 `(t, pos)` samples per track id (leading-edge coordinate on the crossing
  axis, pixels; capture-time monotonic ms).
- Estimate: ≥3 samples → least-squares fit pos(t) → velocity `v`, smoothed
  position. Belt speed = running median of `v` over confident tracks (≥6
  samples). A track whose `v` deviates >50 % from belt speed uses belt speed
  with its latest position. `v` must point toward the "after" side, else no
  prediction.
- Crossing time `t_cross`: extrapolated while upstream; replaced by linear
  interpolation once two consecutive samples straddle the line. Fire time
  `t_fire = t_cross + offset_ms`.
- Scheduler: daemon thread + `threading.Condition`; waits for the earliest
  pending `t_fire`; `observe()` re-arms with refined times. On due, calls
  `on_fire(track_id, class_name)` outside the lock, marks id fired (never
  re-armed).
- `report_observed(track_id, class_name, t_ms)` — fallback for LineZone /
  latch hits on tracks without a prediction: arm at `t_ms + offset_ms`.
  Tracks already armed/fired: no-op.
- Cancel: armed track unseen for ≥10 frames whose last position was >20 % of
  the frame extent upstream of the line → dropped (false detection). Both
  constants module-level, not settings.
- `forget(keep_ids)` — drop state for ids absent ≥2 s. `configure(offset_ms)`
  live. `stop()` joins the thread.

## Integration

- **LiveReader** stamps each frame `time.monotonic()*1000` on queue put;
  `get_frame()` returns `(frame, t_ms)`. Both callers updated.
- **VisionEngine**: emit block factored into `_emit(track_id, class_name)`
  under `_emit_lock` (dedup → counts → seq → CSV → `on_event` callback).
  `process_frame(frame, class_totals, t_ms=None)`: when predictive on, feeds
  the predictor each frame and routes LineZone/latch hits to
  `report_observed`; returns `[]` for those. When off: unchanged behaviour.
  `configure_predictive(enabled, offset_ms)`; `on_event` callback attribute;
  predictor rebuilt on line/frame-size change (`_setup_line`), preserved on
  `swap_inferencer`.
- **StreamHandler**: per-event loop body → `_handle_event(event)` (last_detected,
  UDP publish, monitor). Called by the loop and by `engine.on_event`.
- **Logging**: one INFO line per fired event (`src=pred|obs`, offset, samples,
  v, est_err_ms when a later straddle refines the crossing); summary
  (median/p95 est_err) every 100 fires. Event CSV unchanged.

## Settings / UI (Flask only)

`predictive_trigger: bool=false`, `trigger_offset_ms: int=0` (−2000..2000).
Validated + allow-listed in `POST /api/settings`, live-applied via
`configure_predictive`. Dev-gated checkbox + number field next to "Time
dedup" in `index.html` / `main.js`.

## Testing

- `isidet/tests/test_predictive_trigger.py` (pytest, fake clock, no thread):
  constant velocity fires at crossing+offset; negative offset fires early;
  8-frame gap at line still fires at extrapolated time; straddle refines;
  far-upstream vanish cancels; appear-at-line falls back to observed+offset;
  no re-arm after fire; wrong-direction velocity → no prediction.
- Offline replay script on `isidet/data/cam_20260602_105526.mp4`: predicted
  vs observed fire time per parcel, spread stats.

## Rollout

`pred:`-prefixed commits on `fps`; pull on site as isi-linux; rebuild; enable
from Settings with offset 0, line 0.71; automaticien measures; tune offset.
Revert = untick, or `git reset` to the commit before the `pred:` series.

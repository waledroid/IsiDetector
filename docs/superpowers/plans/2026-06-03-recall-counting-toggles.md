# Recall-focused line-crossing toggles — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise line-crossing recall (stop under-counting parcels at low FPS) via two latency-neutral toggles (tracker FPS calibration + frame-gap-tolerant crossing) plus retuned existing knobs, all live-togglable from the dev Settings page, without disturbing the automaticien's `seq`/leading-edge/PLC-timing integration.

**Architecture:** A new pure `CrossingDetector` (before/after latch per track on the leading-edge anchor) supplements supervision's `LineZone` by OR-ing crossings into the *same* `DedupGate`/`seq` path — so recovered crossings never double-count and `seq` stays gap-free. `ByteTrack` is rebuilt with the measured stream FPS so its Kalman keeps fast-parcel IDs stable across the line. Existing knobs (time-dedup, `track_buffer`, conf) move into one "Counting accuracy" Settings block. Both Flask and FastAPI backends stay in parity.

**Tech Stack:** Python, supervision 0.28 (`sv.ByteTrack`, `sv.LineZone`, `sv.Detections.get_anchors_coordinates`), Flask + FastAPI, vanilla JS, Docker (CPU/OpenVINO).

**Spec:** `docs/superpowers/specs/2026-06-03-recall-counting-toggles-design.md`

---

## File structure

- **Create** `isidet/src/shared/crossing.py` — `CrossingDetector`, pure logic, no cv2/supervision import.
- **Create** `isidet/tests/test_crossing.py` — plain-python assert tests (repo has no pytest; run with `python`).
- **Modify** `isidet/src/shared/vision_engine.py` — store line geometry for crossing; use `CrossingDetector`; `frame_rate` into `ByteTrack`; `configure_counting()`.
- **Modify** `webapp/isitec_api/stream_handler.py` + `webapp/isitec_app/stream_handler.py` — measure FPS, inject `frame_rate`, wire new settings, live-apply.
- **Modify** `webapp/isitec_api/app.py` + `webapp/isitec_app/app.py` — `/api/settings` validate + live-apply new fields.
- **Modify** `webapp/isitec_api/templates/index.html` + `webapp/isitec_app/templates/index.html` — "Counting accuracy" UI block.
- **Modify** `webapp/isitec_api/static/js/main.js` + `webapp/isitec_app/static/js/main.js` — load/save new fields.
- **Modify** `isidet/configs/inference/common.yaml` (+ `cpu.yaml`/`gpu.yaml` comments) — defaults.
- **Modify** `webapp/isitec_api/settings.json` + `webapp/isitec_app/settings.json` — default `dedup_time_enabled: false`, new keys.
- **Create** `tools/count_eval.py` — labeled-truth miss-rate harness.

---

## Task 1: CrossingDetector pure module (frame-gap-tolerant crossing)

**Files:**
- Create: `isidet/src/shared/crossing.py`
- Test: `isidet/tests/test_crossing.py`

- [ ] **Step 1: Write the failing test**

Create `isidet/tests/test_crossing.py`:

```python
"""Plain-python tests (repo has no pytest). Run: python isidet/tests/test_crossing.py"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.shared.crossing import CrossingDetector


def test_counts_clean_crossing_left_to_right():
    d = CrossingDetector()
    # line at x=100, 'after' side is x>100 (left_to_right)
    assert d.update([1], [90.0], 100.0, after_is_greater=True) == set()   # before
    assert d.update([1], [110.0], 100.0, after_is_greater=True) == {1}    # crossed
    print("ok clean crossing")


def test_counts_once_only():
    d = CrossingDetector()
    d.update([1], [90.0], 100.0, True)
    assert d.update([1], [110.0], 100.0, True) == {1}
    assert d.update([1], [120.0], 100.0, True) == set()   # already fired
    print("ok counts once")


def test_recovers_when_flip_frame_is_dropped():
    # No frame ever lands exactly at the line; track jumps 80 -> (gap) -> 130.
    d = CrossingDetector()
    assert d.update([7], [80.0], 100.0, True) == set()
    assert d.update([7], [130.0], 100.0, True) == {7}     # still counted
    print("ok recovers dropped flip")


def test_respects_belt_direction_right_to_left():
    # 'after' side is x<100 (right_to_left); moving from 130 -> 70 should count.
    d = CrossingDetector()
    assert d.update([3], [130.0], 100.0, after_is_greater=False) == set()
    assert d.update([3], [70.0], 100.0, after_is_greater=False) == {3}
    print("ok right_to_left")


def test_no_count_for_wrong_direction():
    # left_to_right line, object only ever on the after side, then wanders further
    # after — never seen before the line -> never counts (avoids phantom counts of
    # objects that enter already past the line).
    d = CrossingDetector()
    assert d.update([5], [110.0], 100.0, True) == set()
    assert d.update([5], [150.0], 100.0, True) == set()
    print("ok no wrong-direction count")


def test_two_close_tracks_both_count():
    d = CrossingDetector()
    d.update([1, 2], [90.0, 85.0], 100.0, True)
    assert d.update([1, 2], [110.0, 105.0], 100.0, True) == {1, 2}
    print("ok two close tracks")


def test_forget_prunes_state():
    d = CrossingDetector()
    d.update([1], [90.0], 100.0, True)
    d.update([1], [110.0], 100.0, True)
    d.forget(keep_ids={2})           # 1 no longer active
    # id 1 reused later as a brand-new track -> may count again (new physical parcel)
    d.update([1], [90.0], 100.0, True)
    assert d.update([1], [110.0], 100.0, True) == {1}
    print("ok forget prunes")


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print("ALL CROSSING TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec deploy-web-1 python3 /opt/isitec/isidet/tests/test_crossing.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.shared.crossing'`

(Note: `isidet/tests/` and `isidet/src/shared/crossing.py` are bind-mounted? They are **not** — `isidet/src` is baked into the image. For the red/green loop during dev, run against the host checkout instead: `cd /home/aatanda/logistic-fps && PYTHONPATH=isidet python3 isidet/tests/test_crossing.py`.)

Run (host): `cd /home/aatanda/logistic-fps && PYTHONPATH=isidet python3 isidet/tests/test_crossing.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.shared.crossing'`

- [ ] **Step 3: Write minimal implementation**

Create `isidet/src/shared/crossing.py`:

```python
"""Frame-gap-tolerant line crossing — recall booster for low-FPS streams.

supervision's ``LineZone`` fires on the instantaneous frame-to-frame side flip of
an anchor. At low FPS a fast parcel can move more than its own size between frames;
if the exact flip frame is dropped (or the track has a 1-frame detection gap at the
line) ``LineZone`` can miss it. ``CrossingDetector`` instead *latches* each track
once its leading-edge anchor has been seen strictly *before* the line, then fires
once when that same track is later seen *after* the line in belt order — tolerant of
any number of skipped frames in between.

Pure logic: no cv2 / supervision import, so it is unit-testable in isolation. It is
OR-ed with ``LineZone`` in ``VisionEngine`` and feeds the SAME ``DedupGate`` /
``seq`` path, so a crossing caught by both is still counted exactly once.
"""


class CrossingDetector:
    def __init__(self):
        self._seen_before: set = set()   # track ids observed strictly before the line
        self._fired: set = set()         # track ids already reported crossed

    def update(self, track_ids, positions, line_coord: float,
               after_is_greater: bool) -> set:
        """Report track ids that newly crossed this frame.

        Args:
            track_ids:  iterable of int tracker ids present this frame.
            positions:  iterable of float — the leading-edge anchor's coordinate on
                        the crossing axis (x for a vertical line, y for horizontal),
                        aligned 1:1 with ``track_ids``.
            line_coord: the line's coordinate on that axis (pixels).
            after_is_greater: True when the 'after' (post-crossing) side is
                        ``coord > line_coord`` for the current belt direction,
                        False when it is ``coord < line_coord``.

        Returns:
            set of track ids that crossed for the first time on this call.
        """
        crossed = set()
        for tid, pos in zip(track_ids, positions):
            tid = int(tid)
            if tid in self._fired:
                continue
            if after_is_greater:
                before_side, after_side = pos < line_coord, pos > line_coord
            else:
                before_side, after_side = pos > line_coord, pos < line_coord
            if before_side:
                self._seen_before.add(tid)
            elif after_side and tid in self._seen_before:
                self._fired.add(tid)
                self._seen_before.discard(tid)
                crossed.add(tid)
        return crossed

    def forget(self, keep_ids) -> None:
        """Drop state for tracks no longer active so a reused id can count again."""
        keep = {int(i) for i in keep_ids}
        self._seen_before &= keep
        self._fired &= keep
```

- [ ] **Step 4: Run test to verify it passes**

Run (host): `cd /home/aatanda/logistic-fps && PYTHONPATH=isidet python3 isidet/tests/test_crossing.py`
Expected: prints each `ok ...` line then `ALL CROSSING TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
cd /home/aatanda/logistic-fps
git add isidet/src/shared/crossing.py isidet/tests/test_crossing.py
git commit -m "accuracy: add CrossingDetector (frame-gap-tolerant crossing latch)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Wire CrossingDetector into VisionEngine behind `count_interpolate`

**Files:**
- Modify: `isidet/src/shared/vision_engine.py` (init `~60-78`, `init_line` `~160-194`, `process_frame` `~233-275`)

- [ ] **Step 1: Import + construct detector and counting flags in `__init__`**

In `vision_engine.py`, at the top with the other shared imports add:

```python
from src.shared.crossing import CrossingDetector
```

In `__init__`, immediately after the `self.dedup = DedupGate(...)` block (around line 77), add:

```python
        # Recall booster — OR-ed with LineZone, shares the dedup/seq path.
        _inf = config.get('inference', {})
        self.count_interpolate = bool(_inf.get('count_interpolate', True))
        self.crossing = CrossingDetector()
```

- [ ] **Step 2: Store crossing geometry in `init_line`**

In `init_line`, just before the `self.line_zone = sv.LineZone(...)` call (around line 189), add:

```python
        # Geometry the CrossingDetector needs (axis + line coord + belt "after" side).
        self._trigger_anchor = anchor
        if orientation == 'horizontal':
            self._cross_axis = 1            # compare anchor y
            self._line_coord = float(line_y)
            self._after_is_greater = (self.belt_direction == 'top_to_bottom')
        else:
            self._cross_axis = 0            # compare anchor x
            self._line_coord = float(line_x)
            self._after_is_greater = (self.belt_direction == 'left_to_right')
```

- [ ] **Step 3: OR the detector into the crossing decision in `process_frame`**

Replace the two lines (around 250-251):

```python
        in_cross, out_cross = self.line_zone.trigger(detections=detections)
        all_crossings = in_cross | out_cross
```

with:

```python
        in_cross, out_cross = self.line_zone.trigger(detections=detections)
        all_crossings = in_cross | out_cross

        # Recall recovery: OR in the frame-gap-tolerant latch on the leading-edge
        # anchor. Same dedup/seq path below, so a crossing caught by both counts once.
        if self.count_interpolate and detections.tracker_id is not None and len(detections):
            anchors = detections.get_anchors_coordinates(self._trigger_anchor)
            ids = [int(t) for t in detections.tracker_id]
            coords = [float(a[self._cross_axis]) for a in anchors]
            recovered = self.crossing.update(ids, coords, self._line_coord,
                                              self._after_is_greater)
            if recovered:
                all_crossings = all_crossings | np.array(
                    [t in recovered for t in ids], dtype=bool)
            self.crossing.forget(keep_ids=ids)
```

- [ ] **Step 4: Add `configure_counting` for live apply**

After the existing `swap_inferencer` method (search `def swap_inferencer`), add a sibling method:

```python
    def configure_counting(self, count_interpolate=None):
        """Live-toggle the recall recovery without tearing down session state.

        Preserves counts/tracker/line/dedup. Only flips the interpolation flag;
        CrossingDetector latch state is preserved so in-flight tracks keep their
        before/after history across the toggle.
        """
        if count_interpolate is not None:
            self.count_interpolate = bool(count_interpolate)
```

- [ ] **Step 5: Verify it loads + behaves (runtime, no unit harness for the engine)**

Rebuild and run a stream, confirm no error and the interpolation path executes:

```bash
cd /home/aatanda/logistic-fps/deploy
WEB_BACKEND=fastapi docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build web
# wait healthy, then:
TOK=$(curl -s -X POST localhost:9501/api/dev-auth -H 'Content-Type: application/json' -d '{"password":"change-me"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -s -X POST localhost:9501/api/start -H 'Content-Type: application/json' -d '{"source":"/opt/isitec/webapp/isitec_api/uploads/testvid.mp4","model_type":"yolo"}'
sleep 8
curl -s localhost:9501/api/stats
docker logs deploy-web-1 2>&1 | tail -5   # no traceback
curl -s -X POST localhost:9501/api/stop >/dev/null
```

Expected: `is_running:true`, non-zero counts, no traceback mentioning `crossing`/`get_anchors_coordinates`.

- [ ] **Step 6: Commit**

```bash
cd /home/aatanda/logistic-fps
git add isidet/src/shared/vision_engine.py
git commit -m "accuracy: OR CrossingDetector into VisionEngine (count_interpolate)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: ByteTrack frame_rate calibration

**Files:**
- Modify: `isidet/src/shared/vision_engine.py` (tracker init `~64-68`)
- Modify: `webapp/isitec_api/stream_handler.py` + `webapp/isitec_app/stream_handler.py`

- [ ] **Step 1: Read `frame_rate` from config in the tracker init**

In `vision_engine.py`, replace the `self.tracker = sv.ByteTrack(...)` block (lines ~64-68) with:

```python
        _fr = track_cfg.get('frame_rate')
        _bt_kwargs = dict(
            track_activation_threshold=conf_thresh,
            lost_track_buffer=track_cfg.get('track_buffer', 60),
            minimum_matching_threshold=track_cfg.get('match_thresh', 0.9),
        )
        if _fr:                      # 0/None => let supervision default (30)
            _bt_kwargs['frame_rate'] = int(round(float(_fr)))
        self.tracker = sv.ByteTrack(**_bt_kwargs)
        self._tracker_kwargs = _bt_kwargs    # remembered for live rebuilds
```

- [ ] **Step 2: Measure source FPS and inject into `ve_config` in stream_handler (both backends)**

In **both** `stream_handler.py`, find where `ve_config = _deep_merge(self.config, self.inference_config or {})` is built (api `~1042`). Immediately after the dedup-injection block and before `self.engine = VisionEngine(...)`, add:

```python
                # Tracker FPS calibration — feed the real capture FPS to ByteTrack so
                # its Kalman predicts the true per-frame displacement (fast parcels keep
                # their id across the line). tracker_fps>0 overrides; auto uses the
                # capture's nominal FPS, clamped to a sane 1..120, else 20.
                _ui = self._load_settings() if hasattr(self, '_load_settings') else {}
                ve_config.setdefault('bytetrack', {})
                _manual = float(_ui.get('tracker_fps', 0) or 0)
                if _manual > 0:
                    ve_config['bytetrack']['frame_rate'] = _manual
                elif _ui.get('tracker_fps_auto', True):
                    _native = 0.0
                    try:
                        _native = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    except Exception:
                        _native = 0.0
                    ve_config['bytetrack']['frame_rate'] = _native if 1.0 <= _native <= 120.0 else 20.0
```

(If `self._load_settings()` is not the helper name in this file, use the same call the dedup block above already uses to read `_ui` — reuse that exact variable rather than re-reading.)

- [ ] **Step 3: Verify the tracker received the right frame_rate**

```bash
# after rebuild + start a stream as in Task 2 Step 5, then:
docker exec deploy-web-1 python3 -c "
from isitec_api.app import stream_handler as sh
e = sh.engine
print('frame_rate kwarg:', e._tracker_kwargs.get('frame_rate'))
print('tracker frame_rate:', getattr(e.tracker, 'frame_rate', '?'))
"
```

Expected: a value matching `testvid.mp4`'s FPS (e.g. `25`), not absent.

- [ ] **Step 3b: Expose the calibrated FPS in `get_stats()` (both backends)**

In **both** `stream_handler.py` `get_stats()`, add `frame_rate` to the returned dict (so the UI's read-only "Detected FPS" can show it). Use the value injected into `ve_config['bytetrack']['frame_rate']`, which you should also store on `self` at injection time as `self._tracker_frame_rate`:

```python
            "frame_rate": getattr(self, '_tracker_frame_rate', None),
```

And in the Task 3 Step 2 injection block, after setting `ve_config['bytetrack']['frame_rate'] = ...`, record it:

```python
                self._tracker_frame_rate = ve_config['bytetrack'].get('frame_rate')
```

- [ ] **Step 4: Commit**

```bash
cd /home/aatanda/logistic-fps
git add isidet/src/shared/vision_engine.py webapp/isitec_api/stream_handler.py webapp/isitec_app/stream_handler.py
git commit -m "accuracy: calibrate ByteTrack frame_rate from capture FPS

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Recall defaults — time-dedup OFF + expose track_buffer

**Files:**
- Modify: `isidet/configs/inference/common.yaml`
- Modify: `webapp/isitec_api/settings.json` + `webapp/isitec_app/settings.json`

- [ ] **Step 1: Set config defaults**

In `isidet/configs/inference/common.yaml`, under the `inference:` block, ensure these keys exist with these values (add if missing):

```yaml
inference:
  count_interpolate: true        # recall: frame-gap-tolerant crossing (Task 2)
  dedup_time_enabled: false      # recall: global time guard OFF (dropped distinct parcels)
  dedup_interval_ms: 300         # only used if dedup_time_enabled is turned back on
```

- [ ] **Step 2: Set settings.json defaults (both backends)**

In **both** `webapp/isitec_api/settings.json` and `webapp/isitec_app/settings.json`, set:

```json
  "dedup_time_enabled": false,
  "dedup_interval_ms": 300,
  "count_interpolate": true,
  "tracker_fps_auto": true,
  "tracker_fps": 0,
  "track_buffer": 60
```

(Keep existing keys; only change `dedup_time_enabled` to `false` and add the four new keys. `track_buffer` default stays 60 — see spec; it is exposed for on-site A/B, recommended experiment value 90.)

- [ ] **Step 3: Have the engine read `track_buffer` from settings**

In **both** `stream_handler.py`, in the same FPS-injection block from Task 3 Step 2, add:

```python
                _tb = int(_ui.get('track_buffer', 0) or 0)
                if _tb > 0:
                    ve_config['bytetrack']['track_buffer'] = _tb
```

- [ ] **Step 4: Verify defaults load**

```bash
docker exec deploy-web-1 python3 -c "
import json; s=json.load(open('/opt/isitec/webapp/isitec_api/settings.json'))
assert s['dedup_time_enabled'] is False, s['dedup_time_enabled']
assert s['count_interpolate'] is True
assert s['tracker_fps_auto'] is True and s['tracker_fps'] == 0
assert s['track_buffer'] == 60
print('defaults OK')
"
```

Expected: `defaults OK`

- [ ] **Step 5: Commit**

```bash
cd /home/aatanda/logistic-fps
git add isidet/configs/inference/common.yaml webapp/isitec_api/settings.json webapp/isitec_app/settings.json webapp/isitec_api/stream_handler.py webapp/isitec_app/stream_handler.py
git commit -m "accuracy: default time-dedup OFF, expose track_buffer/tracker_fps

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `/api/settings` validation + live apply (both backends)

**Files:**
- Modify: `webapp/isitec_api/app.py` (the `save_settings`/`/api/settings` handler)
- Modify: `webapp/isitec_app/app.py` (same handler)

- [ ] **Step 1: Validate + persist the new fields (FastAPI)**

In `webapp/isitec_api/app.py` `save_settings`, alongside the existing `dedup_*` validation, add:

```python
        if 'count_interpolate' in request_body:
            new_settings['count_interpolate'] = bool(request_body['count_interpolate'])
        if 'tracker_fps_auto' in request_body:
            new_settings['tracker_fps_auto'] = bool(request_body['tracker_fps_auto'])
        if 'tracker_fps' in request_body:
            _v = float(request_body['tracker_fps'])
            if not (0 <= _v <= 120):
                return JSONResponse({"status": "error", "message": "tracker_fps must be 0-120 (0=auto)"}, status_code=400)
            new_settings['tracker_fps'] = _v
        if 'track_buffer' in request_body:
            _v = int(request_body['track_buffer'])
            if not (1 <= _v <= 600):
                return JSONResponse({"status": "error", "message": "track_buffer must be 1-600"}, status_code=400)
            new_settings['track_buffer'] = _v
```

- [ ] **Step 2: Live-apply interpolation (FastAPI)**

Where the handler already live-applies dedup (`stream_handler.engine.dedup.configure(...)`), add right after it:

```python
        if stream_handler.engine is not None and 'count_interpolate' in request_body:
            stream_handler.engine.configure_counting(
                count_interpolate=bool(request_body['count_interpolate']))
```

(`tracker_fps`/`track_buffer` apply on the next stream start — they rebuild the tracker; note this in the UI help text rather than hot-rebuilding mid-stream, which would reset tracks.)

- [ ] **Step 3: Mirror Steps 1-2 in Flask**

Apply the identical validation in `webapp/isitec_app/app.py` using `jsonify({...}), 400` for the error returns (Flask style) instead of `JSONResponse`, and the same `engine.configure_counting(...)` live-apply.

- [ ] **Step 4: Verify validation + live apply**

```bash
TOK=$(curl -s -X POST localhost:9501/api/dev-auth -H 'Content-Type: application/json' -d '{"password":"change-me"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
# good
curl -s -X POST localhost:9501/api/settings -H "X-Dev-Token: $TOK" -H 'Content-Type: application/json' -d '{"count_interpolate":false,"tracker_fps":0,"track_buffer":90}'; echo
# bad
curl -s -X POST localhost:9501/api/settings -H "X-Dev-Token: $TOK" -H 'Content-Type: application/json' -d '{"track_buffer":9999}'; echo
```

Expected: first → `success`; second → error `track_buffer must be 1-600`.

- [ ] **Step 5: Commit**

```bash
cd /home/aatanda/logistic-fps
git add webapp/isitec_api/app.py webapp/isitec_app/app.py
git commit -m "accuracy: /api/settings validates + live-applies recall toggles

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: "Counting accuracy" Settings UI block (both backends)

**Files:**
- Modify: `webapp/isitec_api/templates/index.html` + `webapp/isitec_app/templates/index.html`
- Modify: `webapp/isitec_api/static/js/main.js` + `webapp/isitec_app/static/js/main.js`

- [ ] **Step 1: Add the UI block (both index.html)**

Immediately after the existing dedup settings group in the dev Settings section, add:

```html
<div class="settings-group" id="countingAccuracyGroup">
  <h4>Counting accuracy <small>(recall — catch every crossing)</small></h4>
  <label class="switch-row">
    <input type="checkbox" id="set_count_interpolate" checked>
    Recover crossings between frames (recommended ON for low FPS)
  </label>
  <label class="switch-row">
    <input type="checkbox" id="set_dedup_time_enabled">
    Time-dedup guard (OFF for recall — ON can drop close-together parcels)
  </label>
  <small>Tracker auto-calibrates to the camera frame rate. Detected FPS:
    <span id="detectedFps">—</span></small>
</div>
```

Note: `tracker_fps` (manual override) and `track_buffer` are intentionally NOT in the
UI — auto-calibration covers them. They remain settable via `settings.json` / `/api/settings`
as engineer escape hatches (Tasks 4-5 keep accepting them), just not operator-facing.

- [ ] **Step 2: Populate the fields on settings load (both main.js)**

In the function that fills the Settings form from `serverSettings` (search where `dedup_time_enabled` is read into the form), add:

```javascript
        document.getElementById('set_count_interpolate').checked = serverSettings.count_interpolate !== false;
        document.getElementById('set_dedup_time_enabled').checked = serverSettings.dedup_time_enabled === true;
```

- [ ] **Step 3: Include the fields in the settings save payload (both main.js)**

In the object POSTed to `/api/settings` (search where `dedup_interval_ms` is added to the payload), add:

```javascript
            count_interpolate: document.getElementById('set_count_interpolate').checked,
            dedup_time_enabled: document.getElementById('set_dedup_time_enabled').checked,
```

- [ ] **Step 3b: Show the detected FPS (read-only) from the stats poll (both main.js)**

In the stats handler (Flask poll of `/api/stats` / FastAPI `/ws/stats` onmessage), where other live fields are written to the DOM, add:

```javascript
        const _fps = (stats && stats.frame_rate) ? Math.round(stats.frame_rate) : null;
        const _el = document.getElementById('detectedFps');
        if (_el) _el.textContent = _fps ? _fps + ' fps' : '—';
```

- [ ] **Step 4: Syntax check + visual verify**

Run: `node --check webapp/isitec_api/static/js/main.js && node --check webapp/isitec_app/static/js/main.js`
Expected: no output (both parse).

Then rebuild and confirm in the browser: Settings → "Counting accuracy" block shows, toggles persist across reload (save, refresh, re-open Settings).

- [ ] **Step 5: Commit**

```bash
cd /home/aatanda/logistic-fps
git add webapp/isitec_api/templates/index.html webapp/isitec_app/templates/index.html webapp/isitec_api/static/js/main.js webapp/isitec_app/static/js/main.js
git commit -m "accuracy: Counting accuracy Settings UI block (both backends)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Eval harness — labeled-truth miss rate

**Files:**
- Create: `tools/count_eval.py`

- [ ] **Step 1: Write the harness**

Create `tools/count_eval.py`:

```python
"""Count-accuracy harness: run a clip through /api/start, compare the resulting
per-class counts to a hand-labeled truth, report miss rate (under-count).

Usage:
  python tools/count_eval.py --base http://localhost:9501 --password change-me \\
      --source /opt/isitec/webapp/isitec_api/uploads/testvid.mp4 \\
      --truth '{"carton": 20, "polybag": 7}' --seconds 90
"""
import argparse, json, time, urllib.request


def _post(base, path, body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method='POST')
    req.add_header('Content-Type', 'application/json')
    if token:
        req.add_header('X-Dev-Token', token)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read() or b'{}')


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=15) as r:
        return json.loads(r.read() or b'{}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://localhost:9501')
    ap.add_argument('--password', default='change-me')
    ap.add_argument('--source', required=True)
    ap.add_argument('--truth', required=True, help='JSON dict of true counts per class')
    ap.add_argument('--seconds', type=int, default=90)
    ap.add_argument('--model', default='yolo')
    args = ap.parse_args()

    truth = json.loads(args.truth)
    token = _post(args.base, '/api/dev-auth', {'password': args.password}).get('token')
    _post(args.base, '/api/start', {'source': args.source, 'model_type': args.model}, token)
    try:
        deadline = time.monotonic() + args.seconds
        last = {}
        while time.monotonic() < deadline:
            st = _get(args.base, '/api/stats')
            last = st.get('counts', {})
            if not st.get('is_running', True):
                break
            time.sleep(2)
    finally:
        _post(args.base, '/api/stop', {}, token)

    print('class      truth  counted  missed  miss%')
    total_t = total_c = 0
    for cls, t in truth.items():
        c = int(last.get(cls, 0)); m = t - c
        total_t += t; total_c += c
        print(f'{cls:<10} {t:>5} {c:>8} {m:>7} {(100*m/t if t else 0):>6.1f}')
    miss = total_t - total_c
    print(f'{"TOTAL":<10} {total_t:>5} {total_c:>8} {miss:>7} {(100*miss/total_t if total_t else 0):>6.1f}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify it runs end-to-end**

Run: `cd /home/aatanda/logistic-fps && python3 tools/count_eval.py --source /opt/isitec/webapp/isitec_api/uploads/testvid.mp4 --truth '{"carton":1,"polybag":3}' --seconds 30`
Expected: prints the table with a TOTAL row (numbers are illustrative; the harness working is the deliverable).

- [ ] **Step 3: Commit**

```bash
cd /home/aatanda/logistic-fps
git add tools/count_eval.py
git commit -m "accuracy: count_eval harness (labeled-truth miss rate)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Integration verification + recall A/B

**Files:** none (verification only)

- [ ] **Step 1: Rebuild FastAPI image with all changes**

```bash
cd /home/aatanda/logistic-fps/deploy
WEB_BACKEND=fastapi docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build web
for i in $(seq 1 40); do [ "$(docker inspect -f '{{.State.Health.Status}}' deploy-web-1 2>/dev/null)" = healthy ] && break; sleep 3; done
```

- [ ] **Step 2: A/B — interpolation OFF (baseline) vs ON, same clip**

```bash
TOK=$(curl -s -X POST localhost:9501/api/dev-auth -H 'Content-Type: application/json' -d '{"password":"change-me"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
SRC=/opt/isitec/webapp/isitec_api/uploads/testvid.mp4
curl -s -X POST localhost:9501/api/settings -H "X-Dev-Token: $TOK" -H 'Content-Type: application/json' -d '{"count_interpolate":false,"dedup_time_enabled":true}' >/dev/null
python3 tools/count_eval.py --source "$SRC" --truth '{"carton":1,"polybag":3}' --seconds 40   # baseline
curl -s -X POST localhost:9501/api/settings -H "X-Dev-Token: $TOK" -H 'Content-Type: application/json' -d '{"count_interpolate":true,"dedup_time_enabled":false}' >/dev/null
python3 tools/count_eval.py --source "$SRC" --truth '{"carton":1,"polybag":3}' --seconds 40   # recall config
```

Expected: recall-config TOTAL counted ≥ baseline counted (never lower). Record both.

- [ ] **Step 3: Invariant — seq gap-free, one datagram per count**

```bash
# with a stream running under the recall config, capture UDP on the host loopback
docker exec deploy-web-1 python3 -c "
import socket, json
s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(('127.0.0.1',9502)); s.settimeout(15)
seqs=[]
try:
    while len(seqs)<10:
        d,_=s.recvfrom(1024); seqs.append(json.loads(d)['seq'])
except Exception: pass
print('seqs:', seqs)
print('gap-free:', seqs==list(range(seqs[0], seqs[0]+len(seqs))) if seqs else 'n/a')
" &
# (point UDP at 127.0.0.1:9502 via /api/udp first, then start a stream)
```

Expected: `seqs` strictly increment by 1 (gap-free); one datagram per counted crossing.

- [ ] **Step 4: Confirm leading-edge anchor unchanged**

Run: `docker logs deploy-web-1 2>&1 | grep "\[LINE\]" | tail -1`
Expected: `... (leading_edge) ...` — anchor unchanged by any of this work.

- [ ] **Step 5: Push the branch**

```bash
cd /home/aatanda/logistic-fps
git push origin fps
```

---

## Self-review notes

- **Spec coverage:** tracker FPS calibration → Task 3; predicted-path crossing → Tasks 1-2; time-dedup OFF default → Task 4; UI block → Task 6; live-apply → Task 5; seq/leading-edge/timing invariants → Task 8; eval harness → Task 7; dropped filters → not built (correct).
- **UI simplification (refinement after review):** `tracker_fps` and `track_buffer` are NOT exposed in the operator UI — FPS auto-calibration makes them redundant for the normal case. They remain settable via `settings.json`/`/api/settings`/YAML as engineer escape hatches (Tasks 4-5 still accept + default them; `track_buffer` default stays 60). The UI shows only `count_interpolate`, `dedup_time_enabled`, and a read-only detected-FPS readout. conf (`yolo_conf`/`detr_conf`) is already in the existing Settings form — unchanged.
- **Type consistency:** `count_interpolate`, `tracker_fps_auto`, `tracker_fps`, `track_buffer`, `dedup_time_enabled` use identical names across settings.json, `/api/settings`, the UI ids (`set_*`), and `ve_config`. `CrossingDetector.update(track_ids, positions, line_coord, after_is_greater)` / `.forget(keep_ids)` signatures match between Task 1 (def) and Task 2 (call). `configure_counting(count_interpolate=...)` matches between Task 2 (def) and Task 5 (call).
- **State across hot-swap:** `self.crossing` (CrossingDetector) is created in `__init__` and not rebuilt by `swap_inferencer`, so its latch survives a model swap like `counted_ids` does.

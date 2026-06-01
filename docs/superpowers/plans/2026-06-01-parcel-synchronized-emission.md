# Parcel-Synchronized UDP Emission — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit exactly one UDP datagram per physical parcel, at the parcel's *center* crossing of a fixed line, so the Celio PLC's timing-based correlation binds it to the right parcel — with an operator-toggleable time-dedup guard on top of the always-on track-ID dedup.

**Architecture:** Change the line-crossing anchor to `CENTER` (new default; PLC owns the gate, so leading-edge is obsolete). Extract dedup into a small testable `DedupGate` (track-ID base always on + optional time guard). Wire `dedup_time_enabled` / `dedup_interval_ms` from `settings.json` through both web backends into `VisionEngine`. `seq` is unaffected (it is stamped post-dedup at `publish()`).

**Tech Stack:** Python, Ultralytics/supervision (`sv.LineZone`, ByteTrack), Flask + FastAPI web backends (kept in parity), OpenVINO/ONNX inference, Docker compose. No pytest in this repo — verification uses focused `python3` harnesses and the live Docker stack (matching project practice).

**Spec:** `docs/superpowers/specs/2026-06-01-parcel-synchronized-emission-design.md`

---

## File Structure

- **Create** `isidet/src/shared/dedup_gate.py` — the dedup decision unit (track-ID + time guard). One responsibility, no supervision/torch deps → unit-testable in isolation.
- **Modify** `isidet/src/shared/vision_engine.py` — use `DedupGate`; default anchor `CENTER`; read dedup/anchor config; log time-suppressions.
- **Modify** `isidet/configs/inference/common.yaml` — `inference.trigger_anchor: center`; ByteTrack stabilization.
- **Modify** `webapp/isitec_app/settings.json` + `webapp/isitec_api/settings.json` — add `dedup_time_enabled`, `dedup_interval_ms`.
- **Modify** `webapp/isitec_app/stream_handler.py` + `webapp/isitec_api/stream_handler.py` — read the two settings, inject into `ve_config['inference']`.
- **Modify** `webapp/isitec_app/app.py` + `webapp/isitec_api/app.py` — validate the two settings in `/api/settings`.
- **Modify** `webapp/isitec_app/templates/index.html` + `webapp/isitec_api/templates/index.html` — Settings form fields.
- **Modify** `webapp/isitec_app/static/js/main.js` + `webapp/isitec_api/static/js/main.js` — load/save the fields.
- **Modify** `CLAUDE.md` — update Trigger-semantics section (center default).

> Web edits are **identical** across `isitec_app` (Flask) and `isitec_api` (FastAPI) unless noted; apply to both.

---

## Task 1: `DedupGate` unit

**Files:**
- Create: `isidet/src/shared/dedup_gate.py`

- [ ] **Step 1: Write the failing harness**

Create `isidet/src/shared/dedup_gate.py` empty for now, then create a scratch harness and run it (it should fail to import a real class):

```python
# run: cd <repo> && PYTHONPATH=isidet python3 - <<'PY'
from src.shared.dedup_gate import DedupGate

# track-ID base: same id emits once, ever
g = DedupGate(time_enabled=False)
assert g.should_emit(5, 0.0) is True
g.record(5, 0.0)
assert g.should_emit(5, 9999.0) is False, "same id must never re-emit"

# two different ids both emit when time guard off
assert g.should_emit(6, 0.0) is True

# time guard: new id within interval is suppressed, beyond interval emits
g2 = DedupGate(time_enabled=True, interval_ms=300)
assert g2.should_emit(1, 1000.0) is True
g2.record(1, 1000.0)
assert g2.should_emit(2, 1200.0) is False, "new id 200ms later must be time-suppressed"
assert g2.should_emit(2, 1300.0) is True,  "new id 300ms later must pass"

# already-counted id is suppressed regardless of time
g2.record(2, 1300.0)
assert g2.should_emit(2, 5000.0) is False

# configure() updates toggle/interval without losing counted_ids
g2.configure(time_enabled=False, interval_ms=50)
assert g2.should_emit(1, 1301.0) is False, "id 1 still counted after reconfigure"
assert g2.should_emit(7, 1301.0) is True

print("ALL_ASSERTIONS_PASSED")
PY
```

Run it. Expected: `ModuleNotFoundError`/`ImportError` (class not defined yet).

- [ ] **Step 2: Implement `DedupGate`**

Write `isidet/src/shared/dedup_gate.py`:

```python
"""Dedup decision for line-crossing emission — one datagram per physical parcel.

Two layers (see docs/superpowers/specs/2026-06-01-parcel-synchronized-emission-design.md):
  - track-ID base (always on): one emission per ByteTrack track id, for its lifetime.
  - time guard (operator toggle): suppress a crossing within `interval_ms` of the LAST
    emitted datagram, to absorb ID-churn that gives one parcel a fresh id.

Emit rule (AND): emit iff (id not yet counted) AND
                          (time guard off OR elapsed >= interval_ms since last emit).
"""

_PRUNE_AT = 50_000


class DedupGate:
    def __init__(self, time_enabled: bool = True, interval_ms: int = 300):
        self.time_enabled = bool(time_enabled)
        self.interval_ms = int(interval_ms)
        self.counted_ids: set = set()
        self._last_emit_ms = None

    def should_emit(self, track_id: int, now_ms: float) -> bool:
        if track_id in self.counted_ids:
            return False
        if (self.time_enabled and self._last_emit_ms is not None
                and (now_ms - self._last_emit_ms) < self.interval_ms):
            return False
        return True

    def time_suppressed(self, track_id: int, now_ms: float) -> bool:
        """True when a NEW id is blocked solely by the time guard (for logging)."""
        return track_id not in self.counted_ids and not self.should_emit(track_id, now_ms)

    def record(self, track_id: int, now_ms: float) -> None:
        self.counted_ids.add(track_id)
        self._last_emit_ms = now_ms
        if len(self.counted_ids) > _PRUNE_AT:
            keep = sorted(self.counted_ids)[len(self.counted_ids) // 2:]
            self.counted_ids = set(keep)

    def configure(self, time_enabled: bool, interval_ms: int) -> None:
        """Live-update toggle/interval, preserving counted_ids state."""
        self.time_enabled = bool(time_enabled)
        self.interval_ms = int(interval_ms)
```

- [ ] **Step 3: Run the harness to verify it passes**

Re-run the Step 1 harness. Expected: `ALL_ASSERTIONS_PASSED`.

- [ ] **Step 4: Commit**

```bash
git add isidet/src/shared/dedup_gate.py
git commit -m "feat(dedup): add DedupGate (track-ID base + optional time guard)"
```

---

## Task 2: Integrate `DedupGate` into `VisionEngine`

**Files:**
- Modify: `isidet/src/shared/vision_engine.py` (import; `__init__` ~line 70-82; crossing loop ~line 234-252)

- [ ] **Step 1: Import and construct the gate from config**

Add to imports (after line 8 `from src.utils.event_logger import EventLogger`):

```python
from src.shared.dedup_gate import DedupGate
```

In `__init__`, replace `self.counted_ids = set()` (line 71) with the gate, reading config (the `inf_cfg` local already exists later at the logging block — compute it here or reuse `config.get('inference', {})`):

```python
        # Dedup: track-ID base (always on) + optional time guard (operator toggle).
        _inf = config.get('inference', {})
        self.dedup = DedupGate(
            time_enabled=bool(_inf.get('dedup_time_enabled', True)),
            interval_ms=int(_inf.get('dedup_interval_ms', 300)),
        )
```

- [ ] **Step 2: Rewrite the crossing-emit loop to use the gate**

Replace lines 234-252 (the `new_events` loop **and** the prune block) with:

```python
        new_events = []
        now_ms = time.monotonic() * 1000.0   # one clock per frame; within-frame ties share it
        for i, crossed in enumerate(all_crossings):
            if crossed and detections.tracker_id is not None:
                t_id = int(detections.tracker_id[i])
                if self.dedup.should_emit(t_id, now_ms):
                    class_id = int(detections.class_id[i])
                    name = self.inferencer.class_names.get(class_id, "object")
                    class_totals[name] = class_totals.get(name, 0) + 1
                    self.dedup.record(t_id, now_ms)
                    new_events.append({"class": name, "id": t_id})
                    self.event_logger.log(name, t_id)
                elif self.dedup.time_suppressed(t_id, now_ms):
                    logger.info(f"⏱️ dedup-suppressed crossing id={t_id} "
                                f"(<{self.dedup.interval_ms}ms since last emit)")
```

(The prune logic now lives in `DedupGate.record`, so the old `if len(self.counted_ids) > 50_000:` block is removed.)

- [ ] **Step 3: Find any remaining `counted_ids` references and repoint them**

Run:

```bash
cd <repo> && grep -n "counted_ids" isidet/src/shared/vision_engine.py
```

Expected after edits: references only via `self.dedup.counted_ids` (e.g. in `swap_inferencer`'s preserved-state docstring/logging, if any). Repoint any leftover `self.counted_ids` to `self.dedup.counted_ids`. **Do not reset `self.dedup` in `swap_inferencer`** — it must persist across hot-swap (preserves counts + dedup state, as the old `counted_ids` did).

- [ ] **Step 4: Verify import + construction (harness)**

```bash
cd <repo> && PYTHONPATH=isidet python3 - <<'PY'
import ast, sys
src = open("isidet/src/shared/vision_engine.py").read()
ast.parse(src)                      # syntax OK
assert "self.dedup = DedupGate(" in src
assert "self.dedup.should_emit(" in src
assert "self.dedup.record(" in src
assert "if len(self.counted_ids) > 50_000" not in src, "old prune block must be gone"
print("VISION_ENGINE_OK")
PY
```

Expected: `VISION_ENGINE_OK`.

- [ ] **Step 5: Commit**

```bash
git add isidet/src/shared/vision_engine.py
git commit -m "refactor(vision): use DedupGate for one-per-parcel emission"
```

---

## Task 3: Center anchor as the new default

**Files:**
- Modify: `isidet/src/shared/vision_engine.py` (anchor selection ~line 169-173)
- Modify: `isidet/configs/inference/common.yaml` (`inference.trigger_anchor`)

- [ ] **Step 1: Make the anchor config-driven, defaulting to CENTER**

Replace lines 169-173 (the `# Pick the leading-edge anchor ...` block) with:

```python
        # Trigger anchor. Default CENTER: the parcel emits from the SAME belt
        # position regardless of size, which the Celio PLC's timing correlation
        # requires. Leading-edge (size-dependent) is kept only as an opt-in.
        _anchor_name = config.get('inference', {}).get('trigger_anchor', 'center')
        if _anchor_name == 'leading_edge':
            anchor = self._ANCHOR_MAP.get(
                (orientation, self.belt_direction),
                sv.Position.BOTTOM_CENTER,
            )
        else:
            anchor = sv.Position.CENTER
```

> `config` must be reachable here. If this code is inside a method without `config` in scope, store it in `__init__` as `self.config = config` (it likely already is — confirm with `grep -n "self.config" isidet/src/shared/vision_engine.py`; if present use `self.config`).

- [ ] **Step 2: Add the config default**

In `isidet/configs/inference/common.yaml`, under the top-level `inference:` block, add:

```yaml
  # Line-crossing trigger anchor. 'center' = emit when the parcel CENTER crosses
  # the line (fixed position, size-independent) — required by the PLC timing
  # correlation. 'leading_edge' = legacy size-dependent anchor (opt-in only).
  trigger_anchor: center
```

- [ ] **Step 3: Verify**

```bash
cd <repo> && PYTHONPATH=isidet python3 - <<'PY'
import yaml, ast
ast.parse(open("isidet/src/shared/vision_engine.py").read())
c = yaml.safe_load(open("isidet/configs/inference/common.yaml"))
assert c["inference"]["trigger_anchor"] == "center"
src = open("isidet/src/shared/vision_engine.py").read()
assert "sv.Position.CENTER" in src and "trigger_anchor" in src
print("ANCHOR_OK")
PY
```

Expected: `ANCHOR_OK`.

- [ ] **Step 4: Commit**

```bash
git add isidet/src/shared/vision_engine.py isidet/configs/inference/common.yaml
git commit -m "feat(vision): default center anchor (configurable), leading-edge opt-in"
```

---

## Task 4: ByteTrack stabilization (reduce ID churn)

**Files:**
- Modify: `isidet/configs/inference/common.yaml` (`inference.bytetrack.track_buffer`)
- Modify: `isidet/configs/inference/cpu.yaml` and `gpu.yaml` (`bytetrack.match_thresh`)

- [ ] **Step 1: Inspect current values**

```bash
cd <repo> && grep -n -A4 "bytetrack" isidet/configs/inference/common.yaml isidet/configs/inference/cpu.yaml isidet/configs/inference/gpu.yaml
```

- [ ] **Step 2: Raise track_buffer and loosen match_thresh**

In `common.yaml` `inference.bytetrack`: set `track_buffer: 90` (was 60 — keeps a track alive ~3 s longer through brief misses, so a re-appearing parcel reuses its id instead of getting a new one).

In `cpu.yaml` and `gpu.yaml` `bytetrack`: set `match_thresh: 0.8` (looser than the prior 0.9 — re-attaches a reappearing detection to its existing track more readily). Keep each file's existing other keys.

> These are **placeholder defaults** to be re-tuned against real belt footage (Open item in spec). The goal is fewer fresh IDs per physical parcel.

- [ ] **Step 3: Verify configs parse and carry the values**

```bash
cd <repo> && PYTHONPATH=isidet python3 - <<'PY'
import yaml
common = yaml.safe_load(open("isidet/configs/inference/common.yaml"))
cpu = yaml.safe_load(open("isidet/configs/inference/cpu.yaml"))
assert common["inference"]["bytetrack"]["track_buffer"] == 90
assert cpu["bytetrack"]["match_thresh"] == 0.8
print("BYTETRACK_OK")
PY
```

Expected: `BYTETRACK_OK`.

- [ ] **Step 4: Commit**

```bash
git add isidet/configs/inference/common.yaml isidet/configs/inference/cpu.yaml isidet/configs/inference/gpu.yaml
git commit -m "tune(bytetrack): longer track_buffer + looser match_thresh to cut ID churn"
```

---

## Task 5: Operator settings keys (both backends)

**Files:**
- Modify: `webapp/isitec_app/settings.json`, `webapp/isitec_api/settings.json`

- [ ] **Step 1: Add the two keys to both files**

In **each** `settings.json`, add (after the existing `udp_port` line, keeping valid JSON):

```json
  "dedup_time_enabled": true,
  "dedup_interval_ms": 300,
```

- [ ] **Step 2: Verify both parse**

```bash
cd <repo> && python3 - <<'PY'
import json
for f in ["webapp/isitec_app/settings.json","webapp/isitec_api/settings.json"]:
    d = json.load(open(f))
    assert d["dedup_time_enabled"] is True and d["dedup_interval_ms"] == 300, f
print("SETTINGS_JSON_OK")
PY
```

Expected: `SETTINGS_JSON_OK`.

- [ ] **Step 3: Commit**

```bash
git add webapp/isitec_app/settings.json webapp/isitec_api/settings.json
git commit -m "feat(settings): add dedup_time_enabled + dedup_interval_ms defaults"
```

---

## Task 6: Wire settings → `VisionEngine` (both backends)

**Files:**
- Modify: `webapp/isitec_app/stream_handler.py`, `webapp/isitec_api/stream_handler.py` (the `ve_config = _deep_merge(...)` site)

- [ ] **Step 1: Read the two settings and inject into `ve_config['inference']`**

Find the merge line (Flask ~1008, FastAPI ~1037):

```python
                ve_config = _deep_merge(self.config, self.inference_config or {})
```

Immediately **after** it, add (reads the operator settings.json, falling back to engine defaults):

```python
                # Operator dedup controls (settings.json) override the inference-config defaults.
                try:
                    _sp = Path(__file__).parent / 'settings.json'
                    _ui = json.load(open(_sp)) if _sp.exists() else {}
                except Exception:
                    _ui = {}
                ve_config.setdefault('inference', {})
                if isinstance(_ui.get('dedup_time_enabled'), bool):
                    ve_config['inference']['dedup_time_enabled'] = _ui['dedup_time_enabled']
                if isinstance(_ui.get('dedup_interval_ms'), int) and 0 <= _ui['dedup_interval_ms'] <= 60000:
                    ve_config['inference']['dedup_interval_ms'] = _ui['dedup_interval_ms']
```

> `Path` and `json` are already imported in `stream_handler.py` (used by the existing settings.json read). Confirm with `grep -n "^import json\|from pathlib import Path\|import json" webapp/isitec_app/stream_handler.py`.

- [ ] **Step 2: Verify both backends still import and carry the wiring**

```bash
cd <repo> && PYTHONPATH=isidet:webapp python3 - <<'PY'
import ast
for f in ["webapp/isitec_app/stream_handler.py","webapp/isitec_api/stream_handler.py"]:
    s = open(f).read(); ast.parse(s)
    assert "dedup_time_enabled" in s and "dedup_interval_ms" in s and "ve_config['inference']" in s, f
print("WIRING_OK")
PY
```

Expected: `WIRING_OK`.

- [ ] **Step 3: Commit**

```bash
git add webapp/isitec_app/stream_handler.py webapp/isitec_api/stream_handler.py
git commit -m "feat(stream): pass operator dedup settings into VisionEngine config"
```

---

## Task 7: `/api/settings` validation (both backends)

**Files:**
- Modify: `webapp/isitec_app/app.py`, `webapp/isitec_api/app.py`

- [ ] **Step 1: Locate the settings handler + allowed-keys list**

```bash
cd <repo> && grep -n "udp_port\|allowed\|valid.*key\|def update_settings\|def save_settings\|request_body\[\|data\[" webapp/isitec_app/app.py | head
```

Note the allow-list (the tuple that includes `'rtsp_url', 'udp_host', 'udp_port', 'auto_start'`) and the per-field validation block that mirrors `udp_port`.

- [ ] **Step 2: Add the two keys to the allow-list and validate them**

Add `'dedup_time_enabled', 'dedup_interval_ms'` to that allow-list tuple (both backends).

Then, next to the existing `udp_port` validation, add (Flask uses `data`, FastAPI uses `request_body` — match the local variable name in that file):

```python
    if 'dedup_time_enabled' in data:
        if not isinstance(data['dedup_time_enabled'], bool):
            return jsonify({"status": "error", "message": "dedup_time_enabled must be a boolean"}), 400
    if 'dedup_interval_ms' in data:
        try:
            n = int(data['dedup_interval_ms'])
            if not (0 <= n <= 60000):
                raise ValueError
            data['dedup_interval_ms'] = n
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "dedup_interval_ms must be 0-60000"}), 400
```

> FastAPI: use `request_body` instead of `data`, and its existing error-return shape (mirror that file's `udp_port` block exactly).

- [ ] **Step 3: Apply the live update so the toggle takes effect without a stream restart**

Next to the existing `update_target(...)` call (the `if 'udp_host' in data or 'udp_port' in data:` block), add a sibling that reconfigures the running gate:

```python
    if 'dedup_time_enabled' in data or 'dedup_interval_ms' in data:
        try:
            stream_handler.engine.dedup.configure(
                current.get('dedup_time_enabled', True),
                int(current.get('dedup_interval_ms', 300)),
            )
        except Exception:
            pass  # engine may not be running yet — settings still saved & used on next start
```

> Use whatever local holds the merged/saved settings in that handler (named `current` in the udp block). FastAPI: same logic, its variable names.

- [ ] **Step 4: Verify both parse and reference the keys**

```bash
cd <repo> && python3 - <<'PY'
import ast
for f in ["webapp/isitec_app/app.py","webapp/isitec_api/app.py"]:
    s = open(f).read(); ast.parse(s)
    assert "dedup_time_enabled" in s and "dedup_interval_ms" in s and ".dedup.configure(" in s, f
print("API_OK")
PY
```

Expected: `API_OK`.

- [ ] **Step 5: Commit**

```bash
git add webapp/isitec_app/app.py webapp/isitec_api/app.py
git commit -m "feat(api): validate + live-apply dedup settings in /api/settings"
```

---

## Task 8: Settings UI (both backends)

**Files:**
- Modify: `webapp/isitec_app/templates/index.html`, `webapp/isitec_api/templates/index.html`
- Modify: `webapp/isitec_app/static/js/main.js`, `webapp/isitec_api/static/js/main.js`

- [ ] **Step 1: Add form fields after the UDP port input**

In each `index.html`, right after the `set_udp_port` input block, add:

```html
                                <label title="Suppress duplicate detections of the same parcel within the interval below (track-ID dedup is always on).">Time dedup</label>
                                <input type="checkbox" id="set_dedup_time" checked>
                                <label data-i18n="set_dedup_interval_label">Dedup interval (ms)</label>
                                <input type="number" id="set_dedup_interval" class="set-select" min="0" max="60000" value="300">
```

- [ ] **Step 2: Load the values on settings fetch**

In each `main.js`, find the block that sets `udpHostEl.value = serverSettings.udp_host ?? '10.0.0.1'` and add after it:

```javascript
        const dedupTimeEl = document.getElementById('set_dedup_time');
        if (dedupTimeEl) dedupTimeEl.checked = serverSettings.dedup_time_enabled ?? true;
        const dedupIntEl = document.getElementById('set_dedup_interval');
        if (dedupIntEl) dedupIntEl.value = serverSettings.dedup_interval_ms ?? 300;
```

- [ ] **Step 3: Include the values on settings save**

In each `main.js`, find the save payload object that contains `udp_host:` / `udp_port:` and add:

```javascript
                dedup_time_enabled: document.getElementById('set_dedup_time').checked,
                dedup_interval_ms:  parseInt(document.getElementById('set_dedup_interval').value),
```

- [ ] **Step 4: Verify references exist in all four files**

```bash
cd <repo> && for f in webapp/isitec_app/templates/index.html webapp/isitec_api/templates/index.html webapp/isitec_app/static/js/main.js webapp/isitec_api/static/js/main.js; do
  grep -q "set_dedup_time" "$f" && grep -q "dedup_interval" "$f" && echo "OK $f" || echo "MISSING $f"
done
```

Expected: `OK` for all four.

- [ ] **Step 5: Commit**

```bash
git add webapp/isitec_app/templates/index.html webapp/isitec_api/templates/index.html webapp/isitec_app/static/js/main.js webapp/isitec_api/static/js/main.js
git commit -m "feat(ui): Settings controls for time-dedup toggle + interval"
```

---

## Task 9: Update CLAUDE.md trigger semantics

**Files:**
- Modify: `CLAUDE.md` (the "Trigger semantics (sorter-first)" section)

- [ ] **Step 1: Replace the leading-edge framing with the center-anchor default**

Find the "## Trigger semantics" section and add a leading note (keep the `_ANCHOR_MAP` table as the documented `leading_edge` opt-in):

```markdown
**Default anchor is now `center`** (config `inference.trigger_anchor`): the datagram fires when a
parcel's **center** crosses the line, giving a fixed, size-independent emit position — required by
the Celio PLC's timing-based correlation (the PLC, not IsiDetector, drives the sort gate). The
leading-edge map below is the legacy `trigger_anchor: leading_edge` opt-in.

Dedup: track-ID (one emission per ByteTrack id) is always on; a time guard
(`dedup_time_enabled`, default on; `dedup_interval_ms`, default 300) suppresses churned re-emissions
of the same parcel. `seq` is stamped post-dedup, so it stays a gap-free delivery counter.
```

- [ ] **Step 2: Verify**

```bash
cd <repo> && grep -q "Default anchor is now .center." CLAUDE.md && echo "DOCS_OK"
```

Expected: `DOCS_OK`.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: trigger semantics — center anchor default + dedup controls"
```

---

## Task 10: End-to-end verification on the live stack

**Files:** none (verification only)

- [ ] **Step 1: Rebuild + restart the web container (CPU profile)**

```bash
cd <repo>/deploy && docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build web
# wait for healthy:
for i in $(seq 1 20); do [ "$(docker inspect -f '{{.State.Health.Status}}' deploy-web-1 2>/dev/null)" = healthy ] && break; sleep 4; done
docker exec deploy-web-1 sh -c 'grep -c "self.dedup" isidet/src/shared/vision_engine.py'   # -> >=1
```

- [ ] **Step 2: Plant a loopback UDP listener, auth, point UDP at it, start the test video**

```bash
docker exec deploy-web-1 sh -c 'cat > /tmp/lst.py <<EOF
import socket
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(("127.0.0.1",9502))
open("/tmp/rx.log","w").close()
f=open("/tmp/rx.log","a")
while True:
    d,_=s.recvfrom(2048); f.write(d.decode()+"\n"); f.flush()
EOF'
docker exec -d deploy-web-1 python3 /tmp/lst.py
T=$(curl -s -m5 -X POST http://127.0.0.1:9501/api/dev-auth -H 'Content-Type: application/json' -d '{"password":"change-me"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
curl -s -m5 -X POST http://127.0.0.1:9501/api/udp -H "X-Dev-Token: $T" -H 'Content-Type: application/json' -d '{"host":"127.0.0.1","port":9502}'
# ensure testvid is staged in the uploads mount (see session notes), then:
curl -s -m15 -X POST http://127.0.0.1:9501/api/start -H "X-Dev-Token: $T" -H 'Content-Type: application/json' \
  -d '{"source":"/opt/isitec/webapp/isitec_app/uploads/testvid.mp4","model_type":"yolo","weights":"isidet/runs/segment/models/yolo/yolo26n_320_200/weights/openvino/model.xml"}'
```

- [ ] **Step 3: After ~40 s, confirm one-per-parcel + gap-free seq + persisted log**

```bash
docker exec deploy-web-1 sh -c 'wc -l /tmp/rx.log; tail -3 /tmp/rx.log'
docker exec deploy-web-1 sh -c 'python3 - <<PY
import json
seqs=[json.loads(l)["seq"] for l in open("/tmp/rx.log") if l.strip()]
assert seqs==list(range(1,len(seqs)+1)), f"seq not gap-free: {seqs[:20]}"
print("SEQ_GAPFREE", len(seqs))
PY'
ls -la <repo>/isidet/logs/events/   # events_*.csv present on host mount, rows == datagrams
```
Expected: `seq` 1..N gap-free; datagram count ≈ distinct parcels in the clip (no over-fire); CSV row count == datagram count.

- [ ] **Step 4: Toggle time-dedup OFF via the API and confirm it takes effect live**

```bash
curl -s -m5 -X POST http://127.0.0.1:9501/api/settings -H "X-Dev-Token: $T" -H 'Content-Type: application/json' -d '{"dedup_time_enabled": false}'
docker exec deploy-web-1 sh -c 'grep "dedup-suppressed" /var/log/* 2>/dev/null; docker logs deploy-web-1 2>&1 | tail -5' || docker logs deploy-web-1 2>&1 | grep -c "dedup-suppressed"
curl -s -m5 http://127.0.0.1:9501/api/settings -H "X-Dev-Token: $T" | python3 -c 'import sys,json;print("dedup_time_enabled=",json.load(sys.stdin).get("dedup_time_enabled"))'
```
Expected: setting reads back `false`; no crash; (on clean testvid, suppression count stays 0 either way — toggle is verified by the persisted/returned value and a clean restart).

- [ ] **Step 5: Final commit (none expected — verification only). Record the result.**

If all checks pass, the feature is verified end-to-end. If over-fire appears on the *site* later, tune `dedup_interval_ms` / ByteTrack using the gap-histogram method in the spec.

---

## Self-review notes (for the planner)

- Spec coverage: center anchor (T3), one-per-parcel via track-ID base + time toggle (T1/T2/T6/T7), settings exposure both backends (T5-T8), seq-post-dedup preserved (T2 ordering), ByteTrack stability (T4), docs (T9), e2e + tuning method (T10). ✓
- `seq` ordering requirement is satisfied structurally: dedup happens in `vision_engine` (T2) before `publish()` stamps `seq` — no task changes that order.
- Open items carried from spec: real-footage tuning of `dedup_interval_ms` + ByteTrack (placeholders here); optional cadence confirmation with automaticien.

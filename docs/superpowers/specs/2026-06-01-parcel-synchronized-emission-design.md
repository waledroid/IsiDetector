# Parcel-synchronized UDP emission — design

**Date:** 2026-06-01
**Branch:** `fps`
**Status:** approved design, pending spec review → implementation plan

## Context

IsiDetector is integrated as the **backup classifier** behind Celio's existing barcode-based
sorter (the "automate"/PLC, programmed in IsiPlc). The barcode scanner is the primary identifier;
our `carton`/`polybag` classification is consumed by the PLC **only on a barcode no-read today
(Phase A)**, and later as an always-on cross-check once IsiDetector is certified (Phase B). The
delivery path is identical in both phases — only the PLC-side consumption changes.

The same automate/PLC is used at **every** site, so this becomes the **new default behavior for all
deployments** (no per-site mode flag).

### How the PLC actually consumes our data (reverse-engineered from `Programme Automate Celio/`)

- The PLC tracks **every physical parcel** via photocells (`USER->Paquet[]`), reads its barcode
  (`Code_Barre`), and routes by destination.
- It correlates an incoming camera classification to a parcel **by timing** (confirmed by the
  automaticien): the parcel passes a reference point, and a classification arriving inside a time
  window is bound to that parcel; otherwise the parcel is flagged `NON_LU_CAM` ("not read by
  camera").
- Time base is **milliseconds** (`BLOC0017`: "base de temps 1/10 ms", stored `/10`).
  - **Camera acceptance window: 600–1100 ms** (`Mini_Camera`/`Maxi_Camera`, `Seuil_Maxi_Tps_Camera`
    in `BLOC0002`) → **500 ms tolerance**.
  - Stations (B2…B24 / W3…W23) spaced ~1100 ms; transit ≈ 14 s.
- The automaticien's explicit requirement: *"tu choisis un endroit fixe et je gère la fenêtre de
  temps de mon côté"* — **we emit at one fixed belt position; he calibrates the window to it.**
- The bridge `DetectionCarton.exe` parses our JSON (`class`+`id` via regex), maps
  `carton→ClasseObjet=1`, `polybag→0`, forwards ModbusUDP to the PLC, and dedups only **exact
  duplicate `id`s** (network dupes) — it does **not** collapse our multi-ID over-firing.

### Why the current behavior breaks this

1. **Leading-edge anchor** (`_ANCHOR_MAP` in `isidet/src/shared/vision_engine.py`) makes the trigger
   position depend on parcel **size** → inconsistent position. (Originally chosen to maximize a
   sorter-gate reaction window — but **the PLC owns the gate here**, so that rationale is obsolete.)
2. **Over-firing**: ByteTrack ID churn splits one physical parcel into several track IDs, each
   emitting a datagram (the `1271 datagrams sent` observation). Under timing correlation, an *extra*
   datagram binds to the **wrong** parcel — actively corrupting the mapping.

These — not packet loss — are the likely cause of the reported "20 of 74" (`NON_LU_CAM`).

## Goal

> Emit **exactly one datagram per physical parcel**, at the instant the parcel's **center** crosses
> one fixed line, with consistent latency. The PLC's 500 ms window absorbs the rest.

## Design

### 1. Center anchor (consistent position)
Trigger line-crossing on the bbox **center** (`sv.Position.CENTER`) instead of the size-dependent
leading-edge anchor. Every parcel then emits from the same belt position regardless of size.
- `line_position` / `line_orientation` unchanged (where the line sits).
- `belt_direction` retained for in/out crossing direction; it no longer selects the anchor.
- The leading-edge anchor mapping is **kept available via config** (not default) — cheap insurance,
  no regression for any hypothetical non-PLC use.

### 2. One datagram per parcel — **selectable dedup strategy** (operator setting)

Over-firing (one physical parcel → several emissions, from ByteTrack ID churn) is killed by a
**configurable dedup strategy**, exposed in Settings. Two methods, selectable:

- **Time-based (default).** Suppress a line-crossing that occurs within `dedup_interval_ms` of the
  previous emission. **Justified by the live flow:** parcels arrive **clearly separated** on this
  belt (stations are ~1100 ms apart; observed spacing is comfortably wider than ID-churn, which is
  <100 ms). A reasonable interval (default **300 ms**) absorbs churn while never reaching real
  parcels. *Accepted trade-off:* in the **rare** case of joined/overlapping parcels, the second may
  be suppressed — an acceptable penalty given how infrequently it occurs.
- **Track-ID-based (toggle).** "One ByteTrack ID → one emission" via the existing `counted_ids` set:
  each parcel fires once for its track's lifetime; two close parcels are two IDs → two emissions, so
  it has **no** close-parcel merge risk. Best paired with tracker stabilization (below). Toggle this
  on when ID churn is under control, or when tighter parcel spacing is expected.

Both are config-driven and selectable. **Default ships as `both` (300 ms)** — the most protective
combination. Exact rule (an **AND**): emit a crossing only if **(its track ID is not in
`counted_ids`) AND (≥ `dedup_interval_ms` since the last emitted datagram)**; on emit, add the ID to
`counted_ids` and update the global last-emit timestamp. This catches *churned-new-ID* over-fire
(via the time term) **and** *stable-ID re-cross/straddle* over-fire (via the track-ID term) — modes
that each method alone misses one of. The only cost is merging two real parcels closer than the
interval, which the live flow rules out (min observed spacing 703 ms ≫ 300 ms).

**Tracker stability (supporting either mode):** tune ByteTrack (longer `lost_track_buffer`, looser
`match_thresh`) + keep CLAHE/SpecularGuard on dark polybags, so one parcel keeps one ID. This makes
track-ID dedup reliable and reduces the churn that time dedup has to absorb.

**Overlapping / touching parcels** are a *detection* (segmentation) matter, not a dedup one, and are
the accepted rare-case penalty above — out of scope for this sync design.

### 3. Per-frame data flow (unchanged except the trigger rule)
detect → track → for each tracked **center** crossing the line, not in `counted_ids` **and** past
the dedup guard → emit **one** datagram (`class`, `seq`, `ts`; `id` kept for info/logging) +
`EventLogger.log`. UDP/`seq`/persistence already shipped.

### 4. Config & defaults
**Operator-facing (Settings UI + `settings.json`)** — dedup is exposed so it can be changed on-site
without redeploy:
- `dedup_strategy: both` — one of `both` | `time` | `track_id` | `off` (default **`both`**).
- `dedup_interval_ms: 300` — used by `both`/`time` modes (tunable; default 300 ms).
- (`track_id` term needs no operator field; it keys on the existing `counted_ids`.)

This means the Settings plumbing must carry these: `settings.json` keys, the Settings form
(`templates/index.html` + `static/js/main.js`), `/api/settings` validation, and wiring into
`StreamHandler` → `VisionEngine`. Both web backends (Flask + FastAPI) stay in parity.

**Engine/config-file (mode-driven `inference` config; not operator-facing):**
- `trigger_anchor: center`  (was effectively leading-edge)
- ByteTrack `lost_track_buffer`, `match_thresh` tuned for stability — supports either dedup mode.

All are config values, not hard-codes, so they can be calibrated per belt without code changes.

## Edge cases / risks
- **ID churn → over-firing** (the observed problem). Killed by the chosen dedup strategy: `time`
  collapses near-simultaneous re-detections; `track_id` emits once per ID. Tracker stabilization
  reduces churn for both.
- **`both`/`time` merging two close parcels.** Possible *in principle* (the time term), but the live
  flow shows parcels **clearly separated** (min 703 ms ≫ the 300 ms interval), so it effectively
  never triggers. The **rare** joined/overlap case is an **accepted penalty**; if it ever becomes a
  problem, switch the setting to `track_id` (no merge risk). Recorded as a deliberate trade-off.
- **Overlapping/touching parcels seen as one detection** → one datagram (under-count). A
  **segmentation/detection** limitation (model quality), independent of dedup mode; the accepted
  rare-case penalty above. Out of scope here.
- **Latency spikes** (FPS drop) push a datagram out of the 500 ms window → `NON_LU_CAM`. Keep
  per-frame latency bounded; FPS/UDP latency already surface in `/api/performance`.
- **Center anchor + belt_direction**: confirm in/out counting still uses `belt_direction` correctly
  once the anchor is decoupled from it.

## Verification
- **Local (`testvid.mp4`, CPU/OpenVINO):** exactly one datagram per visible parcel (no over-fire),
  `seq` gap-free, all emitted as the parcel center crosses the line; dedup-suppression log sane.
- **On the belt:** datagram count vs physical parcel count over a fixed window; `NON_LU_CAM` rate
  should drop sharply. Compare our event-log count, the bridge's received count, and the PLC's
  correlated count to localize any residual gap.

## Open items
1. (Low priority) Confirm with automaticien the **max cadence / min parcel pitch** → sanity-check
   that `dedup_interval_ms` (default 300 ms) stays below it. Live observation already shows parcels
   clearly separated, so this is a confirmation, not a blocker.
2. Confirm `belt_direction` semantics post-anchor-change.
3. ByteTrack tuning values to be set against real belt footage (placeholder defaults until then).

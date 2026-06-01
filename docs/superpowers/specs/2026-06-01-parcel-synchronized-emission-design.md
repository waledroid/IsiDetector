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

### 2. One datagram per parcel (kill over-firing)
Two layers, in `vision_engine.py`:
- **(a) Tracking stability (root cause):** tune ByteTrack so one physical parcel keeps one track ID
  — longer `lost_track_buffer`, looser `match_thresh` — and keep CLAHE/SpecularGuard on dark
  polybags. With stable IDs, `counted_ids` already yields one emit per parcel.
- **(b) Dedup guard (safety net):** suppress a line-crossing that occurs within a configurable
  **minimum interval** of the previously emitted crossing, to absorb residual ID-churn. Every
  suppressed crossing is **logged** so over-aggressive tuning is visible.

### 3. Per-frame data flow (unchanged except the trigger rule)
detect → track → for each tracked **center** crossing the line, not in `counted_ids` **and** past
the dedup guard → emit **one** datagram (`class`, `seq`, `ts`; `id` kept for info/logging) +
`EventLogger.log`. UDP/`seq`/persistence already shipped.

### 4. Config & defaults
New config keys (with the mode-driven `inference` config; defaults shown):
- `trigger_anchor: center`  (was effectively leading-edge)
- `dedup_min_interval_ms: 200`  (absorbs <100 ms churn, ≪ 500 ms window; **tunable**)
- ByteTrack: `lost_track_buffer`, `match_thresh` tuned for stability.

All are config values, not hard-codes, so they can be calibrated per belt without code changes.

## Edge cases / risks
- **Two parcels closer than `dedup_min_interval_ms` merge** (under-count). Mitigation: keep the
  interval conservative, log suppressions, and tune against real footage. **Open item:** ask the
  automaticien for max cadence (parcels/min) or minimum parcel pitch to set a safe ceiling.
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
1. Ask automaticien: **max cadence / min parcel pitch** → finalize `dedup_min_interval_ms` ceiling.
2. Confirm `belt_direction` semantics post-anchor-change.
3. ByteTrack tuning values to be set against real belt footage (placeholder defaults until then).

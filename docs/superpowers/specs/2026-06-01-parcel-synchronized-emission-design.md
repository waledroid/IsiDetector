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

### 2. One datagram per parcel — dedup by **identity**, not time

The mechanism is **identity-based**, in `vision_engine.py`:

- **Primary — one track ID → one emission.** Each physical parcel is one ByteTrack track; the
  existing `counted_ids` set emits it exactly once. Crucially, **two parcels close together are two
  *different* track IDs → two emissions**, even 50 ms apart — identity dedup handles tight spacing
  correctly, which a time-based filter cannot.
- **Enabler — tracking stability.** The only reason over-firing happens is ID *churn* (one parcel
  flickering into several track IDs). Fix the root cause: tune ByteTrack (longer `lost_track_buffer`,
  looser `match_thresh`) + keep CLAHE/SpecularGuard on dark polybags, so one parcel keeps one ID.
- **Safety net — a *bounded* time guard, demoted.** As a last resort against any residual churn,
  optionally suppress a second crossing within a **very short** interval of the previous emission.
  This interval MUST be set **well below the minimum real parcel pitch**, so it can only ever catch
  same-parcel re-detection (churn is <100 ms) and can **never** merge two real parcels. Every
  suppression is **logged**. If tracking is stable enough, this guard stays effectively off.

> **Time is never the primary discriminator.** Identity is. The guard is a small, bounded backstop,
> not the mechanism.

**Overlapping / touching parcels** are a *detection* matter, not a dedup one: whether we emit one or
two datagrams depends on whether the model **segments** them as two instances or one blob. Tracked
as a separate detection-quality concern, out of scope for this sync design.

### 3. Per-frame data flow (unchanged except the trigger rule)
detect → track → for each tracked **center** crossing the line, not in `counted_ids` **and** past
the dedup guard → emit **one** datagram (`class`, `seq`, `ts`; `id` kept for info/logging) +
`EventLogger.log`. UDP/`seq`/persistence already shipped.

### 4. Config & defaults
New config keys (with the mode-driven `inference` config; defaults shown):
- `trigger_anchor: center`  (was effectively leading-edge)
- ByteTrack: `lost_track_buffer`, `match_thresh` tuned for stability — **the primary lever.**
- `dedup_guard_ms: 0`  (the bounded safety net; **off by default**, enabled only if churn persists.
  When set, MUST stay well below the min parcel pitch — see Open items.)

All are config values, not hard-codes, so they can be calibrated per belt without code changes.

## Edge cases / risks
- **ID churn → over-firing** (the observed problem). Primary mitigation is tracking stability so one
  parcel keeps one ID; the bounded `dedup_guard_ms` is the backstop. Because identity is the
  discriminator, **two close real parcels are not at risk of merging** (distinct IDs) — the only way
  the guard could merge real parcels is if it were mis-set *above* the min pitch, which the Open-item
  cadence check exists to prevent.
- **Overlapping/touching parcels seen as one detection** → one datagram (under-count). This is a
  **segmentation/detection** limitation (model quality), not a dedup issue; out of scope here.
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
1. Ask automaticien: **max cadence / min parcel pitch** → only to confirm the `dedup_guard_ms`
   ceiling stays safely below it (guard is off by default, so this is a safety bound, not a blocker).
2. Confirm `belt_direction` semantics post-anchor-change.
3. ByteTrack tuning values to be set against real belt footage (placeholder defaults until then).

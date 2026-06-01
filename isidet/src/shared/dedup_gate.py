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

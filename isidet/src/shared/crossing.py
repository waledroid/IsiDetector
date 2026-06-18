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

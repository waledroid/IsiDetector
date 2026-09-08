"""Predictive line trigger — fire the sorter datagram at *predicted crossing + offset*.

Why: the observed-crossing trigger fires on the first processed frame after the
leading edge passes the line. A detection gap at the line (glare on a polybag)
delays that by several frames, and the sorter sees a 500 ms spread. Here each
track's crossing time is estimated from a least-squares fit of its recent
positions while it is still upstream, refined by interpolation once samples
straddle the line, and a scheduler thread fires at the exact millisecond.

All times are monotonic milliseconds on ONE clock: the frame capture stamp from
LiveReader and the scheduler's ``clock()`` (default ``time.monotonic()*1000``).

Coordinates are normalised internally so the "after" side is always *greater*:
``s = pos if after_is_greater else -pos``. Distance-to-line ``L - s`` is positive
while upstream, and the belt velocity must be positive.

Design: docs/superpowers/specs/2026-09-08-predictive-trigger-design.md
"""
from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

HISTORY = 12                 # samples kept per track
MIN_SAMPLES = 3              # samples needed for a per-track fit
CONFIDENT_SAMPLES = 6        # samples before a track's v feeds the belt-speed median
BELT_WINDOW = 25             # per-track velocities kept for the belt-speed median
VEL_DEV_FRAC = 0.5           # |v - belt| / belt above this → use belt speed instead
CANCEL_UNSEEN_FRAMES = 10    # armed + unseen this many frames …
CANCEL_UPSTREAM_FRAC = 0.20  # … while still > 20 % of the extent upstream → cancel
FORGET_MS = 2000.0           # drop state for ids unseen this long
ERR_WINDOW = 500             # est_err samples kept for stats()
SUMMARY_EVERY = 100          # fires between summary log lines
BELT_MIN_SUPPORT = 5         # belt-median entries before it may override a track's own fit
BELT_FEED_DISP_FRAC = 0.05   # a track must have moved >= 5 % of the extent to feed the median
V_FLOOR_MS = 20000.0         # v floor = extent / 20 s: slower than that is "not moving"
REVERSED_WARN_AFTER = 5      # confident tracks moving the WRONG way before we warn
REVERSED_WARN_EVERY_MS = 60000.0


class _Track:
    __slots__ = ('samples', 'cls', 'unseen', 'last_seen', 't_fire', 'fired_at',
                 'src', 'v', 'n_fit', 't_cross', 'refined')

    def __init__(self, cls: str, t_ms: float):
        self.samples: Deque[Tuple[float, float]] = deque(maxlen=HISTORY)
        self.cls = cls
        self.unseen = 0
        self.last_seen = t_ms
        self.t_fire: Optional[float] = None      # armed fire time (None = not armed)
        self.fired_at: Optional[float] = None    # clock time we fired (None = not yet)
        self.src = ''                            # 'extrap' | 'belt' | 'interp' | 'obs'
        self.v = 0.0
        self.n_fit = 0
        self.t_cross: Optional[float] = None
        self.refined = False                     # interpolated (straddle) estimate seen


def _fit(samples: Iterable[Tuple[float, float]]) -> Tuple[float, float, float]:
    """Least squares s = a + v*t → (v, a, t_last). Caller ensures len >= 2."""
    pts = list(samples)
    n = len(pts)
    mt = sum(t for t, _ in pts) / n
    ms = sum(s for _, s in pts) / n
    var = sum((t - mt) ** 2 for t, _ in pts)
    if var <= 0:
        return 0.0, ms, pts[-1][0]
    cov = sum((t - mt) * (s - ms) for t, s in pts)
    v = cov / var
    return v, ms - v * mt, pts[-1][0]


class PredictiveTrigger:
    def __init__(self, line_coord: float, after_is_greater: bool, extent: float,
                 offset_ms: int = 0,
                 on_fire: Optional[Callable[[int, str, dict], None]] = None,
                 clock: Optional[Callable[[], float]] = None,
                 start_thread: bool = True):
        self._sign = 1.0 if after_is_greater else -1.0
        self._L = self._sign * float(line_coord)
        self._extent = float(extent)
        self._v_floor = self._extent / V_FLOOR_MS
        self.offset_ms = float(offset_ms)
        self.on_fire = on_fire
        self._clock = clock or (lambda: time.monotonic() * 1000.0)
        self._tracks: Dict[int, _Track] = {}
        self._belt: Deque[float] = deque(maxlen=BELT_WINDOW)
        self.errors_ms: Deque[float] = deque(maxlen=ERR_WINDOW)
        self.fire_count = 0
        self.reversed_count = 0          # confident tracks that moved toward the 'before' side
        self._reversed_warned_at = None
        self._cond = threading.Condition()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        if start_thread:
            self.start()

    # ------------------------------------------------------------------ public
    @property
    def belt_speed(self) -> float:
        """Median belt velocity (px/ms, positive = toward the after side); 0 if unknown."""
        return statistics.median(self._belt) if self._belt else 0.0

    def track_ids(self) -> List[int]:
        return list(self._tracks)

    def pending_fire_time(self, track_id: int) -> Optional[float]:
        tr = self._tracks.get(int(track_id))
        return tr.t_fire if tr and tr.fired_at is None else None

    def configure(self, offset_ms: int) -> None:
        """Live offset change — reschedules every armed track."""
        with self._cond:
            delta = float(offset_ms) - self.offset_ms
            self.offset_ms = float(offset_ms)
            for tr in self._tracks.values():
                if tr.t_fire is not None and tr.fired_at is None:
                    tr.t_fire += delta
            self._cond.notify()

    def observe(self, track_ids, positions, classes, t_ms: float) -> None:
        """One call per frame. ``positions``: leading-edge coord on the crossing axis."""
        seen = set()
        with self._cond:
            for tid, pos, cls in zip(track_ids, positions, classes):
                tid = int(tid)
                seen.add(tid)
                s = self._sign * float(pos)
                tr = self._tracks.get(tid)
                if tr is None:
                    tr = self._tracks[tid] = _Track(cls, t_ms)
                tr.cls = cls
                tr.unseen = 0
                tr.last_seen = t_ms
                prev = tr.samples[-1] if tr.samples else None
                tr.samples.append((t_ms, s))
                self._estimate(tr, prev, t_ms, s)
            for tid, tr in self._tracks.items():
                if tid in seen:
                    continue
                tr.unseen += 1
                if (tr.t_fire is not None and tr.fired_at is None
                        and tr.unseen >= CANCEL_UNSEEN_FRAMES
                        and (self._L - tr.samples[-1][1]) > CANCEL_UPSTREAM_FRAC * self._extent):
                    logger.info(f"[PRED] cancel id={tid} unseen={tr.unseen} "
                                f"dist={self._L - tr.samples[-1][1]:.0f}px")
                    tr.t_fire = None
            self._cond.notify()

    def report_observed(self, track_id: int, cls: str, t_ms: float) -> None:
        """LineZone / latch saw a crossing. Arm at t+offset only if we have no estimate."""
        tid = int(track_id)
        with self._cond:
            tr = self._tracks.get(tid)
            if tr is None:
                tr = self._tracks[tid] = _Track(cls, t_ms)
            if tr.fired_at is not None or tr.t_fire is not None:
                return
            tr.t_cross, tr.src = float(t_ms), 'obs'
            tr.t_fire = tr.t_cross + self.offset_ms
            self._cond.notify()

    def forget(self, keep_ids, now_ms: Optional[float] = None) -> None:
        """Drop state for ids absent >= FORGET_MS (keeps armed ones until they fire/cancel)."""
        now = self._clock() if now_ms is None else now_ms
        keep = {int(i) for i in keep_ids}
        with self._cond:
            for tid in list(self._tracks):
                tr = self._tracks[tid]
                if tid in keep or (now - tr.last_seen) < FORGET_MS:
                    continue
                if tr.t_fire is not None and tr.fired_at is None:
                    continue
                del self._tracks[tid]

    def due(self, now_ms: Optional[float] = None) -> List[Tuple[int, str, dict]]:
        """Fire every armed track whose time has come. Callback runs outside the lock."""
        now = self._clock() if now_ms is None else now_ms
        fired: List[Tuple[int, str, dict]] = []
        with self._cond:
            for tid, tr in self._tracks.items():
                if tr.t_fire is not None and tr.fired_at is None and tr.t_fire <= now:
                    tr.fired_at = now
                    meta = {'src': tr.src, 'offset_ms': self.offset_ms, 'n': tr.n_fit,
                            'v': round(tr.v * self._sign, 4), 'late_ms': round(now - tr.t_fire, 1)}
                    fired.append((tid, tr.cls, meta))
        for tid, cls, meta in fired:
            self.fire_count += 1
            logger.info(f"[PRED] fire id={tid} cls={cls} src={meta['src']} "
                        f"offset={meta['offset_ms']:.0f} n={meta['n']} v={meta['v']} "
                        f"late={meta['late_ms']}ms")
            if self.on_fire:
                try:
                    self.on_fire(tid, cls, meta)
                except Exception as e:      # never let a consumer error kill the scheduler
                    logger.error(f"[PRED] on_fire failed for id={tid}: {e}")
            if self.fire_count % SUMMARY_EVERY == 0:
                self._log_summary()
        return fired

    def stats(self) -> dict:
        errs = sorted(self.errors_ms)
        if not errs:
            return {'n': 0}
        p = lambda q: errs[min(len(errs) - 1, int(q * len(errs)))]
        return {'n': len(errs), 'median_ms': round(statistics.median(errs), 1),
                'p95_abs_ms': round(sorted(abs(e) for e in errs)[min(len(errs) - 1, int(0.95 * len(errs)))], 1),
                'min_ms': round(errs[0], 1), 'max_ms': round(errs[-1], 1), 'belt_px_ms': round(self.belt_speed, 4)}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name='pred-trigger', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        if self._thread and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    # ----------------------------------------------------------------- private
    def _estimate(self, tr: _Track, prev: Optional[Tuple[float, float]], t_ms: float, s: float) -> None:
        if tr.fired_at is not None:
            # Already fired: a straddle still tells us how good the estimate was.
            if prev is not None and not tr.refined and prev[1] < self._L <= s and tr.t_cross is not None:
                t_int = prev[0] + (self._L - prev[1]) / (s - prev[1]) * (t_ms - prev[0])
                self.errors_ms.append(tr.fired_at - (t_int + self.offset_ms))
                tr.refined = True
            return
        if tr.refined:
            return
        # Straddle → exact-ish interpolation, best estimate we will ever have.
        if prev is not None and prev[1] < self._L <= s:
            t_int = prev[0] + (self._L - prev[1]) / (s - prev[1]) * (t_ms - prev[0])
            tr.t_cross, tr.src, tr.refined = t_int, 'interp', True
            tr.t_fire = t_int + self.offset_ms
            return
        if len(tr.samples) == CONFIDENT_SAMPLES:
            # Direction sanity (either side of the line): a track clearly travelling
            # toward the 'before' side means belt_direction is reversed in Settings
            # (trigger anchor on the TRAILING edge, predictions impossible).
            v_chk, _, _ = _fit(tr.samples)
            disp_chk = tr.samples[-1][1] - tr.samples[0][1]
            if v_chk < -self._v_floor and -disp_chk >= BELT_FEED_DISP_FRAC * self._extent:
                self.reversed_count += 1
                self._maybe_warn_reversed(t_ms)
        if s >= self._L:
            return                              # past the line, no straddle: fallback via report_observed
        belt = self.belt_speed
        belt_ok = belt > self._v_floor and len(self._belt) >= BELT_MIN_SUPPORT
        n = len(tr.samples)
        if n >= MIN_SAMPLES:
            v, a, t_last = _fit(tr.samples)
            # Feed the belt median only from tracks that really travelled: a
            # near-static false detection must never define the belt speed.
            disp = tr.samples[-1][1] - tr.samples[0][1]
            if n >= CONFIDENT_SAMPLES and v > self._v_floor and disp >= BELT_FEED_DISP_FRAC * self._extent:
                self._belt.append(v)
                belt = self.belt_speed
                belt_ok = len(self._belt) >= BELT_MIN_SUPPORT
            if v <= self._v_floor:
                if not belt_ok:
                    return                      # wrong way / stopped and no belt reference
                v, s_now, src = belt, s, 'belt'
            elif belt_ok and abs(v - belt) / belt > VEL_DEV_FRAC:
                v, s_now, src = belt, s, 'belt'
            else:
                v, s_now, src = v, a + v * t_last, 'extrap'
        elif belt_ok:
            v, s_now, src = belt, s, 'belt'
        else:
            return
        tr.v, tr.n_fit, tr.src = v, n, src
        tr.t_cross = t_ms + max(0.0, (self._L - s_now)) / v
        tr.t_fire = tr.t_cross + self.offset_ms

    def _maybe_warn_reversed(self, t_ms: float) -> None:
        if self.reversed_count < REVERSED_WARN_AFTER:
            return
        if (self._reversed_warned_at is not None
                and (t_ms - self._reversed_warned_at) < REVERSED_WARN_EVERY_MS):
            return
        self._reversed_warned_at = t_ms
        logger.warning(f"[PRED] {self.reversed_count} tracks moved AGAINST belt_direction — "
                       "the setting looks reversed (trigger anchor is on the TRAILING edge; "
                       "predictions impossible). Flip belt_direction in Settings.")

    def _log_summary(self) -> None:
        logger.info(f"[PRED] summary fires={self.fire_count} {self.stats()}")

    def _run(self) -> None:
        while True:
            with self._cond:
                if not self._running:
                    return
                now = self._clock()
                nxt = min((tr.t_fire for tr in self._tracks.values()
                           if tr.t_fire is not None and tr.fired_at is None), default=None)
                if nxt is None:
                    self._cond.wait()
                    continue
                if nxt > now:
                    self._cond.wait(timeout=(nxt - now) / 1000.0)
                    continue
            self.due()

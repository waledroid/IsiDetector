"""PredictiveTrigger unit tests — fake clock, no scheduler thread.

Geometry: horizontal line at y=400 in a 576-px-tall frame, belt top→bottom
(after_is_greater=True). 25 fps → 40 ms frames. v = 0.5 px/ms = 20 px/frame.
A parcel starting at y=100 at t=0 crosses at t=600 ms.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.shared.predictive_trigger import PredictiveTrigger

LINE, EXTENT, DT, V = 400.0, 576.0, 40.0, 0.5


def make(offset_ms=0, fired=None):
    fired = fired if fired is not None else []
    trig = PredictiveTrigger(line_coord=LINE, after_is_greater=True, extent=EXTENT,
                             offset_ms=offset_ms,
                             on_fire=lambda tid, cls, meta: fired.append((tid, cls)),
                             start_thread=False)
    return trig, fired


def run_track(trig, tid, t_end, t0=0.0, y0=100.0, cls='carton'):
    """Observe a constant-velocity track from t0 up to and including t_end."""
    t = t0
    while t <= t_end + 1e-9:
        trig.observe([tid], [y0 + V * (t - t0)], [cls], t)
        t += DT


def test_constant_velocity_fires_at_crossing_plus_offset():
    trig, fired = make(offset_ms=300)
    run_track(trig, 1, t_end=400)          # last sample y=300, still upstream
    assert trig.pending_fire_time(1) is not None
    assert abs(trig.pending_fire_time(1) - 900) < 2
    assert trig.due(899) == []
    trig.due(900)
    assert fired == [(1, 'carton')]


def test_negative_offset_fires_before_line():
    trig, fired = make(offset_ms=-200)
    run_track(trig, 1, t_end=360)          # y=280
    assert abs(trig.pending_fire_time(1) - 400) < 2
    trig.due(400)
    assert fired == [(1, 'carton')]


def test_detection_gap_at_line_still_fires_on_extrapolation():
    trig, fired = make(offset_ms=0)
    run_track(trig, 1, t_end=520)          # y=360, 2 frames before the line
    for k in range(1, 9):                  # 8 frames with the track missing
        trig.observe([], [], [], 520 + k * DT)
    assert trig.pending_fire_time(1) is not None
    trig.due(600)
    assert fired == [(1, 'carton')]
    # Track reappears past the line: straddle refines the estimate, no re-arm,
    # and the error is recorded for the stats line.
    trig.observe([1], [540.0], ['carton'], 880)
    assert trig.pending_fire_time(1) is None
    assert fired == [(1, 'carton')]
    assert len(trig.errors_ms) == 1 and abs(trig.errors_ms[0]) < 2


def test_straddle_refines_fire_time():
    trig, fired = make(offset_ms=100)
    run_track(trig, 1, t_end=560)          # y=380 → predicted crossing 600 → fire 700
    assert abs(trig.pending_fire_time(1) - 700) < 2
    # Parcel actually moved faster: observed at y=410 at t=600 → interp 586.7
    trig.observe([1], [410.0], ['carton'], 600)
    assert abs(trig.pending_fire_time(1) - 686.7) < 1
    assert trig.due(686) == []
    trig.due(687)
    assert fired == [(1, 'carton')]


def test_far_upstream_vanish_is_cancelled():
    trig, fired = make(offset_ms=0)
    run_track(trig, 1, t_end=80)           # 3 samples, y=100..140, armed
    assert trig.pending_fire_time(1) is not None
    for k in range(1, 11):                 # unseen for 10 frames, far from the line
        trig.observe([], [], [], 80 + k * DT)
    assert trig.pending_fire_time(1) is None
    trig.due(10_000)
    assert fired == []


def test_observed_fallback_for_track_without_prediction():
    trig, fired = make(offset_ms=250)
    trig.observe([7], [420.0], ['polybag'], 1000)   # first seen past the line
    assert trig.pending_fire_time(7) is None
    trig.report_observed(7, 'polybag', 1000)
    assert abs(trig.pending_fire_time(7) - 1250) < 1e-6
    trig.due(1250)
    assert fired == [(7, 'polybag')]


def test_observed_report_does_not_override_prediction():
    trig, fired = make(offset_ms=300)
    run_track(trig, 1, t_end=560)          # armed at 900
    trig.report_observed(1, 'carton', 640)  # LineZone sees it a frame late
    assert abs(trig.pending_fire_time(1) - 900) < 2


def test_no_rearm_after_fire():
    trig, fired = make(offset_ms=0)
    run_track(trig, 1, t_end=560)
    trig.due(600)
    assert fired == [(1, 'carton')]
    trig.observe([1], [420.0], ['carton'], 640)
    trig.report_observed(1, 'carton', 640)
    trig.observe([1], [440.0], ['carton'], 680)
    assert trig.pending_fire_time(1) is None
    trig.due(5000)
    assert fired == [(1, 'carton')]


def test_wrong_direction_gives_no_prediction():
    trig, fired = make(offset_ms=0)
    for k in range(6):
        trig.observe([1], [300.0 - 20 * k], ['carton'], k * DT)
    assert trig.pending_fire_time(1) is None


def test_belt_speed_fallback_for_short_track():
    trig, fired = make(offset_ms=0)
    run_track(trig, 1, t_end=440)          # 12 samples: >=5 feed the belt median
    assert abs(trig.belt_speed - V) < 0.01
    trig.observe([2], [380.0], ['polybag'], 1000)   # single sample, 20 px upstream
    assert trig.pending_fire_time(2) is not None
    assert abs(trig.pending_fire_time(2) - 1040) < 2


def test_configure_offset_reschedules_pending():
    trig, fired = make(offset_ms=0)
    run_track(trig, 1, t_end=400)
    trig.configure(offset_ms=150)
    assert abs(trig.pending_fire_time(1) - 750) < 2


def test_forget_drops_stale_state():
    trig, fired = make(offset_ms=0)
    run_track(trig, 1, t_end=560)
    trig.due(600)
    trig.forget(keep_ids=[], now_ms=600 + 2500)
    assert 1 not in trig.track_ids()


def test_scheduler_thread_fires_on_real_clock():
    import time, threading
    fired, done = [], threading.Event()
    trig = PredictiveTrigger(line_coord=LINE, after_is_greater=True, extent=EXTENT, offset_ms=0,
                             on_fire=lambda tid, cls, meta: (fired.append(tid), done.set()),
                             start_thread=True)
    try:
        now = time.monotonic() * 1000.0
        # 3 samples 40 ms apart ending "now", 20 px upstream → crossing in 40 ms
        for k, y in enumerate((340.0, 360.0, 380.0)):
            trig.observe([1], [y], ['carton'], now - (2 - k) * DT)
        assert done.wait(timeout=1.0), "scheduler thread did not fire"
        assert fired == [1]
        late = time.monotonic() * 1000.0 - (now + 40.0)
        assert -5 < late < 60, f"fired {late:.1f} ms off schedule"
    finally:
        trig.stop()


def test_static_false_track_does_not_define_belt_speed():
    trig, fired = make(offset_ms=0)
    for k in range(12):                    # jittering, near-static detection
        trig.observe([9], [200.0 + (k % 2) * 0.5], ['carton'], k * DT)
    assert trig.belt_speed == 0.0
    assert trig.pending_fire_time(9) is None


def test_reversed_direction_is_flagged():
    trig, fired = make(offset_ms=0)
    for tid in range(1, 7):                # 6 parcels travelling the wrong way
        for k in range(8):
            trig.observe([tid], [500.0 - 20 * k], ['carton'], tid * 1000 + k * DT)
    assert trig.reversed_count >= 5
    assert trig._reversed_warned_at is not None

"""Offline replay: observed-crossing vs predictive trigger on recorded footage.

Dev-box tool (not shipped to the runtime path). Paces frames at the video's
native fps in REAL time so the predictor's scheduler thread runs against the
same monotonic clock as production. Prints per-mode counts and timing spread.

    PYTHONPATH=isidet python isidet/tests/replay_predictive.py \
        --video isidet/data/cam_20260602_105526.mp4 --seconds 120 --offset 0
"""
import argparse, logging, os, statistics, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2, numpy as np
import supervision as sv
from src.inference.yolo_inferencer import YOLOInferencer
from src.shared.vision_engine import VisionEngine

ROI_X = (0.31, 0.69)       # belt band in the 2880-wide site recording
LINE, ORIENT, BELT = 0.71, 'horizontal', 'top_to_bottom'


def run(mode, args, log_dir):
    inf = YOLOInferencer(args.weights, conf_threshold=0.55, imgsz=320)
    cfg = {'inference': {'conf_threshold': 0.55, 'predictive_trigger': mode == 'predictive',
                         'trigger_offset_ms': args.offset, 'dedup_time_enabled': True,
                         'dedup_interval_ms': 300, 'count_interpolate': True,
                         'logging': {'log_dir': f'{log_dir}/{mode}', 'retention_days': 1}},
           'bytetrack': {'frame_rate': 25, 'track_buffer': 60, 'match_thresh': 0.7}}
    eng = VisionEngine(inf, cfg)
    if not hasattr(eng, 'configure_predictive'): print('(baseline engine: no predictive support)')
    eng.line_orientation, eng.line_position, eng.belt_direction = ORIENT, args.line, args.belt
    events = []
    eng.on_event = lambda ev: (events.append((time.monotonic() * 1000.0, dict(ev))), print(f"EV seq={ev.get('seq')} id={ev['id']} cls={ev['class']} src={ev.get('src')}"))
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    n = int(args.seconds * fps)
    counts, late_px, t0 = {}, [], time.monotonic() * 1000.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.roi_px:                                          # site settings.json roi_points x-range
        x1, x2 = (int(v) for v in args.roi_px.split(','))
    else:
        x1, x2 = int(w * ROI_X[0]), int(w * ROI_X[1])
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frame = frame[:, x1:x2]
        h, ww = frame.shape[:2]
        s = args.max_side / max(h, ww)
        frame = cv2.resize(frame, (int(ww * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ts = t0 + i * 1000.0 / fps
        wait = ts - time.monotonic() * 1000.0
        if wait > 0:
            time.sleep(wait / 1000.0)
        try:
            _, det, evs = eng.process_frame(frame, counts, frame_ts_ms=ts)
        except TypeError:                                   # pre-pred engine (A/B baseline)
            _, det, evs = eng.process_frame(frame, counts)
        for ev in evs:                                      # observed mode only
            events.append((time.monotonic() * 1000.0, dict(ev)))
            print(f"EV seq={ev.get('seq')} id={ev['id']} cls={ev['class']} frame={i}")
            if det.tracker_id is not None:
                idx = [k for k, t in enumerate(det.tracker_id) if int(t) == ev['id']]
                if idx:
                    y = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[idx[0]][1]
                    late_px.append(float(y) - eng._line_coord)
    real = (time.monotonic() * 1000.0 - t0) / 1000.0
    print(f"\n=== {mode}: {i+1} frames in {real:.1f}s ({(i+1)/real:.1f} fps) counts={counts}")
    print(f"    events={len(events)} src={ {k: sum(1 for _, e in events if e.get('src', 'obs') == k) for k in set(e.get('src', 'obs') for _, e in events)} }")
    belt = eng.predictor.belt_speed if eng.predictor else 0.0
    if late_px:
        q = sorted(late_px)
        print(f"    observed lateness px: min={q[0]:.0f} med={statistics.median(q):.0f} p95={q[int(0.95*(len(q)-1))]:.0f} max={q[-1]:.0f}")
    if eng.predictor:
        print(f"    predictor stats: {eng.predictor.stats()}")
    eng.stop()
    return belt, late_px


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--weights', default='isidet/runs/segment/models/yolo/22-06-2026 black/weights/best.pt')
    ap.add_argument('--seconds', type=float, default=120)
    ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--mode', choices=['both', 'observed', 'predictive'], default='both')
    ap.add_argument('--max-side', type=int, default=640, help='downscale long side (site CPU mode = 320)')
    ap.add_argument('--belt', default=BELT, help='belt_direction as in settings.json')
    ap.add_argument('--line', type=float, default=LINE)
    ap.add_argument('--roi-px', default='', help="x1,x2 pixel crop (e.g. 153,563 = site ROI); default = fraction band")
    a = ap.parse_args()
    log_dir = os.environ.get('REPLAY_LOG_DIR', '/tmp/replay_logs')
    belt, late = 0.0, []
    if a.mode in ('both', 'predictive'):
        belt, _ = run('predictive', a, log_dir)
    if a.mode in ('both', 'observed'):
        _, late = run('observed', a, log_dir)
    if late and belt > 0:
        ms = sorted(px / belt for px in late)
        print(f"\n=== observed lateness in ms (belt {belt:.3f} px/ms): min={ms[0]:.0f} med={statistics.median(ms):.0f} "
              f"p95={ms[int(0.95*(len(ms)-1))]:.0f} max={ms[-1]:.0f}")

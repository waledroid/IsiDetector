import sys
import os
import signal
import socket
import secrets
import json
import werkzeug.utils
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import datetime

# After the bucket restructure this file lives at webapp/isitec_app/app.py.
# Python needs two things on sys.path:
#   - webapp/  so `from isitec_app.stream_handler import X` resolves
#   - isidet/  so the ~55 `from src.X import Y` statements throughout the
#              codebase keep working without being rewritten
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', 'isidet')))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))

from isitec_app.stream_handler import StreamHandler
from src.utils.event_logger import EventLogger

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB upload cap
stream_handler = StreamHandler()

# ── Dev-mode authentication ──────────────────────────────────────────────────
DEV_PASSWORD = os.environ.get('DEV_PASSWORD', 'change-me')
_dev_tokens: set[str] = set()

def _check_dev() -> bool:
    """Return True if the request carries a valid dev token."""
    token = request.headers.get('X-Dev-Token', '')
    return token in _dev_tokens

# ── Settings persistence ─────────────────────────────────────────────────────
SETTINGS_PATH = os.path.join(os.path.dirname(__file__), 'settings.json')

def _load_settings() -> dict:
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_settings(data: dict):
    with open(SETTINGS_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def _shutdown(signum, frame):
    """Graceful shutdown on SIGTERM (docker stop) — saves CSV, closes UDP socket."""
    try:
        stream_handler.stop()
    except Exception as e:
        print(f"Shutdown: stop() error: {e}")
    try:
        stream_handler.publisher.close()
    except Exception as e:
        print(f"Shutdown: publisher.close() error: {e}")
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/docs')
@app.route('/docs/<path:path>')
def serve_docs(path='index.html'):
    docs_dir = os.path.join(os.path.dirname(__file__), 'static', 'docs')
    return send_from_directory(docs_dir, path)

@app.route('/video_feed')
def video_feed():
    return Response(stream_handler.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def start_stream():
    data = request.json or {}
    source = data.get('source')
    model_type = data.get('model_type', 'yolo')
    weights = data.get('weights', '')
    imgsz = data.get('imgsz')
    if imgsz: imgsz = int(imgsz)
    conf = data.get('conf')
    if conf: conf = float(conf)

    if model_type not in ('yolo', 'rfdetr'):
        return jsonify({"status": "error", "message": f"Invalid model_type '{model_type}'. Use 'yolo' or 'rfdetr'."}), 400

    success, msg = stream_handler.start(source, model_type, weights, imgsz=imgsz, conf_thresh=conf)
    if success:
        return jsonify({"status": "success", "message": msg})
    return jsonify({"status": "error", "message": msg}), 400

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    
    filename = werkzeug.utils.secure_filename(file.filename)
    filepath = os.path.join(uploads_dir, filename)
    file.save(filepath)
    return jsonify({"status": "success", "filepath": filepath})

@app.route('/api/language', methods=['POST'])
def set_language():
    lang = (request.json or {}).get('language', 'fr')
    if lang not in ('en', 'fr', 'de'):
        return jsonify({"status": "error", "message": f"Unsupported language '{lang}'."}), 400
    stream_handler.set_language(lang)
    return jsonify({"status": "success"})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(stream_handler.get_stats())

@app.route('/api/dev-auth', methods=['POST'])
def dev_auth():
    password = (request.json or {}).get('password', '')
    if password == DEV_PASSWORD:
        token = secrets.token_hex(16)
        _dev_tokens.add(token)
        return jsonify({"status": "success", "token": token})
    return jsonify({"status": "error", "message": "Invalid password"}), 403

@app.route('/api/dev-check', methods=['GET'])
def dev_check():
    if _check_dev():
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 403

@app.route('/api/dev-logout', methods=['POST'])
def dev_logout():
    token = request.headers.get('X-Dev-Token', '')
    _dev_tokens.discard(token)
    return jsonify({"status": "success"})

@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'GET':
        return jsonify({"status": "success", "settings": _load_settings()})
    if not _check_dev():
        return jsonify({"status": "error", "message": "Unauthorized"}), 403
    data = request.json or {}

    # Guard against a known-broken combination before it reaches the
    # inference path. OpenVINO 2026 mistranslates RF-DETR's transformer
    # ops — the IR produces zero detections. Reject the save so the user
    # sees an immediate, actionable error in the Settings panel rather
    # than discovering it after clicking Start.
    rfdetr_w = data.get('rfdetr_weights', '')
    if isinstance(rfdetr_w, str) and rfdetr_w.lower().endswith('.xml'):
        return jsonify({
            "status": "error",
            "message": (
                "RF-DETR is not supported via OpenVINO IR (.xml) — the "
                "conversion produces wrong logits. Pick an RF-DETR .onnx "
                "or .pth file instead."
            ),
        }), 400

    # Range-validate the perf knobs before they hit settings.json.
    # Bad values would only fail later at engine construction, so reject early.
    if 'cpu_threads' in data:
        try:
            n = int(data['cpu_threads'])
            if not (1 <= n <= 64):
                raise ValueError("cpu_threads must be between 1 and 64")
            data['cpu_threads'] = n
        except (ValueError, TypeError) as e:
            return jsonify({"status": "error", "message": str(e)}), 400
    if 'skip_masks' in data:
        data['skip_masks'] = bool(data['skip_masks'])
    if 'skip_traces' in data:
        data['skip_traces'] = bool(data['skip_traces'])

    allowed_keys = (
        'yolo_weights', 'rfdetr_weights', 'yolo_imgsz', 'yolo_conf',
        'detr_imgsz', 'detr_conf', 'line_orientation', 'line_position',
        'belt_direction', 'cpu_threads', 'skip_masks', 'skip_traces',
    )
    current = _load_settings()
    for k in allowed_keys:
        if k in data:
            current[k] = data[k]
    _save_settings(current)
    return jsonify({"status": "success", "settings": current})

@app.route('/api/performance', methods=['GET'])
def get_performance():
    if not _check_dev():
        return jsonify({"status": "error", "message": "Unauthorized"}), 403
    return jsonify(stream_handler.get_performance())

# ── /api/chart helpers ──────────────────────────────────────────────────────
# EventLogger writes one CSV row per line-crossing to
# isidet/logs/events/events_YYYY-MM-DD.csv (columns: ts, class, id).
# For the historical chart we just count events per bucket — no cumulative
# accounting, no midnight resets to reason about. The logger also prunes
# itself to the last 30 days so this dir never grows unbounded.

def _events_dir():
    """isidet/logs/events/ resolved from this file's location
    (webapp/isitec_app/app.py → parents[2] = repo root)."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    return os.path.join(repo_root, 'isidet', 'logs', 'events')


@app.route('/api/chart', methods=['GET'])
def get_chart_data():
    period = request.args.get('period', 'live')

    # Live mode is the running session's counts — shape kept backward-
    # compatible so any legacy consumer still works.
    if period == 'live':
        stats = stream_handler.get_stats()
        return jsonify({"status": "success", "data": stats['counts']})

    now = datetime.datetime.now()
    if period == '24h':
        # End at the NEXT hour boundary so the current partial hour is the
        # rightmost bucket and fills in real time as events arrive. Buckets
        # stay clock-aligned; labels are hour-of-day only.
        window_end = now.replace(minute=0, second=0, microsecond=0) \
                     + datetime.timedelta(hours=1)
        window_start = window_end - datetime.timedelta(hours=24)
        bucket_size = datetime.timedelta(hours=1)
        n_buckets = 24
        label_fmt = "%H:00"
    elif period == '7d':
        today = now.date()
        start_date = today - datetime.timedelta(days=6)
        window_start = datetime.datetime.combine(start_date, datetime.time.min)
        window_end = datetime.datetime.combine(today + datetime.timedelta(days=1),
                                               datetime.time.min)
        bucket_size = datetime.timedelta(days=1)
        n_buckets = 7
        label_fmt = "%a %d"
    elif period == '30d':
        today = now.date()
        start_date = today - datetime.timedelta(days=29)
        window_start = datetime.datetime.combine(start_date, datetime.time.min)
        window_end = datetime.datetime.combine(today + datetime.timedelta(days=1),
                                               datetime.time.min)
        bucket_size = datetime.timedelta(days=1)
        n_buckets = 30
        label_fmt = "%b %d"
    else:
        return jsonify({"status": "success", "view": "timeseries",
                        "buckets": [], "series": {}})

    bucket_secs = bucket_size.total_seconds()
    series: dict[str, list[int]] = {}
    for ts, cls, _eid in EventLogger.read_events(_events_dir(), window_start, window_end):
        idx = int((ts - window_start).total_seconds() // bucket_secs)
        if 0 <= idx < n_buckets:
            if cls not in series:
                series[cls] = [0] * n_buckets
            series[cls][idx] += 1

    # Always surface the stock classes so the chart legend stays stable even
    # when a bucket window has zero events for one of them.
    for c in ('carton', 'polybag'):
        series.setdefault(c, [0] * n_buckets)

    buckets = [
        (window_start + i * bucket_size).strftime(label_fmt)
        for i in range(n_buckets)
    ]

    return jsonify({
        "status": "success",
        "view": "timeseries",
        "buckets": buckets,
        "series": series,
    })


# ── /api/report helpers ─────────────────────────────────────────────────────
# Production-summary aggregates powering the Analytics page. Events come
# from the per-crossing log (isidet/logs/events/); session-level metrics
# like FPS and runtime come from webapp/isitec_app/logs/sessions.json,
# written at stream stop by PerformanceMonitor.save_session_summary.

_SESSIONS_PATH = os.path.join(os.path.dirname(__file__), 'logs', 'sessions.json')


def _resolve_report_window(period: str, from_s: str | None, to_s: str | None):
    """Return (window_start, window_end) for the requested period.

    ``period`` is one of 'today', 'yesterday', '7d', '30d', 'custom'.
    For 'custom' both ``from_s`` and ``to_s`` (YYYY-MM-DD) must be given;
    the window is inclusive of both dates (to_end = to_date + 1 day).
    """
    now = datetime.datetime.now()
    today = now.date()
    if period == 'today':
        start = datetime.datetime.combine(today, datetime.time.min)
        end = start + datetime.timedelta(days=1)
    elif period == 'yesterday':
        start = datetime.datetime.combine(today - datetime.timedelta(days=1), datetime.time.min)
        end = start + datetime.timedelta(days=1)
    elif period == '7d':
        start = datetime.datetime.combine(today - datetime.timedelta(days=6), datetime.time.min)
        end = datetime.datetime.combine(today + datetime.timedelta(days=1), datetime.time.min)
    elif period == '30d':
        start = datetime.datetime.combine(today - datetime.timedelta(days=29), datetime.time.min)
        end = datetime.datetime.combine(today + datetime.timedelta(days=1), datetime.time.min)
    elif period == 'custom' and from_s and to_s:
        start = datetime.datetime.strptime(from_s, '%Y-%m-%d')
        end = datetime.datetime.strptime(to_s, '%Y-%m-%d') + datetime.timedelta(days=1)
    else:
        raise ValueError(f"invalid period/range: {period!r}")
    return start, end


def _aggregate_events(window_start, window_end):
    """Walk the event log for the window and return counts, peak hour, mix."""
    counts: dict[str, int] = {}
    by_hour: dict[str, int] = {}
    for ts, cls, _eid in EventLogger.read_events(_events_dir(), window_start, window_end):
        counts[cls] = counts.get(cls, 0) + 1
        hour_key = ts.strftime('%Y-%m-%d %H:00')
        by_hour[hour_key] = by_hour.get(hour_key, 0) + 1

    for c in ('carton', 'polybag'):
        counts.setdefault(c, 0)
    total = sum(counts.values())
    mix_pct = {
        c: (round(100.0 * n / total, 1) if total > 0 else 0.0)
        for c, n in counts.items()
    }
    if by_hour:
        peak_bucket, peak_events = max(by_hour.items(), key=lambda kv: kv[1])
        # Render peak_bucket as HH:00 on date (human-friendly label handled client-side)
        peak = {"hour": peak_bucket, "events": peak_events}
    else:
        peak = {"hour": None, "events": 0}
    return counts, total, mix_pct, peak


def _aggregate_sessions(window_start, window_end, sessions_path: str):
    """Read sessions.json and aggregate FPS (duration-weighted) + runtime.

    Session ``date`` field is stored as ``"DD-MM-YYYY HH:MM"`` (save time,
    near end-of-session). A session falls in the window if its save time
    is within ``[window_start, window_end)``.
    """
    try:
        if not os.path.exists(sessions_path):
            return {"avg_fps": None, "total_runtime_h": 0.0, "sessions_count": 0,
                    "recent": []}
        with open(sessions_path) as f:
            sessions = json.load(f)
    except Exception:
        return {"avg_fps": None, "total_runtime_h": 0.0, "sessions_count": 0,
                "recent": []}

    weighted_fps = 0.0
    total_duration = 0.0
    count = 0
    for s in sessions:
        try:
            ts = datetime.datetime.strptime(s.get('date', ''), '%d-%m-%Y %H:%M')
        except ValueError:
            continue
        if ts < window_start or ts >= window_end:
            continue
        count += 1
        dur = float(s.get('duration_h') or 0)
        total_duration += dur
        fps = s.get('fps')
        if fps is not None and dur > 0:
            weighted_fps += float(fps) * dur

    avg_fps = round(weighted_fps / total_duration, 1) if total_duration > 0 else None
    return {
        "avg_fps": avg_fps,
        "total_runtime_h": round(total_duration, 2),
        "sessions_count": count,
        "recent": sessions[:5],
    }


@app.route('/api/report', methods=['GET'])
def get_report():
    period = request.args.get('period', 'today')
    from_s = request.args.get('from')
    to_s = request.args.get('to')
    try:
        window_start, window_end = _resolve_report_window(period, from_s, to_s)
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    counts, total, mix_pct, peak = _aggregate_events(window_start, window_end)
    sessions_agg = _aggregate_sessions(window_start, window_end, _SESSIONS_PATH)

    window_hours = (window_end - window_start).total_seconds() / 3600.0
    throughput_per_hour = (
        round(total / window_hours, 1) if window_hours > 0 else 0.0
    )

    return jsonify({
        "status": "success",
        "period": period,
        "window": {
            "from": window_start.isoformat(),
            "to": window_end.isoformat(),
        },
        "counts": {**counts, "total": total},
        "mix_pct": mix_pct,
        "peak": peak,
        "throughput_per_hour": throughput_per_hour,
        "avg_fps": sessions_agg["avg_fps"],
        "total_runtime_h": sessions_agg["total_runtime_h"],
        "sessions_count": sessions_agg["sessions_count"],
        "recent_sessions": sessions_agg["recent"],
    })


@app.route('/api/events/export', methods=['GET'])
def export_events():
    """Stream the event log as a single CSV for the given date range."""
    from_s = request.args.get('from')
    to_s = request.args.get('to')
    if not (from_s and to_s):
        return jsonify({"status": "error",
                        "message": "from and to (YYYY-MM-DD) required"}), 400
    try:
        window_start = datetime.datetime.strptime(from_s, '%Y-%m-%d')
        window_end = datetime.datetime.strptime(to_s, '%Y-%m-%d') \
                     + datetime.timedelta(days=1)
    except ValueError:
        return jsonify({"status": "error", "message": "bad date format"}), 400

    def _generate():
        yield "ts,class,id\n"
        for ts, cls, eid in EventLogger.read_events(_events_dir(),
                                                    window_start, window_end):
            yield f"{ts.isoformat()},{cls},{eid if eid is not None else ''}\n"

    filename = f"events_{from_s}_to_{to_s}.csv"
    return Response(
        _generate(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


@app.route('/api/models', methods=['GET'])
def get_models():
    # After the bucket restructure this file lives at webapp/isitec_app/app.py,
    # so dirname×3 gets us to the repo root. Model artefacts live exclusively
    # under isidet/, so scan only those subtrees — walking the whole repo
    # would cross into mkdocs/site, webapp/, deploy/, etc. for no benefit.
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    isidet_root = os.path.join(project_root, 'isidet')
    yolo_models = []
    rfdetr_models = []

    scan_dirs = [
        os.path.join(isidet_root, 'models'),
        os.path.join(isidet_root, 'runs'),
    ]
    scanned_paths = set()

    all_exts = ('.pt', '.pth', '.onnx', '.xml', '.engine')

    for directory in scan_dirs:
        if not os.path.exists(directory) or not os.path.isdir(directory):
            continue
        for root_dir, _, files in os.walk(directory):
            if '.git' in root_dir or '__pycache__' in root_dir or 'node_modules' in root_dir:
                continue
            for file in files:
                if not file.endswith(all_exts):
                    continue
                abs_path = os.path.join(root_dir, file)
                if abs_path in scanned_paths:
                    continue
                scanned_paths.add(abs_path)

                try:
                    rel_path = os.path.relpath(abs_path, project_root)
                except ValueError:
                    rel_path = abs_path

                entry = {"name": file, "path": rel_path}

                # Skip non-model XML files (e.g. sitemap.xml)
                if file.endswith('.xml') and 'openvino' not in root_dir.lower():
                    continue

                # Classify by directory path + filename context
                path_lower = rel_path.lower()
                name_lower = file.lower()
                is_rfdetr = 'rfdetr' in path_lower or 'detr' in path_lower or 'rf-detr' in name_lower
                is_yolo = 'yolo' in path_lower

                if file.endswith('.pth'):
                    rfdetr_models.append(entry)
                elif file.endswith('.pt'):
                    # .pt files with "detr" in name are RF-DETR pretrained weights
                    if is_rfdetr:
                        rfdetr_models.append(entry)
                    else:
                        yolo_models.append(entry)
                elif is_rfdetr:
                    rfdetr_models.append(entry)
                elif is_yolo:
                    yolo_models.append(entry)
                else:
                    # Unknown context — add to both so user can choose
                    yolo_models.append(entry)
                    rfdetr_models.append(entry)

    yolo_models = sorted(yolo_models, key=lambda x: x["name"])
    rfdetr_models = sorted(rfdetr_models, key=lambda x: x["name"])

    return jsonify({
        "status": "success",
        "yolo_models": yolo_models,
        "rfdetr_models": rfdetr_models
    })

@app.route('/api/udp', methods=['GET', 'POST'])
def udp_target():
    if not _check_dev():
        return jsonify({"status": "error", "message": "Unauthorized"}), 403
    if request.method == 'GET':
        return jsonify(stream_handler.get_udp_target())
    data = request.json or {}
    host = data.get('host', '').strip()
    port = data.get('port')
    if not host:
        return jsonify({"status": "error", "message": "host required"}), 400
    try:
        port = int(port)
        if not (1 <= port <= 65535):
            raise ValueError()
        socket.inet_aton(host)  # validates IPv4; raises error for invalid IP
    except (TypeError, ValueError, OSError):
        return jsonify({"status": "error", "message": "Invalid host or port (port must be 1-65535, host must be valid IP)"}), 400
    stream_handler.set_udp_target(host, port)
    return jsonify({"status": "success", "host": host, "port": port})

@app.route('/api/line', methods=['GET', 'POST'])
def line_config():
    if not _check_dev():
        return jsonify({"status": "error", "message": "Unauthorized"}), 403
    if request.method == 'GET':
        return jsonify(stream_handler.get_line_config())
    data = request.json or {}
    orientation = data.get('orientation')
    position = data.get('position')
    belt_direction = data.get('belt_direction')
    if orientation and orientation not in ('vertical', 'horizontal'):
        return jsonify({"status": "error", "message": "orientation must be 'vertical' or 'horizontal'"}), 400
    if position is not None:
        position = float(position)
        if not (0.1 <= position <= 0.9):
            return jsonify({"status": "error", "message": "position must be 0.1-0.9"}), 400
    valid_directions = ('left_to_right', 'right_to_left', 'top_to_bottom', 'bottom_to_top')
    if belt_direction and belt_direction not in valid_directions:
        return jsonify({"status": "error", "message": f"belt_direction must be one of {valid_directions}"}), 400
    stream_handler.set_line_config(orientation=orientation, position=position, belt_direction=belt_direction)
    current = _load_settings()
    config = stream_handler.get_line_config()
    current['line_orientation'] = config['orientation']
    current['line_position'] = config['position']
    current['belt_direction'] = config['belt_direction']
    _save_settings(current)
    return jsonify({"status": "success", **config})

@app.route('/api/belt_status', methods=['POST'])
def set_belt_status():
    if not _check_dev():
        return jsonify({"status": "error", "message": "Unauthorized"}), 403
    data = request.json or {}
    status = data.get('status')
    if status not in ('active', 'paused'):
        return jsonify({"status": "error", "message": "status must be 'active' or 'paused'"}), 400
    
    belt_active = (status == 'active')
    stream_handler.set_belt_status(belt_active)
    return jsonify({"status": "success", "belt_active": belt_active})

@app.route('/api/stop', methods=['POST'])
def stop_stream():
    success, msg = stream_handler.stop()
    if success:
        return jsonify({"status": "success", "message": msg})
    return jsonify({"status": "error", "message": msg}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9505))
    try:
        print(f"Starting Flask Server on http://0.0.0.0:{port}")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except Exception as e:
        print(f"Flask Critical Error: {e}")
    finally:
        print("Flask Process Exiting...")

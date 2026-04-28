import sys
import os
import socket
import json
import re
import datetime
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Depends, Header, Query
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# After the bucket restructure this file lives at webapp/isitec_api/app.py.
# Python needs two things on sys.path:
#   - webapp/  so `from isitec_api.stream_handler import X` resolves
#   - isidet/  so the ~55 `from src.X import Y` statements throughout the
#              codebase keep working without being rewritten
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', 'isidet')))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))

from isitec_api.stream_handler import StreamHandler
from isitec_api.dependencies import (
    DEV_PASSWORD, require_dev, create_dev_token, discard_dev_token, check_dev_token,
)
from src.utils.event_logger import EventLogger

# ── Globals ──────────────────────────────────────────────────────────────────
stream_handler: StreamHandler | None = None

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

# ── secure_filename (replaces werkzeug) ───────────────────────────────────────
def secure_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r'[^\w\s\-.]', '', name).strip()
    name = re.sub(r'[\s]+', '_', name)
    return name or 'upload'

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global stream_handler
    stream_handler = StreamHandler()
    yield
    # Shutdown
    try:
        stream_handler.stop()
    except Exception:
        pass
    try:
        stream_handler.publisher.close()
    except Exception:
        pass

app = FastAPI(
    title="ISITEC visionAI API",
    docs_url="/swagger",
    lifespan=lifespan,
)

# Mount static files
_base_dir = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(_base_dir, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(_base_dir, "templates"))

# ── Import and register WebSocket routes ──────────────────────────────────────
from isitec_api.websockets import router as ws_router
app.include_router(ws_router)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/video_feed")
def video_feed():
    """MJPEG fallback — kept for backward compatibility. Prefer /ws/video."""
    return StreamingResponse(
        stream_handler.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# FastAPI's built-in ``/swagger`` only exposes the OpenAPI interactive UI;
# operators also need the static MkDocs-built site mounted at ``/docs``.
# Match Flask's behaviour: ``/docs`` → index.html, ``/docs/<sub/path>`` →
# the corresponding file inside webapp/isitec_api/static/docs/.
_docs_dir = os.path.join(_base_dir, "static", "docs")


@app.get("/docs", response_class=HTMLResponse)
@app.get("/docs/", response_class=HTMLResponse)
async def serve_docs_index():
    index_path = os.path.join(_docs_dir, "index.html")
    if not os.path.isfile(index_path):
        return HTMLResponse(
            "<h1>Docs not built yet</h1>"
            "<p>Run <code>cd mkdocs && mkdocs build</code> "
            "and restart the container.</p>",
            status_code=404,
        )
    with open(index_path, encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


@app.get("/docs/{subpath:path}")
async def serve_docs_subpath(subpath: str):
    # Strip any leading slash or drive-like prefix; reject absolute paths
    # and directory-escape attempts before touching the filesystem.
    safe = subpath.lstrip("/").lstrip("\\")
    if ".." in safe.split("/"):
        return JSONResponse({"detail": "invalid path"}, status_code=400)
    target = os.path.join(_docs_dir, safe) if safe else os.path.join(_docs_dir, "index.html")
    # Directory request → index.html inside that directory.
    if os.path.isdir(target):
        target = os.path.join(target, "index.html")
    if not os.path.isfile(target):
        return JSONResponse({"detail": "not found"}, status_code=404)
    return FileResponse(target)


@app.post("/api/start")
def start_stream(request_body: dict):
    source = request_body.get('source')
    model_type = request_body.get('model_type', 'yolo')
    weights = request_body.get('weights', '')
    imgsz = request_body.get('imgsz')
    if imgsz:
        imgsz = int(imgsz)
    conf = request_body.get('conf')
    if conf:
        conf = float(conf)

    if model_type not in ('yolo', 'rfdetr'):
        return JSONResponse(
            {"status": "error", "message": f"Invalid model_type '{model_type}'. Use 'yolo' or 'rfdetr'."},
            status_code=400,
        )

    success, msg = stream_handler.start(source, model_type, weights, imgsz=imgsz, conf_thresh=conf)
    if success:
        return {"status": "success", "message": msg}
    return JSONResponse({"status": "error", "message": msg}, status_code=400)


@app.post("/api/stop")
def stop_stream():
    success, msg = stream_handler.stop()
    if success:
        return {"status": "success", "message": msg}
    return JSONResponse({"status": "error", "message": msg}, status_code=400)


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse({"status": "error", "message": "No selected file"}, status_code=400)

    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    filename = secure_filename(file.filename)
    filepath = os.path.join(uploads_dir, filename)

    with open(filepath, 'wb') as f:
        content = await file.read()
        f.write(content)

    return {"status": "success", "filepath": filepath}


@app.post("/api/language")
async def set_language(request_body: dict):
    lang = request_body.get('language', 'fr')
    if lang not in ('en', 'fr', 'de'):
        return JSONResponse(
            {"status": "error", "message": f"Unsupported language '{lang}'."},
            status_code=400,
        )
    stream_handler.set_language(lang)
    return {"status": "success"}


@app.get("/api/stats")
def get_stats():
    return stream_handler.get_stats()


@app.post("/api/dev-auth")
async def dev_auth(request_body: dict):
    password = request_body.get('password', '')
    if password == DEV_PASSWORD:
        token = create_dev_token()
        return {"status": "success", "token": token}
    return JSONResponse({"status": "error", "message": "Invalid password"}, status_code=403)


@app.get("/api/dev-check")
async def dev_check(_token: str = Depends(require_dev)):
    return {"status": "success"}


@app.post("/api/dev-logout")
async def dev_logout(x_dev_token: str = Header("", alias="X-Dev-Token")):
    discard_dev_token(x_dev_token)
    return {"status": "success"}


@app.get("/api/settings")
async def get_settings():
    return {"status": "success", "settings": _load_settings()}


@app.post("/api/settings")
async def save_settings(request_body: dict, _token: str = Depends(require_dev)):
    # Guard against a known-broken combination before it reaches the
    # inference path. OpenVINO 2026 mistranslates RF-DETR's transformer
    # ops — the IR produces zero detections. Reject the save so the user
    # sees an immediate, actionable error in the Settings panel rather
    # than discovering it after clicking Start.
    rfdetr_w = request_body.get('rfdetr_weights', '')
    if isinstance(rfdetr_w, str) and rfdetr_w.lower().endswith('.xml'):
        return JSONResponse(
            {
                "status": "error",
                "message": (
                    "RF-DETR is not supported via OpenVINO IR (.xml) — the "
                    "conversion produces wrong logits. Pick an RF-DETR .onnx "
                    "or .pth file instead."
                ),
            },
            status_code=400,
        )

    # Range-validate the perf knobs before they hit settings.json.
    if 'cpu_threads' in request_body:
        try:
            n = int(request_body['cpu_threads'])
            if not (1 <= n <= 64):
                raise ValueError("cpu_threads must be between 1 and 64")
            request_body['cpu_threads'] = n
        except (ValueError, TypeError) as e:
            return JSONResponse(
                {"status": "error", "message": str(e)}, status_code=400
            )
    if 'skip_masks' in request_body:
        request_body['skip_masks'] = bool(request_body['skip_masks'])
    if 'skip_traces' in request_body:
        request_body['skip_traces'] = bool(request_body['skip_traces'])

    allowed_keys = (
        'yolo_weights', 'rfdetr_weights', 'yolo_imgsz', 'yolo_conf',
        'detr_imgsz', 'detr_conf', 'line_orientation', 'line_position',
        'belt_direction', 'cpu_threads', 'skip_masks', 'skip_traces',
    )
    current = _load_settings()
    for k in allowed_keys:
        if k in request_body:
            current[k] = request_body[k]
    _save_settings(current)
    return {"status": "success", "settings": current}


@app.get("/api/performance")
def get_performance(_token: str = Depends(require_dev)):
    return stream_handler.get_performance()


# ── /api/chart helpers (mirrored from isitec_app/app.py — keep in sync) ─────
# EventLogger writes one CSV row per line-crossing to
# isidet/logs/events/events_YYYY-MM-DD.csv (columns: ts, class, id). The
# chart endpoint just counts events per bucket; logger self-prunes to
# the last 30 days on init and rollover.

def _events_dir():
    """isidet/logs/events/ resolved from this file's location
    (webapp/isitec_api/app.py → parents[2] = repo root)."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    return os.path.join(repo_root, 'isidet', 'logs', 'events')


@app.get("/api/chart")
def get_chart_data(period: str = Query("live")):
    if period == 'live':
        stats = stream_handler.get_stats()
        return {"status": "success", "data": stats['counts']}

    now = datetime.datetime.now()
    if period == '24h':
        # End at the NEXT hour boundary so the current partial hour is the
        # rightmost bucket and fills in real time as events arrive.
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
        return {"status": "success", "view": "timeseries",
                "buckets": [], "series": {}}

    bucket_secs = bucket_size.total_seconds()
    series: dict[str, list[int]] = {}
    for ts, cls, _eid in EventLogger.read_events(_events_dir(), window_start, window_end):
        idx = int((ts - window_start).total_seconds() // bucket_secs)
        if 0 <= idx < n_buckets:
            if cls not in series:
                series[cls] = [0] * n_buckets
            series[cls][idx] += 1

    for c in ('carton', 'polybag'):
        series.setdefault(c, [0] * n_buckets)

    buckets = [
        (window_start + i * bucket_size).strftime(label_fmt)
        for i in range(n_buckets)
    ]

    return {
        "status": "success",
        "view": "timeseries",
        "buckets": buckets,
        "series": series,
    }


# ── /api/report helpers (mirrored from isitec_app/app.py — keep in sync) ────

_SESSIONS_PATH = os.path.join(os.path.dirname(__file__), 'logs', 'sessions.json')


def _resolve_report_window(period: str, from_s: str | None, to_s: str | None):
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
        peak = {"hour": peak_bucket, "events": peak_events}
    else:
        peak = {"hour": None, "events": 0}
    return counts, total, mix_pct, peak


def _aggregate_sessions(window_start, window_end, sessions_path: str):
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


@app.get("/api/report")
def get_report(period: str = Query("today"),
               from_: str | None = Query(None, alias="from"),
               to: str | None = Query(None)):
    try:
        window_start, window_end = _resolve_report_window(period, from_, to)
    except ValueError as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

    counts, total, mix_pct, peak = _aggregate_events(window_start, window_end)
    sessions_agg = _aggregate_sessions(window_start, window_end, _SESSIONS_PATH)

    window_hours = (window_end - window_start).total_seconds() / 3600.0
    throughput_per_hour = (
        round(total / window_hours, 1) if window_hours > 0 else 0.0
    )

    return {
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
    }


@app.get("/api/events/export")
def export_events(from_: str | None = Query(None, alias="from"),
                  to: str | None = Query(None)):
    if not (from_ and to):
        return JSONResponse({"status": "error",
                             "message": "from and to (YYYY-MM-DD) required"},
                            status_code=400)
    try:
        window_start = datetime.datetime.strptime(from_, '%Y-%m-%d')
        window_end = datetime.datetime.strptime(to, '%Y-%m-%d') \
                     + datetime.timedelta(days=1)
    except ValueError:
        return JSONResponse({"status": "error", "message": "bad date format"},
                            status_code=400)

    def _generate():
        yield "ts,class,id\n"
        for ts, cls, eid in EventLogger.read_events(_events_dir(),
                                                    window_start, window_end):
            yield f"{ts.isoformat()},{cls},{eid if eid is not None else ''}\n"

    filename = f"events_{from_}_to_{to}.csv"
    return StreamingResponse(
        _generate(),
        media_type='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


@app.get("/api/models")
def get_models():
    # After the bucket restructure this file lives at webapp/isitec_api/app.py,
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

                if file.endswith('.xml') and 'openvino' not in root_dir.lower():
                    continue

                path_lower = rel_path.lower()
                is_rfdetr = 'rfdetr' in path_lower or 'detr' in path_lower or 'rf-detr' in file.lower()
                is_yolo = 'yolo' in path_lower

                if file.endswith('.pth'):
                    rfdetr_models.append(entry)
                elif file.endswith('.pt'):
                    if is_rfdetr:
                        rfdetr_models.append(entry)
                    else:
                        yolo_models.append(entry)
                elif is_rfdetr:
                    rfdetr_models.append(entry)
                elif is_yolo:
                    yolo_models.append(entry)
                else:
                    yolo_models.append(entry)
                    rfdetr_models.append(entry)

    yolo_models = sorted(yolo_models, key=lambda x: x["name"])
    rfdetr_models = sorted(rfdetr_models, key=lambda x: x["name"])

    return {
        "status": "success",
        "yolo_models": yolo_models,
        "rfdetr_models": rfdetr_models,
    }


@app.get("/api/udp")
def get_udp_target(_token: str = Depends(require_dev)):
    return stream_handler.get_udp_target()


@app.post("/api/udp")
def set_udp_target(request_body: dict, _token: str = Depends(require_dev)):
    host = request_body.get('host', '').strip()
    port = request_body.get('port')
    if not host:
        return JSONResponse({"status": "error", "message": "host required"}, status_code=400)
    try:
        port = int(port)
        if not (1 <= port <= 65535):
            raise ValueError()
        socket.inet_aton(host)
    except (TypeError, ValueError, OSError):
        return JSONResponse(
            {"status": "error", "message": "Invalid host or port (port must be 1-65535, host must be valid IP)"},
            status_code=400,
        )
    stream_handler.set_udp_target(host, port)
    return {"status": "success", "host": host, "port": port}


@app.get("/api/line")
def get_line(_token: str = Depends(require_dev)):
    return stream_handler.get_line_config()


@app.post("/api/line")
def set_line(request_body: dict, _token: str = Depends(require_dev)):
    orientation = request_body.get('orientation')
    position = request_body.get('position')
    belt_direction = request_body.get('belt_direction')
    if orientation and orientation not in ('vertical', 'horizontal'):
        return JSONResponse(
            {"status": "error", "message": "orientation must be 'vertical' or 'horizontal'"},
            status_code=400,
        )
    if position is not None:
        position = float(position)
        if not (0.1 <= position <= 0.9):
            return JSONResponse(
                {"status": "error", "message": "position must be 0.1-0.9"},
                status_code=400,
            )
    valid_directions = ('left_to_right', 'right_to_left', 'top_to_bottom', 'bottom_to_top')
    if belt_direction and belt_direction not in valid_directions:
        return JSONResponse(
            {"status": "error", "message": f"belt_direction must be one of {valid_directions}"},
            status_code=400,
        )
    stream_handler.set_line_config(orientation=orientation, position=position, belt_direction=belt_direction)
    current = _load_settings()
    config = stream_handler.get_line_config()
    current['line_orientation'] = config['orientation']
    current['line_position'] = config['position']
    current['belt_direction'] = config['belt_direction']
    _save_settings(current)
    return {"status": "success", **config}


@app.post("/api/belt_status")
def set_belt_status(request_body: dict, _token: str = Depends(require_dev)):
    status = request_body.get('status')
    if status not in ('active', 'paused'):
        return JSONResponse(
            {"status": "error", "message": "status must be 'active' or 'paused'"},
            status_code=400,
        )
    belt_active = (status == 'active')
    stream_handler.set_belt_status(belt_active)
    return {"status": "success", "belt_active": belt_active}


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 9501))
    print(f"Starting FastAPI Server on http://0.0.0.0:{port}")
    uvicorn.run(app, host='0.0.0.0', port=port)

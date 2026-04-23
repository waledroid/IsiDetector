import sys
import os
import socket
import json
import re
import csv
import datetime
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Depends, Header, Query
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add the parent directory (~/logistic) to Python's path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isitec_api.stream_handler import StreamHandler
from isitec_api.dependencies import (
    DEV_PASSWORD, require_dev, create_dev_token, discard_dev_token, check_dev_token,
)

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
    allowed_keys = ('yolo_weights', 'rfdetr_weights', 'yolo_imgsz', 'yolo_conf', 'detr_imgsz', 'detr_conf', 'line_orientation', 'line_position', 'belt_direction')
    current = _load_settings()
    for k in allowed_keys:
        if k in request_body:
            current[k] = request_body[k]
    _save_settings(current)
    return {"status": "success", "settings": current}


@app.get("/api/performance")
def get_performance(_token: str = Depends(require_dev)):
    return stream_handler.get_performance()


@app.get("/api/chart")
def get_chart_data(period: str = Query("live")):
    if period == 'live':
        stats = stream_handler.get_stats()
        return {"status": "success", "data": stats['counts']}

    now = datetime.datetime.now()
    if period == '24h':
        cutoff = now - datetime.timedelta(days=1)
    elif period == '7d':
        cutoff = now - datetime.timedelta(days=7)
    elif period == '30d':
        cutoff = now - datetime.timedelta(days=30)
    else:
        cutoff = now - datetime.timedelta(days=1)

    counts = {}
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    if not os.path.exists(logs_dir):
        return {"status": "success", "data": counts}

    for filename in os.listdir(logs_dir):
        if not filename.endswith('.csv'):
            continue
        try:
            date_part = filename.replace('report_', '').replace('.csv', '')
            file_date = datetime.datetime.strptime(date_part, '%d-%m-%Y')
            if file_date.date() < cutoff.date():
                continue
        except ValueError:
            pass

        filepath = os.path.join(logs_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        ts_str = row[0]
                        class_name = row[1]
                        try:
                            row_ts = datetime.datetime.fromisoformat(ts_str)
                            if row_ts >= cutoff:
                                counts[class_name] = counts.get(class_name, 0) + 1
                        except ValueError:
                            pass
        except Exception:
            pass

    return {"status": "success", "data": counts}


@app.get("/api/models")
def get_models():
    project_root = os.path.dirname(os.path.dirname(__file__))
    yolo_models = []
    rfdetr_models = []

    scan_dirs = [project_root, os.path.join(project_root, 'models'), os.path.join(project_root, 'runs')]
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

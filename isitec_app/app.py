import sys
import os
import signal
import socket
import secrets
import json
import werkzeug.utils
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import csv
import datetime

# Add the parent directory (~/logistic) to Python's path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isitec_app.stream_handler import StreamHandler

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

    allowed_keys = ('yolo_weights', 'rfdetr_weights', 'yolo_imgsz', 'yolo_conf', 'detr_imgsz', 'detr_conf', 'line_orientation', 'line_position', 'belt_direction')
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

@app.route('/api/chart', methods=['GET'])
def get_chart_data():
    period = request.args.get('period', 'live')
    
    if period == 'live':
        stats = stream_handler.get_stats()
        return jsonify({"status": "success", "data": stats['counts']})
        
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
        return jsonify({"status": "success", "data": counts})

    for filename in os.listdir(logs_dir):
        if not filename.endswith('.csv'):
            continue
        # Log filenames are report_DD-MM-YYYY.csv — skip files outside the period
        # without opening them (avoids reading the entire logs directory for short periods)
        try:
            date_part = filename.replace('report_', '').replace('.csv', '')
            file_date = datetime.datetime.strptime(date_part, '%d-%m-%Y')
            if file_date.date() < cutoff.date():
                continue
        except ValueError:
            pass  # unexpected filename format — read it anyway

        filepath = os.path.join(logs_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
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
            pass  # Skip corrupted logs safely
            
    return jsonify({"status": "success", "data": counts})

@app.route('/api/models', methods=['GET'])
def get_models():
    project_root = os.path.dirname(os.path.dirname(__file__))
    yolo_models = []
    rfdetr_models = []

    # Directories to scan
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

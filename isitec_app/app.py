import sys
import os
import signal
import socket
import werkzeug.utils
from flask import Flask, render_template, Response, request, jsonify
import csv
import datetime

# Add the parent directory (~/logistic) to Python's path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isitec_app.stream_handler import StreamHandler

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB upload cap
stream_handler = StreamHandler()

def _shutdown(signum, frame):
    """Graceful shutdown on SIGTERM (docker stop) — saves CSV, closes UDP socket."""
    stream_handler.stop()
    stream_handler.publisher.close()
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)

@app.route('/')
def index():
    return render_template('index.html')

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
    lang = (request.json or {}).get('language', 'en')
    if lang not in ('en', 'fr', 'de'):
        return jsonify({"status": "error", "message": f"Unsupported language '{lang}'."}), 400
    stream_handler.set_language(lang)
    return jsonify({"status": "success"})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(stream_handler.get_stats())

@app.route('/api/performance', methods=['GET'])
def get_performance():
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
    
    for directory in scan_dirs:
        if not os.path.exists(directory) or not os.path.isdir(directory):
            continue
        for root_dir, _, files in os.walk(directory):
            # Skip irrelevant large dirs to ensure speed
            if '.git' in root_dir or '__pycache__' in root_dir or 'node_modules' in root_dir:
                continue
            for file in files:
                if file.endswith('.pt') or file.endswith('.pth') or file.endswith('.onnx'):
                    abs_path = os.path.join(root_dir, file)
                    if abs_path in scanned_paths:
                        continue
                    scanned_paths.add(abs_path)
                    
                    # Store path relative to project root for cleaner display
                    try:
                        rel_path = os.path.relpath(abs_path, project_root)
                    except ValueError:
                        rel_path = abs_path
                        
                    entry = {"name": file, "path": rel_path}
                    
                    # YOLO typically uses .pt, RF-DETR uses .pth or .onnx
                    if file.endswith('.pt'):
                        yolo_models.append(entry)
                    elif file.endswith('.pth') or file.endswith('.onnx'):
                        rfdetr_models.append(entry)
                        
    # Sort alphabetically by filename
    yolo_models = sorted(yolo_models, key=lambda x: x["name"])
    rfdetr_models = sorted(rfdetr_models, key=lambda x: x["name"])
    
    return jsonify({
        "status": "success",
        "yolo_models": yolo_models,
        "rfdetr_models": rfdetr_models
    })

@app.route('/api/udp', methods=['GET', 'POST'])
def udp_target():
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

@app.route('/api/stop', methods=['POST'])
def stop_stream():
    success, msg = stream_handler.stop()
    if success:
        return jsonify({"status": "success", "message": msg})
    return jsonify({"status": "error", "message": msg}), 400

if __name__ == '__main__':
    # Using 0.0.0.0 to allow access from any IP, useful when containerized.
    # debug=False ensures the app is more stable and won't restart on background errors.
    try:
        print("🚀 Starting Flask Server on http://0.0.0.0:9501")
        app.run(host='0.0.0.0', port=9501, debug=False, threaded=True)
    except Exception as e:
        print(f"🔥 Flask Critical Error: {e}")
    finally:
        print("💀 Flask Process Exiting...")

import sys
import os
import werkzeug.utils
from flask import Flask, render_template, Response, request, jsonify
import csv
import datetime

# Add the parent directory (~/logistic) to Python's path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isitec_app.stream_handler import StreamHandler

app = Flask(__name__)
stream_handler = StreamHandler()

import threading
import time

def heartbeat():
    while True:
        # print("💓 Flask Heartbeat: Server component is active")
        time.sleep(10)

threading.Thread(target=heartbeat, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(stream_handler.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def start_stream():
    data = request.json
    source = data.get('source')
    model_type = data.get('model_type', 'yolo') # yolo or rfdetr
    weights = data.get('weights', '')
    
    success, msg = stream_handler.start(source, model_type, weights)
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
    lang = request.json.get('language', 'en')
    stream_handler.set_language(lang)
    return jsonify({"status": "success"})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(stream_handler.get_stats())

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
        except Exception as e:
            pass # Skip corrupted logs safely
            
    return jsonify({"status": "success", "data": counts})

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

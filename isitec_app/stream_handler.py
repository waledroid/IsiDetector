import os
import cv2
import time
import threading
import logging
import supervision as sv
import numpy as np
import datetime
import csv
import json
import queue
import yaml
from pathlib import Path

# Helper for JSON serialization of NumPy types
def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    return obj


# Import inferencers from the parent directory
from src.inference.yolo_inferencer import YOLOInferencer
from src.inference.rfdetr_inferencer import RFDETRInferencer
from src.shared.vision_engine import VisionEngine

logger = logging.getLogger("IsiDetector-Web")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class TelemetryPublisher:
    """Simulates an Industrial MQTT publisher or Socket connection to Robot Arms."""
    def __init__(self):
        self.enabled = True

    def publish(self, class_name, track_id):
        if self.enabled:
            payload = {
                "event": "line_crossed",
                "object": str(class_name),
                "id": int(track_id) if hasattr(track_id, '__int__') else track_id,
                "timestamp": datetime.datetime.now().isoformat()
            }
            logger.info(f"[MQTT] PUBLISH -> topic: isitec/sorting, payload: {json.dumps(payload)}")

class LiveReader:
    """High-Stability Zero-Lag Background Reader."""
    def __init__(self, source):
        self.source = source
        self.cap = None
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        logger.info(f"🚀 LiveReader: Thread started for source {self.source}")
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.source)
                    if not self.cap.isOpened():
                        logger.error(f"❌ LiveReader: Failed to open {self.source}. Retrying in 2s...")
                        time.sleep(2)
                        continue
                    logger.info(f"✅ LiveReader: Connection established to {self.source}")

                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"⚠️ LiveReader: Stream disconnected. Retrying...")
                    self.cap.release()
                    self.cap = None
                    time.sleep(1)
                    continue

                if not self.q.empty():
                    try: self.q.get_nowait()
                    except: pass
                self.q.put(frame)
            except Exception as e:
                logger.error(f"🔥 LiveReader: Error: {e}")
                time.sleep(1)

    def get_frame(self):
        try: return self.q.get(timeout=0.2)
        except: return None

    def stop(self):
        logger.info("🛑 LiveReader: Stopping thread...")
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

class StreamHandler:
    def __init__(self):
        self.running = False
        self.reader = None
        self.lock = threading.Lock()
        
        # Vision state
        self.engine = None
        self.class_totals = {}
        self.last_detected = None
        self.mode_text = ""
        self.language = 'en'
        self.latest_annotated = None
        self.inf_thread = None
        self.source_is_image = False
        self.frame_ready = threading.Event()
        
        # IoT Telemetry
        self.publisher = TelemetryPublisher()
        
        # Load Config
        config_path = Path(__file__).resolve().parent.parent / "configs/train.yaml"
        self.config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        
        self.imgsz = self.config.get('image_size', 640)
        self.web_imgsz = min(self.imgsz, 640) # Safety cap for web throughput

    def set_language(self, lang):
        self.language = lang

    def get_stats(self):
        with self.lock:
            stats = {
                "is_running": self.running,
                "counts": self.class_totals,
                "last_detected": self.last_detected
            }
            return sanitize_for_json(stats)

    def draw_isi_ui(self, frame, counts, mode_text):
        h, w = frame.shape[:2]
        y_offset = 50
        max_text_width = 300
        
        # Black scoreboard area
        cv2.rectangle(frame, (w - 320, 0), (w, 130), (0, 0, 0), -1)

        for class_name, count in counts.items():
            text = f"{count:,} {class_name.upper()}S"
            color = (0, 255, 0) if "carton" in class_name.lower() else (255, 120, 0)
            cv2.putText(frame, text, (w - 300, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            y_offset += 45

        # Mode overlay
        cv2.rectangle(frame, (w - 180, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, mode_text, (w - 170, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def start(self, source, model_type, weights, device=None):
        """
        Start inference session.
        :param device: None = auto (GPU if available), "cpu" = force CPU, "cuda" = force GPU
        """
        with self.lock:
            if self.running:
                self.stop()

            if self.engine:
                del self.engine
                self.engine = None

            try:
                conf_thresh = self.config.get('inference', {}).get('conf_threshold', 0.3)
                is_detr = 'rfdetr' in model_type.lower() or (weights and 'rfdetr' in weights.lower())
                device_label = "CPU" if device == "cpu" else "GPU"

                if is_detr:
                    model_path = weights or "models/rfdetr/25-03-2026_0043/checkpoint_best_ema.pth"
                    if not Path(model_path).exists():
                        found = list(Path("runs/segment/models/rfdetr").rglob("*.onnx"))
                        if found: model_path = str(found[0])
                    base_engine = RFDETRInferencer(model_path=model_path, conf_threshold=conf_thresh, device=device)
                    self.mode_text = f"MODE 2 (DETR • {device_label})"
                else:
                    model_path = weights or "runs/segment/models/yolo/25-03-20262/weights/best.pt"
                    if not Path(model_path).exists():
                        found_pt = list(Path("runs/segment/models/yolo").rglob("*.pt"))
                        if found_pt:
                            model_path = str(found_pt[0])
                        else:
                            found_onnx = list(Path("runs/segment/models/yolo").rglob("*.onnx"))
                            if found_onnx: model_path = str(found_onnx[0])

                    base_engine = YOLOInferencer(model_path=model_path, conf_threshold=conf_thresh, device=device)
                    self.mode_text = f"MODE 1 (YOLO • {device_label})"
                
                self.engine = VisionEngine(inferencer=base_engine, config=self.config)
            except Exception as e:
                logger.error(f"Failed to load AI engine: {e}")
                return False, f"Model Error: {str(e)}"

            self.class_totals = {name: 0 for name in base_engine.class_names.values()}
            self.source_is_image = str(source).lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            
            # Launch Background Reader for Zero-Lag
            src_val = int(source) if str(source).isdigit() else source
            self.reader = LiveReader(src_val)
            
            # Launch Inference Thread
            self.running = True
            self.inf_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.inf_thread.start()
            
            logger.info(f"✅ Web App: Processing started ({self.mode_text})")
            return True, "Processing started."

    def stop(self):
        with self.lock:
            if not self.running: return True, "Already stopped."
            self.running = False
            
            inf_thread = self.inf_thread
            reader = self.reader
            engine = self.engine
            counts = self.class_totals.copy()

            self.inf_thread = None
            self.reader = None
            self.engine = None

        def _cleanup():
            try:
                if inf_thread and inf_thread.is_alive():
                    inf_thread.join(timeout=2.0)
                
                if reader:
                    reader.stop()
                
                if engine:
                    engine.cleanup(counts)
                
                logger.info("🛑 Web App: Session Cleanup Finished (Thread Joined & CSV Saved)")
            except Exception as e:
                logger.error(f"🔥 Error during cleanup: {e}")

        threading.Thread(target=_cleanup, daemon=True).start()
        return True, "Stopping session..."

    def _inference_loop(self):
        """Dedicated thread for AI processing to keep the UI fluid."""
        logger.info("🎬 Inference Thread: Started")
        while self.running:
            with self.lock:
                reader = self.reader
                engine = self.engine
            
            if not reader or not engine:
                time.sleep(0.01)
                continue
                
            frame = reader.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue
                
            try:
                annotated, detections, event = engine.process_frame(frame, self.class_totals)
                
                if event:
                    with self.lock:
                        self.last_detected = {"class": event['class'], "time": datetime.datetime.now().strftime("%H:%M:%S")}
                    self.publisher.publish(event['class'], event['id'])

                annotated = self.draw_isi_ui(annotated, self.class_totals, self.mode_text)
                
                # Performance Optimization: Downsample for Web if needed
                h, w = annotated.shape[:2]
                if w > self.web_imgsz or h > self.web_imgsz:
                    annotated = cv2.resize(annotated, (self.web_imgsz, self.web_imgsz))
                
                # Quality 50 is much lighter for 30fps MJPEG
                _, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                self.latest_annotated = buffer.tobytes()
                self.frame_ready.set() # Trigger web yield
            except Exception as e:
                logger.error(f"Inference Loop Error: {e}")
                time.sleep(0.1)
        logger.info("🛑 Inference Thread: Stopped")

    def generate_frames(self):
        while True:
            # Wait for a new frame or timeout (to send STANDBY)
            self.frame_ready.wait(timeout=0.1)
            self.frame_ready.clear()

            with self.lock:
                running = self.running
                source_is_img = self.source_is_image
                frame_bytes = self.latest_annotated

            if not running or frame_bytes is None:
                blank = np.zeros((self.web_imgsz, self.web_imgsz, 3), dtype=np.uint8)
                cv2.putText(blank, "STANDBY", (int(self.web_imgsz/3), int(self.web_imgsz/2)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                _, buffer = cv2.imencode('.jpg', blank)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + f"{len(frame_bytes)}".encode() + b'\r\n\r\n' +
                       frame_bytes + b'\r\n')
                time.sleep(1)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + f"{len(frame_bytes)}".encode() + b'\r\n\r\n' +
                   frame_bytes + b'\r\n')

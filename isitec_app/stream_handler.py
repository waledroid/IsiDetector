import cv2
import time
import threading
import logging
import supervision as sv
import numpy as np
import datetime
import os
import csv
import json

# Import inferencers from the parent directory
from src.inference.yolo_inferencer import YOLOInferencer
from src.inference.rfdetr_inferencer import RFDETRInferencer

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
                "object": class_name,
                "id": track_id,
                "timestamp": datetime.datetime.now().isoformat()
            }
            # In production, this can use paho-mqtt to emit to mosquitto/PLC
            logger.info(f"[MQTT] PUBLISH -> topic: isitec/sorting, payload: {json.dumps(payload)}")

class StreamHandler:
    def __init__(self):
        self.running = False
        self.cap = None
        self.engine = None
        self.tracker = None
        self.line_zone = None
        self.mode_text = ""
        self.lock = threading.Lock()
        
        # Annotators
        self.mask_annotator = sv.MaskAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=0, text_scale=0)
        
        # State
        self.counted_ids = set()
        self.class_totals = {}
        self.last_detected = None
        
        # IoT Telemetry
        self.publisher = TelemetryPublisher()
        
        # Logging
        self.log_file = None
        self.csv_writer = None
        self.language = 'en'

        # Create logs directory
        self.logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)

    def set_language(self, lang):
        self.language = lang

    def get_stats(self):
        with self.lock:
            return {
                "is_running": self.running,
                "counts": self.class_totals,
                "last_detected": self.last_detected
            }

    def draw_isi_ui(self, frame, counts, mode_text):
        h, w = frame.shape[:2]
        y_offset = 50
        max_text_width = 320
        
        for class_name, count in counts.items():
            text = f"{count:,} {class_name.upper()}S"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            if tw + 40 > max_text_width:
                max_text_width = tw + 40

        cv2.rectangle(frame, (w - max_text_width, 0), (w, 130), (0, 0, 0), -1)

        for class_name, count in counts.items():
            text = f"{count:,} {class_name.upper()}S"
            color = (0, 255, 0) if "carton" in class_name.lower() else (255, 120, 0)
            cv2.putText(frame, text, (w - (max_text_width - 20), y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            y_offset += 50

        cv2.rectangle(frame, (w - 200, h - 50), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, mode_text, (w - 180, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame

    def start(self, source, model_type, weights):
        with self.lock:
            if self.running:
                return False, "A stream is already running. Stop it first."
            
            # 0. Provide default weights if blank from UI
            if not weights:
                if 'rfdetr' in model_type.lower():
                    weights = "models/best_rfdetr.onnx" # Placeholder or path you use
                else:
                    weights = "models/best.onnx"
            
            # 1. Setup engine
            try:
                if 'rfdetr' in model_type.lower() or 'rfdetr' in weights.lower():
                    self.engine = RFDETRInferencer(model_path=weights, conf_threshold=0.3)
                    self.mode_text = "MODE 2"
                else:
                    self.engine = YOLOInferencer(model_path=weights, conf_threshold=0.3)
                    self.mode_text = "MODE 1"
            except Exception as e:
                return False, f"Failed to load engine: {str(e)}"

            self.class_totals = {name: 0 for name in self.engine.class_names.values()}
            self.counted_ids = set()

            # 2. Setup Source
            src_val = int(source) if str(source).isdigit() else source
            self.cap = cv2.VideoCapture(src_val)
            
            if not self.cap.isOpened():
                return False, f"Failed to open source: {source}"
                
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                return False, "Failed to read from source."
                
            h, w = frame.shape[:2]
            
            # 3. Setup Tracker and Line
            self.tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=60, match_thresh=0.9)
            line_x = int(w * 0.65) # Match run_live.py logic
            self.line_zone = sv.LineZone(
                start=sv.Point(line_x, 0), 
                end=sv.Point(line_x, h),
                triggering_anchors=[sv.Position.BOTTOM_CENTER]
            )
            
            # 4. Setup Logging for live streams
            if str(source).isdigit() or str(source).startswith('rtsp://') or str(source).startswith('http://'):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filepath = os.path.join(self.logs_dir, f"detections_{timestamp}.csv")
                self.log_file = open(log_filepath, 'w', newline='')
                self.csv_writer = csv.writer(self.log_file)
                self.csv_writer.writerow(['Timestamp', 'Class', 'Action'])
            else:
                self.log_file = None
                self.csv_writer = None

            self.running = True
            logger.info(f"Started stream: {source} with {model_type}")
            return True, "Stream started successfully."

    def stop(self):
        with self.lock:
            if not self.running:
                return False, "No stream is currently running."
            
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            
            if self.log_file:
                self.log_file.close()
                self.log_file = None
                self.csv_writer = None
                
            return True, "Stream stopped successfully."

    def generate_frames(self):
        while True:
            if not self.running or self.cap is None:
                # Standby image when no active stream
                blank = np.zeros((640, 640, 3), dtype=np.uint8)
                if getattr(self, 'language', 'en') == 'fr':
                    cv2.putText(blank, "EN ATTENTE", (220, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
                    cv2.putText(blank, "EN ATTENTE DE FLUX", (150, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
                else:
                    cv2.putText(blank, "STANDBY", (240, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
                    cv2.putText(blank, "AWAITING STREAM", (160, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
                ret, buffer = cv2.imencode('.jpg', blank)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                continue
            
            # 1. Prediction
            detections = self.engine.predict_frame(frame)
            
            # 2. Tracking
            detections = self.tracker.update_with_detections(detections)
            
            # 3. Trigger Line
            in_cross, out_cross = self.line_zone.trigger(detections=detections)
            all_crossings = in_cross | out_cross
            
            for i, crossed in enumerate(all_crossings):
                if crossed and detections.tracker_id is not None:
                    t_id = detections.tracker_id[i]
                    if t_id not in self.counted_ids:
                        class_id = detections.class_id[i]
                        name = self.engine.class_names.get(class_id, "object")
                        
                        # 1. Update Internal Counts
                        self.class_totals[name] = self.class_totals.get(name, 0) + 1
                        self.counted_ids.add(t_id)
                        self.last_detected = {
                            "class": name,
                            "time": datetime.datetime.now().strftime("%H:%M:%S")
                        }
                        
                        # 2. Fire telemetry to Industrial Robot/PLC
                        self.publisher.publish(name, t_id)
                        
                        # 3. Write CSV Log if needed
                        if self.csv_writer:
                            ts = datetime.datetime.now().isoformat()
                            self.csv_writer.writerow([ts, name, 'Crossed Line'])

            # 4. Annotation
            annotated = frame.copy()
            if detections.mask is not None:
                annotated = self.mask_annotator.annotate(scene=annotated, detections=detections)
            
            annotated = self.trace_annotator.annotate(scene=annotated, detections=detections)
            annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
            
            if detections.tracker_id is not None:
                labels = [f"#{t_id} {self.engine.class_names.get(c_id, 'obj')}" 
                          for c_id, t_id in zip(detections.class_id, detections.tracker_id)]
                annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
            
            annotated = self.line_annotator.annotate(frame=annotated, line_counter=self.line_zone)
            annotated = self.draw_isi_ui(annotated, self.class_totals, self.mode_text)

            ret, buffer = cv2.imencode('.jpg', annotated)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


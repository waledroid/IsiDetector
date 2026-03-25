#!/usr/bin/env python3
import sys
import cv2
import queue
import threading
import argparse
import logging
import supervision as sv
import numpy as np
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference.yolo_inferencer import YOLOInferencer
from src.inference.rfdetr_inferencer import RFDETRInferencer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("IsiDetector-Live")

class LiveStreamReader:
    """Background thread to keep the RTSP/Camera buffer fresh (Zero-Lag)."""
    def __init__(self, source):
        # Convert "0" to int for local USB webcams
        self.source = int(source) if source.isdigit() else source
        self.cap = cv2.VideoCapture(self.source)
        self.q = queue.Queue(maxsize=2) # Only keep 2 newest frames
        self.running = True
        
        if not self.cap.isOpened():
            logger.error(f"❌ Could not connect to: {source}")
            sys.exit(1)
            
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: break
            if not self.q.empty():
                try: self.q.get_nowait() # Drop stale frame
                except queue.Empty: pass
            self.q.put(frame)

    def get_frame(self):
        return self.q.get()

    def stop(self):
        self.running = False
        self.cap.release()

def draw_isi_ui(frame, counts, mode_text):
    """Custom UI with per-class counters and Mode indicator."""
    h, w = frame.shape[:2]
    
    # 1. Per-Class Counters (Top Right)
    max_text_width = 320
    for name, count in counts.items():
        text = f"{count:,} {name.upper()}S"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        max_text_width = max(max_text_width, tw + 40)

    cv2.rectangle(frame, (w - max_text_width, 0), (w, 130), (0, 0, 0), -1)
    y_off = 50
    for name, count in counts.items():
        color = (0, 255, 0) if "carton" in name.lower() else (255, 120, 0)
        cv2.putText(frame, f"{count:,} {name.upper()}S", (w - (max_text_width - 20), y_off), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        y_off += 50

    # 2. Mode Indicator (Bottom Right)
    cv2.rectangle(frame, (w - 200, h - 50), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, mode_text, (w - 180, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

def main():
    parser = argparse.ArgumentParser(description="IsiDetector Live RTSP Stream")
    parser.add_argument('--weights', type=str, required=True, help="Path to weights")
    parser.add_argument('--source', type=str, required=True, help="RTSP URL or 0 for Webcam")
    args = parser.parse_args()

    # 1. Engine & Mode Logic
    weights_path = str(args.weights).lower()
    
    if weights_path.endswith('.onnx'):
        # Switch to our new high-speed ONNX engine
        from src.inference.onnx_inferencer import ONNXInferencer
        logger.info(f"⚡ Loading High-Speed ONNX Engine: {args.weights}")
        engine = ONNXInferencer(model_path=args.weights, conf_threshold=0.3)
        mode_text = "MODE ONNX"
    elif 'rfdetr' in weights_path:
        # Standard PyTorch RF-DETR
        engine = RFDETRInferencer(model_path=args.weights, conf_threshold=0.3)
        mode_text = "MODE 2"
    else:
        # Standard YOLO
        engine = YOLOInferencer(model_path=args.weights, conf_threshold=0.3)
        mode_text = "MODE 1"

    # 2. Setup Analytics
    stream = LiveStreamReader(args.source)
    # Give the thread a moment to buffer the first frame
    first_frame = stream.get_frame()
    h, w = first_frame.shape[:2]
    
    tracker = sv.ByteTrack()
    # Vertical line in center
    line_zone = sv.LineZone(start=sv.Point(w//2, 0), end=sv.Point(w//2, h))

    # Annotators
    mask_annotator, trace_annotator = sv.MaskAnnotator(), sv.TraceAnnotator()
    box_annotator, label_annotator = sv.BoxAnnotator(), sv.LabelAnnotator(text_scale=0.5)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=0, text_scale=0)

    counted_ids = set()
    class_totals = {name: 0 for name in engine.class_names.values()}

    logger.info(f"📡 {mode_text} Live Stream Active: {args.source}")
    
    try:
        while True:
            # 1. Grab the frame
            frame = stream.get_frame()
            if frame is None: continue

            # 2. AI Inference
            frame_input = np.ascontiguousarray(frame)
            detections = engine.predict_frame(frame_input)
            
            # 3. Tracking
            detections = tracker.update_with_detections(detections)
            
            # 4. Counting Logic (Trigger the line)
            # This is where in_cross and out_cross are defined
            in_cross, out_cross = line_zone.trigger(detections=detections)
            all_crossings = in_cross | out_cross
            
            for i, crossed in enumerate(all_crossings):
                if crossed and detections.tracker_id is not None:
                    t_id = detections.tracker_id[i]
                    if t_id not in counted_ids:
                        class_id = detections.class_id[i]
                        name = engine.class_names.get(class_id, "object")
                        class_totals[name] = class_totals.get(name, 0) + 1
                        counted_ids.add(t_id)

            # 5. Visual Composition
            annotated = frame.copy()
            
            if detections.mask is not None:
                annotated = mask_annotator.annotate(scene=annotated, detections=detections)
            
            annotated = trace_annotator.annotate(scene=annotated, detections=detections)
            annotated = box_annotator.annotate(scene=annotated, detections=detections)
            
            # Safely build labels for tracked objects
            if detections.tracker_id is not None:
                labels = [f"#{t_id} {engine.class_names.get(c_id, 'obj')}" 
                          for c_id, t_id in zip(detections.class_id, detections.tracker_id)]
                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
            
            annotated = line_annotator.annotate(frame=annotated, line_counter=line_zone)
            
            # Final UI Elements
            annotated = draw_isi_ui(annotated, class_totals, mode_text)

            cv2.imshow("IsiDetector", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

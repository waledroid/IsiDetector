# src/inference/yolo_inferencer.py
import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from src.inference.base_inferencer import BaseInferencer

class YOLOInferencer(BaseInferencer):
    """Concrete implementation for YOLO Segmentation models with 640px scaling."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        super().__init__(model_path, conf_threshold)
        _tmp_model = self._load_model()
        self.class_names = _tmp_model.names 

    def _load_model(self):
        return YOLO(self.model_path)

    def predict_frame(self, frame: np.ndarray):
        """Processes a video frame with optimized 640px downsizing."""
        orig_h, orig_w = frame.shape[:2]
        
        # 1. Scaling Logic (Only downsize if larger than 640)
        if max(orig_h, orig_w) > 640:
            scale = 640 / max(orig_h, orig_w)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            input_frame = cv2.resize(frame, (new_w, new_h))
            needs_scaling = True
        else:
            input_frame = frame
            scale = 1.0
            needs_scaling = False

        # 2. Inference
        results = self.model(input_frame, conf=self.conf_threshold, imgsz=640, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # 3. Manual Scaling Back
        if needs_scaling and len(detections) > 0:
            inv_scale = 1 / scale
            detections.xyxy = detections.xyxy.copy()
            detections.xyxy *= inv_scale
            
            if detections.mask is not None:
                resized_masks = []
                for mask in detections.mask:
                    m = cv2.resize(mask.astype("uint8"), (orig_w, orig_h), 
                                 interpolation=cv2.INTER_NEAREST)
                    resized_masks.append(m.astype(bool))
                detections.mask = np.array(resized_masks)
                
        return detections

    def predict(self, source: str, show: bool = False, save: bool = False):
        source_val = int(source) if str(source).isdigit() else source
        results_gen = self.model.predict(
            source=source_val, conf=self.conf_threshold, stream=True, verbose=False
        )
        for r in results_gen:
            detections = sv.Detections.from_ultralytics(r)
            yield {"path": r.path, "detections": detections, "raw": r}

    def get_summary(self, result) -> dict:
        detections = result['detections']
        return {"file_name": Path(result['path']).name, "counts": {"Total": len(detections)}}

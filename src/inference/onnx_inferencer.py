# src/inference/onnx_inferencer.py
import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv
from pathlib import Path
from src.inference.base_inferencer import BaseInferencer

class ONNXInferencer(BaseInferencer):
    """High-speed ONNX Inference Engine for both YOLO and RF-DETR."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        super().__init__(model_path, conf_threshold)
        
        #========================== 1. Initialize ONNX Runtime with CUDA (GPU) support==========================
        providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Force CPU only
        # providers = ['CPUExecutionProvider'] 
        # self.session = ort.InferenceSession(model_path, providers=providers)
        #=======================================================================================================
        
        # 2. Get Input Metadata
        self.input_node = self.session.get_inputs()[0]
        self.input_name = self.input_node.name
        self.input_shape = self.input_node.shape # e.g. [1, 3, 432, 432]
        
        # 3. Get Output Names to detect model type (YOLO vs RF-DETR)
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.is_rfdetr = "dets" in self.output_names
        
        # Map indices to your custom classes
        self.class_names = {0: "carton", 1: "polybag"}

    def _load_model(self):
        return None # Managed by onnxruntime session

    def preprocess(self, frame):
        """Resizes and normalizes frame to NCHW format."""
        h, w = self.input_shape[2], self.input_shape[3]
        img = cv2.resize(frame, (w, h))
        img = img.transpose((2, 0, 1)) # HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        return img

    def predict_frame(self, frame: np.ndarray):
        """Core inference loop used by run_live.py"""
        orig_h, orig_w = frame.shape[:2]
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})

        if self.is_rfdetr:
            return self._postprocess_rfdetr(outputs, orig_w, orig_h)
        else:
            return self._postprocess_yolo(outputs, orig_w, orig_h)

    def _postprocess_rfdetr(self, outputs, orig_w, orig_h):
        bboxes = outputs[0][0]  # [200, 4]
        logits = outputs[1][0]  # [200, 91]
        masks  = outputs[2][0]  # [200, 108, 108]

        # 1. Slice logits to ignore the background class if necessary
        # Often index 0 is background in DETR ONNX, so your 'Carton' might be 1
        scores = np.max(logits, axis=1)
        class_ids = np.argmax(logits, axis=1)

        # 2. Thresholding
        mask_idx = (scores > self.conf_threshold)
        
        if not np.any(mask_idx):
            return sv.Detections.empty()

        # 3. COORDINATE FIX: 
        # Check if boxes are [x_center, y_center, w, h] or [x1, y1, x2, y2]
        # Most DETR ONNX exports use normalized [0, 1] x_center format.
        curr_boxes = bboxes[mask_idx]
        
        # Convert Center-XYWH to XYXY if needed
        # (Comment this block out if your boxes are already XYXY)
        x_c, y_c, w, h = curr_boxes[:, 0], curr_boxes[:, 1], curr_boxes[:, 2], curr_boxes[:, 3]
        new_boxes = np.zeros_like(curr_boxes)
        new_boxes[:, 0] = x_c - (w / 2)
        new_boxes[:, 1] = y_c - (h / 2)
        new_boxes[:, 2] = x_c + (w / 2)
        new_boxes[:, 3] = y_c + (h / 2)
        
        # Scale to original resolution
        rescale = np.array([orig_w, orig_h, orig_w, orig_h])
        final_boxes = new_boxes * rescale

        # 4. Label Shift Fix: 
        # If your labels are shifted by 1, subtract 1 here:
        final_class_ids = class_ids[mask_idx] - 1 
        # Ensure no negative IDs
        final_class_ids = np.clip(final_class_ids, 0, 1)

        # 5. Mask Resize
        final_masks = []
        for m in masks[mask_idx]:
            m_resized = cv2.resize(m, (orig_w, orig_h))
            final_masks.append(m_resized > 0.5)

        return sv.Detections(
            xyxy=final_boxes,
            confidence=scores[mask_idx],
            class_id=final_class_ids.astype(int),
            mask=np.array(final_masks)
        )

    def _postprocess_yolo(self, outputs, orig_w, orig_h):
        # 1. Parse Tensors
        # output0: [1, 38, 8400] -> [8400, 38]
        # output1: [1, 32, 160, 160] (Mask Prototypes)
        preds = np.squeeze(outputs[0]).T 
        protos = outputs[1] 

        # 2. Split: [box(4), scores(2), mask_coeffs(32)]
        boxes = preds[:, :4]
        scores = preds[:, 4:6]
        mask_coeffs = preds[:, 6:]

        # 3. Get Class and Max Score
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # 4. Filter by Threshold
        idx = confidences > self.conf_threshold
        if not np.any(idx):
            return sv.Detections.empty()

        # 5. Apply Manual NMS (This removes the "lots of boxes" problem)
        # Convert XYWH to XYXY for NMS
        nm_boxes = boxes[idx]
        x, y, w, h = nm_boxes[:, 0], nm_boxes[:, 1], nm_boxes[:, 2], nm_boxes[:, 3]
        boxes_xyxy = np.stack([x - w/2, y - h/2, x + w/2, y + h/2], axis=-1)
        
        # Supervision's NMS is very fast
        detections = sv.Detections(
            xyxy=boxes_xyxy,
            confidence=confidences[idx],
            class_id=class_ids[idx]
        )
        detections = detections.with_nms(threshold=0.5)

        # 6. COORDINATE SCALE (640 -> 1080p)
        scale_x, scale_y = orig_w / 640, orig_h / 640
        detections.xyxy[:, [0, 2]] *= scale_x
        detections.xyxy[:, [1, 3]] *= scale_y

        # 7. TODO: MASK DECODING
        # YOLO masks require a matrix multiplication of (mask_coeffs @ protos)
        # We will keep them as 'None' for 1 second to verify the boxes first.
        return detections
        
    # --- Methods required by BaseInferencer Contract ---

    def predict(self, source: str, show: bool = False, save: bool = False):
        """Predict on a static image file."""
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Source not found: {source}")
        
        detections = self.predict_frame(frame)
        
        # Small summary dict for consistency with runner
        yield {"path": source, "detections": detections, "raw": detections}

    def get_summary(self, result) -> dict:
        """Returns detection counts for reporting."""
        detections = result['detections']
        return {
            "file_name": Path(result['path']).name,
            "counts": {"Total": len(detections)}
        }

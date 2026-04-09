"""
OpenVINO Inference Engine — optimized for CPU (Intel) deployment.
Loads models exported via export_engine.py (.xml + .bin format).
"""

import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

try:
    import openvino as ov
except ImportError:
    raise ImportError("openvino not installed. pip install openvino")

from src.inference.base_inferencer import BaseInferencer


class OpenVINOInferencer(BaseInferencer):
    """OpenVINO inference engine — runs on CPU (or Intel iGPU if available)."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        device: str = None,
        imgsz: int = None,
    ):
        super().__init__(model_path, conf_threshold, device, imgsz)

        xml_path = Path(model_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"OpenVINO model not found: {xml_path}")

        core = ov.Core()
        model = core.read_model(str(xml_path))

        # Determine device: "CPU", "GPU" (Intel iGPU), or "AUTO"
        ov_device = "CPU"
        if self.device and self.device.upper() in ("GPU", "AUTO"):
            available = core.available_devices
            if "GPU" in available:
                ov_device = "GPU"
            else:
                logger.info(f"Intel GPU not available, using CPU. Available: {available}")

        self.compiled = core.compile_model(model, ov_device)
        self.infer_request = self.compiled.create_infer_request()

        # Parse model IO
        self.input_layer = self.compiled.input(0)
        input_shape = self.input_layer.shape  # [1, 3, H, W]
        self.model_h = input_shape[2]
        self.model_w = input_shape[3]

        # Detect model type from output names/shapes
        output_names = [self.compiled.output(i).get_any_name() for i in range(len(self.compiled.outputs))]
        self.is_rfdetr = any(n in output_names for n in ('dets', 'pred_logits', 'bboxes', 'labels'))
        self._has_mask_output = len(self.compiled.outputs) > 1
        self._num_outputs = len(self.compiled.outputs)

        # Number of classes
        if self.is_rfdetr:
            for i, name in enumerate(output_names):
                if name in ('labels', 'pred_logits'):
                    self.nc = self.compiled.output(i).shape[-1]
                    break
        else:
            out_shape = self.compiled.output(0).shape
            if len(out_shape) == 3:
                mask_coeffs = 32 if self._has_mask_output else 0
                self.nc = out_shape[2] - 4 - mask_coeffs
                if self.nc <= 0:
                    self.nc = 2

        logger.info(f"OpenVINO model loaded: {xml_path}")
        logger.info(f"  Type: {'RF-DETR' if self.is_rfdetr else 'YOLO'} | "
                     f"Input: {self.model_w}x{self.model_h} | "
                     f"Classes: {self.nc} | Device: {ov_device}")

    def _load_model(self):
        return self.compiled if hasattr(self, 'compiled') else None

    # ── Preprocessing ────────────────────────────────────────────────────────

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.model_w, self.model_h), interpolation=cv2.INTER_LINEAR)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        return np.ascontiguousarray(img[np.newaxis, ...])

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_frame(self, frame: np.ndarray) -> sv.Detections:
        input_tensor = self.preprocess(frame)

        try:
            self.infer_request.infer({0: input_tensor})
            outputs = [self.infer_request.get_output_tensor(i).data.copy()
                       for i in range(self._num_outputs)]
        except Exception as e:
            logger.error(f"OpenVINO inference failed: {e}")
            return sv.Detections.empty()

        orig_h, orig_w = frame.shape[:2]
        try:
            if self.is_rfdetr:
                return self._postprocess_rfdetr(outputs, orig_w, orig_h)
            else:
                return self._postprocess_yolo(outputs, orig_w, orig_h)
        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            return sv.Detections.empty()

    # ── YOLO Postprocessing ──────────────────────────────────────────────────
    # Same logic as ONNXInferencer — boxes are in model pixel space (0-640)

    def _postprocess_yolo(self, outputs: List[np.ndarray], orig_w: int, orig_h: int) -> sv.Detections:
        preds = outputs[0][0] if outputs[0].ndim == 3 else outputs[0]
        if preds.shape[0] == 0:
            return sv.Detections.empty()

        boxes = preds[:, :4].copy()
        scores = preds[:, 4:4 + self.nc]
        mask_coeffs = preds[:, 4 + self.nc:] if self._has_mask_output else None

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        keep = confidences > self.conf_threshold
        if not np.any(keep):
            return sv.Detections.empty()

        boxes = boxes[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]
        if mask_coeffs is not None:
            mask_coeffs = mask_coeffs[keep]

        # Scale from model space to original
        scale_x = orig_w / self.model_w
        scale_y = orig_h / self.model_h
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
        if not np.all(valid):
            boxes, class_ids, confidences = boxes[valid], class_ids[valid], confidences[valid]
            if mask_coeffs is not None:
                mask_coeffs = mask_coeffs[valid]

        if len(boxes) == 0:
            return sv.Detections.empty()

        masks = None
        if mask_coeffs is not None and len(outputs) > 1:
            proto = outputs[1][0] if outputs[1].ndim == 4 else outputs[1]
            masks = self._process_masks(proto, mask_coeffs, boxes, orig_h, orig_w)

        return sv.Detections(
            xyxy=boxes, confidence=confidences,
            class_id=class_ids.astype(int), mask=masks,
        )

    def _process_masks(self, proto: np.ndarray, coeffs: np.ndarray,
                       boxes: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        try:
            proto_h, proto_w = proto.shape[1], proto.shape[2]
            masks_raw = coeffs @ proto.reshape(proto.shape[0], -1)
            masks_raw = 1.0 / (1.0 + np.exp(-masks_raw.reshape(-1, proto_h, proto_w)))

            n = len(boxes)
            masks = np.zeros((n, orig_h, orig_w), dtype=np.uint8)
            sx, sy = proto_w / orig_w, proto_h / orig_h

            for i in range(n):
                mask_i = masks_raw[i].copy()
                px1 = max(0, int(boxes[i, 0] * sx))
                py1 = max(0, int(boxes[i, 1] * sy))
                px2 = min(proto_w, int(boxes[i, 2] * sx))
                py2 = min(proto_h, int(boxes[i, 3] * sy))
                mask_i[:py1, :] = 0; mask_i[py2:, :] = 0
                mask_i[:, :px1] = 0; mask_i[:, px2:] = 0
                masks[i] = (cv2.resize(mask_i, (orig_w, orig_h)) > 0.5).astype(np.uint8)
            return masks
        except Exception as e:
            logger.warning(f"Mask processing failed: {e}")
            return None

    # ── RF-DETR Postprocessing ───────────────────────────────────────────────

    def _postprocess_rfdetr(self, outputs: List[np.ndarray], orig_w: int, orig_h: int) -> sv.Detections:
        bboxes = outputs[0][0] if outputs[0].ndim == 3 else outputs[0]
        logits = outputs[1][0] if outputs[1].ndim == 3 else outputs[1]

        probs = 1.0 / (1.0 + np.exp(-logits))
        scores = np.max(probs, axis=1)
        class_ids = np.argmax(probs, axis=1)

        keep = scores > self.conf_threshold
        if not np.any(keep):
            return sv.Detections.empty()

        bboxes = bboxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        boxes_xyxy = np.column_stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        boxes_xyxy[:, [0, 2]] *= orig_w
        boxes_xyxy[:, [1, 3]] *= orig_h
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        masks = None
        if len(outputs) > 2:
            mask_preds = outputs[2][0] if outputs[2].ndim == 4 else outputs[2]
            try:
                mask_preds = mask_preds[keep]
                n = len(mask_preds)
                masks = np.zeros((n, orig_h, orig_w), dtype=np.uint8)
                for i in range(n):
                    m = 1.0 / (1.0 + np.exp(-mask_preds[i]))
                    masks[i] = (cv2.resize(m, (orig_w, orig_h)) > 0.5).astype(np.uint8)
            except Exception as e:
                logger.warning(f"RF-DETR mask processing failed: {e}")

        return sv.Detections(
            xyxy=boxes_xyxy, confidence=scores,
            class_id=class_ids.astype(int), mask=masks,
        )

    # ── Required abstract methods ────────────────────────────────────────────

    def predict(self, source: str, show: bool = False, save: bool = False):
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Source not found: {source}")
        detections = self.predict_frame(frame)
        yield {"path": source, "detections": detections, "raw": detections}

    def get_summary(self, result: dict) -> dict:
        detections = result['detections']
        class_counts = {}
        if detections.class_id is not None:
            for cid in detections.class_id:
                name = self.class_names.get(cid, str(cid))
                class_counts[name] = class_counts.get(name, 0) + 1
        return {
            "file_name": Path(result['path']).name,
            "total_detections": len(detections),
            "class_counts": class_counts,
        }

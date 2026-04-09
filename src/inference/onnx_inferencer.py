"""
Optimized ONNX Inference Engine — GPU (CUDA) or CPU fallback.
Compatible with ONNX Runtime 1.24+
Handles both YOLO (nms=True export) and RF-DETR ONNX models.
"""

import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime not installed. pip install onnxruntime-gpu or onnxruntime")

# Detect CPU/GPU conflict: both packages installed but CUDA provider missing
_available = ort.get_available_providers()
if 'CUDAExecutionProvider' not in _available:
    try:
        import importlib.metadata as _meta
        _has_cpu = bool(_meta.distribution('onnxruntime'))
        _has_gpu = bool(_meta.distribution('onnxruntime-gpu'))
        if _has_cpu and _has_gpu:
            logger.warning(
                "Both onnxruntime and onnxruntime-gpu are installed — CPU version is shadowing GPU. "
                "Fix: pip uninstall onnxruntime -y && pip install onnxruntime-gpu"
            )
    except _meta.PackageNotFoundError:
        pass

from src.inference.base_inferencer import BaseInferencer


class OptimizedONNXInferencer(BaseInferencer):
    """ONNX inference engine with automatic GPU/CPU selection."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        device: str = None,
        imgsz: int = None,
        gpu_device_id: int = 0,
    ):
        super().__init__(model_path, conf_threshold, device, imgsz)

        self.gpu_device_id = gpu_device_id

        # Session setup
        self.sess_options = ort.SessionOptions()
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_options.enable_cpu_mem_arena = False
        self.sess_options.enable_mem_pattern = True
        self.sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess_options.intra_op_num_threads = 2
        self.sess_options.inter_op_num_threads = 2
        self.sess_options.log_severity_level = 3

        self.session = self._create_session(str(model_path))
        self._parse_model_metadata()
        self._warmup()

        logger.info(f"ONNX model loaded: {model_path}")
        logger.info(f"  Type: {'RF-DETR' if self.is_rfdetr else 'YOLO'} | "
                     f"Input: {self.model_w}x{self.model_h} | "
                     f"Classes: {self.nc} | "
                     f"Provider: {self.session.get_providers()[0]}")

    def _load_model(self):
        return self.session if hasattr(self, 'session') else None

    # ── Session Creation ─────────────────────────────────────────────────────

    def _create_session(self, model_path: str) -> ort.InferenceSession:
        available = ort.get_available_providers()
        force_cpu = (self.device == "cpu")

        if force_cpu or 'CUDAExecutionProvider' not in available:
            providers = ['CPUExecutionProvider']
        else:
            cuda_opts = {
                'device_id': self.gpu_device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'HEURISTIC',
                'do_copy_in_default_stream': True,
            }
            providers = [('CUDAExecutionProvider', cuda_opts), 'CPUExecutionProvider']

        return ort.InferenceSession(model_path, sess_options=self.sess_options, providers=providers)

    # ── Model Metadata ───────────────────────────────────────────────────────

    def _parse_model_metadata(self):
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = list(inp.shape)
        if self.input_shape[0] is None:
            self.input_shape[0] = 1

        self.model_h = self.input_shape[2]
        self.model_w = self.input_shape[3]

        self.output_names = [o.name for o in self.session.get_outputs()]
        self.is_rfdetr = any(n in self.output_names for n in ('dets', 'pred_logits', 'bboxes', 'labels'))

        # Detect number of classes from output shape
        if self.is_rfdetr:
            # RF-DETR: labels output shape [1, N, num_classes]
            for o in self.session.get_outputs():
                if o.name in ('labels', 'pred_logits'):
                    self.nc = o.shape[-1]
                    break
        else:
            # YOLO with NMS: output [1, 300, 4+nc+32]
            out_shape = self.session.get_outputs()[0].shape
            if len(out_shape) == 3:
                # Check if there's a mask proto output
                has_masks = len(self.session.get_outputs()) > 1
                mask_coeffs = 32 if has_masks else 0
                self.nc = out_shape[2] - 4 - mask_coeffs
                if self.nc <= 0:
                    self.nc = 2

        # Store whether model has mask output
        self._has_mask_output = len(self.session.get_outputs()) > 1

    def _warmup(self):
        dummy = np.random.randn(*self.input_shape).astype(np.float32)
        for _ in range(3):
            self.session.run(None, {self.input_name: dummy})

    # ── Preprocessing ────────────────────────────────────────────────────────

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.model_w, self.model_h), interpolation=cv2.INTER_LINEAR)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        return np.ascontiguousarray(img[np.newaxis, ...])

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_frame(self, frame: np.ndarray) -> sv.Detections:
        input_tensor = self.preprocess(frame)
        try:
            outputs = self.session.run(None, {self.input_name: input_tensor})
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            return sv.Detections.empty()

        orig_h, orig_w = frame.shape[:2]
        try:
            if self.is_rfdetr:
                return self._postprocess_rfdetr(outputs, orig_w, orig_h)
            else:
                return self._postprocess_yolo(outputs, orig_w, orig_h, frame)
        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            return sv.Detections.empty()

    # ── YOLO Postprocessing ──────────────────────────────────────────────────

    def _postprocess_yolo(self, outputs: List[np.ndarray], orig_w: int, orig_h: int,
                          frame: np.ndarray) -> sv.Detections:
        preds = outputs[0][0]  # [300, 4+nc+32]
        if preds.shape[0] == 0:
            return sv.Detections.empty()

        boxes = preds[:, :4].copy()  # [x1, y1, x2, y2] in model pixel space (0-640)
        scores = preds[:, 4:4 + self.nc]
        mask_coeffs = preds[:, 4 + self.nc:] if self._has_mask_output else None

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence
        keep = confidences > self.conf_threshold
        if not np.any(keep):
            return sv.Detections.empty()

        boxes = boxes[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]
        if mask_coeffs is not None:
            mask_coeffs = mask_coeffs[keep]

        # Scale boxes from model space (640x640) to original image space
        scale_x = orig_w / self.model_w
        scale_y = orig_h / self.model_h
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # Clip to image boundaries
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        # Remove invalid boxes
        valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
        if not np.all(valid):
            boxes = boxes[valid]
            class_ids = class_ids[valid]
            confidences = confidences[valid]
            if mask_coeffs is not None:
                mask_coeffs = mask_coeffs[valid]

        if len(boxes) == 0:
            return sv.Detections.empty()

        # Process masks if available
        masks = None
        if mask_coeffs is not None and len(outputs) > 1:
            masks = self._process_yolo_masks(outputs[1][0], mask_coeffs, boxes, orig_h, orig_w)

        return sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids.astype(int),
            mask=masks,
        )

    def _process_yolo_masks(self, proto: np.ndarray, coeffs: np.ndarray,
                            boxes: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        """Decode YOLO mask coefficients using mask prototypes.

        Args:
            proto: Mask prototypes [32, proto_h, proto_w] (e.g. 32x160x160)
            coeffs: Per-detection mask coefficients [N, 32]
            boxes: Detection boxes in original image space [N, 4]
            orig_h, orig_w: Original image dimensions
        """
        try:
            proto_h, proto_w = proto.shape[1], proto.shape[2]

            # Matrix multiply: [N, 32] @ [32, proto_h*proto_w] → [N, proto_h*proto_w]
            masks_raw = coeffs @ proto.reshape(proto.shape[0], -1)
            masks_raw = masks_raw.reshape(-1, proto_h, proto_w)

            # Sigmoid activation
            masks_raw = 1.0 / (1.0 + np.exp(-masks_raw))

            # Crop each mask to its bounding box (in proto space), then resize
            n = len(boxes)
            masks = np.zeros((n, orig_h, orig_w), dtype=np.uint8)

            scale_x = proto_w / orig_w
            scale_y = proto_h / orig_h

            for i in range(n):
                # Box in proto space
                px1 = int(boxes[i, 0] * scale_x)
                py1 = int(boxes[i, 1] * scale_y)
                px2 = int(boxes[i, 2] * scale_x)
                py2 = int(boxes[i, 3] * scale_y)

                px1 = max(0, min(px1, proto_w))
                py1 = max(0, min(py1, proto_h))
                px2 = max(px1 + 1, min(px2, proto_w))
                py2 = max(py1 + 1, min(py2, proto_h))

                # Zero out mask outside the box
                mask_i = masks_raw[i].copy()
                mask_i[:py1, :] = 0
                mask_i[py2:, :] = 0
                mask_i[:, :px1] = 0
                mask_i[:, px2:] = 0

                # Resize to original image and threshold
                mask_resized = cv2.resize(mask_i, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                masks[i] = (mask_resized > 0.5).astype(np.uint8)

            return masks
        except Exception as e:
            logger.warning(f"Mask processing failed: {e}")
            return None

    # ── RF-DETR Postprocessing ───────────────────────────────────────────────

    def _postprocess_rfdetr(self, outputs: List[np.ndarray], orig_w: int, orig_h: int) -> sv.Detections:
        # Find outputs by name
        out_map = {name: outputs[i] for i, name in enumerate(self.output_names)}

        bboxes = out_map.get('dets', outputs[0])[0]    # [N, 4] center format, normalized
        logits = out_map.get('labels', outputs[1])[0]   # [N, num_classes] raw logits
        mask_out = out_map.get('masks', outputs[2] if len(outputs) > 2 else None)  # [1, N, mh, mw]

        # Apply sigmoid to convert logits → probabilities
        probs = 1.0 / (1.0 + np.exp(-logits))
        scores = np.max(probs, axis=1)
        class_ids = np.argmax(probs, axis=1)

        # Filter by confidence
        keep = scores > self.conf_threshold
        if not np.any(keep):
            return sv.Detections.empty()

        bboxes = bboxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # Convert center format [cx, cy, w, h] → corner [x1, y1, x2, y2]
        cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        boxes_xyxy = np.column_stack([
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
        ])

        # Scale from normalized (0-1) to original image space
        boxes_xyxy[:, [0, 2]] *= orig_w
        boxes_xyxy[:, [1, 3]] *= orig_h

        # Clip
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        # Process masks if available
        masks = None
        if mask_out is not None:
            masks = self._process_rfdetr_masks(mask_out[0], keep, orig_h, orig_w, boxes_xyxy)

        # Map class IDs: RF-DETR uses COCO 91-class output, our classes are 0=carton, 1=polybag
        # The model was fine-tuned on 2 classes, but the ONNX output keeps the 91-class head.
        # Class mapping: the fine-tuned model places our classes at specific indices.
        # We remap to 0-indexed by keeping the argmax as-is for now.

        return sv.Detections(
            xyxy=boxes_xyxy,
            confidence=scores,
            class_id=class_ids.astype(int),
            mask=masks,
        )

    def _process_rfdetr_masks(self, mask_preds: np.ndarray, keep: np.ndarray,
                               orig_h: int, orig_w: int, boxes: np.ndarray) -> np.ndarray:
        """Process RF-DETR per-detection mask predictions.

        Args:
            mask_preds: [N_all, mask_h, mask_w] raw mask predictions
            keep: Boolean mask for which detections passed confidence threshold
            orig_h, orig_w: Original image dimensions
            boxes: Filtered detection boxes in original image space
        """
        try:
            mask_preds = mask_preds[keep]  # [N_kept, mask_h, mask_w]
            n = len(mask_preds)
            masks = np.zeros((n, orig_h, orig_w), dtype=np.uint8)

            for i in range(n):
                mask_i = 1.0 / (1.0 + np.exp(-mask_preds[i]))  # sigmoid
                mask_resized = cv2.resize(mask_i, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                masks[i] = (mask_resized > 0.5).astype(np.uint8)

            return masks
        except Exception as e:
            logger.warning(f"RF-DETR mask processing failed: {e}")
            return None

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


# Alias for backward compatibility
ONNXInferencer = OptimizedONNXInferencer

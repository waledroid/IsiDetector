# src/shared/vision_engine.py
import cv2
import time
import logging
import supervision as sv
import numpy as np
from datetime import datetime
from src.utils.analytics_logger import DailyLogger

logger = logging.getLogger(__name__)

class VisionEngine:
    """Unified orchestrator for detection, tracking, counting, and annotation.

    The single stateful object used by both the CLI scripts and the
    Flask web app. Wraps a model-agnostic inferencer and adds:

    - **ByteTrack** persistent object tracking (IDs survive occlusion).
    - **LineZone** crossing detection — counts each object exactly once
      per session.
    - **DailyLogger** hourly CSV snapshots (auto-rotates at midnight).
    - Supervision annotators for masks, traces, bounding boxes, and labels.

    The engine is model-agnostic: pass any inferencer that implements
    :class:`~src.inference.base_inferencer.BaseInferencer`.

    Args:
        inferencer: An instance of ``YOLOInferencer``,
            ``RFDETRInferencer``, or ``OptimizedONNXInferencer``.
        config: The full ``train.yaml`` config dict, used for confidence
            threshold, tracker parameters, and logging settings.

    Example:
        ```python
        from src.inference.onnx_inferencer import OptimizedONNXInferencer
        from src.shared.vision_engine import VisionEngine

        engine = VisionEngine(
            inferencer=OptimizedONNXInferencer("best.onnx"),
            config=config
        )
        annotated, detections, event = engine.process_frame(frame, counts)
        if event:
            print(f"{event['class']} crossed the line (ID #{event['id']})")
        ```
    """

    def __init__(self, inferencer, config: dict):
        self.inferencer = inferencer
        self.config = config
        
        # 1. Pipeline Config
        inf_cfg = config.get('inference', {})
        conf_thresh = inf_cfg.get('conf_threshold', 0.3)
        track_cfg = inf_cfg.get('tracker', {})
        
        # 2. Tracking & Core Analytics
        self.tracker = sv.ByteTrack(
            track_activation_threshold=conf_thresh,
            lost_track_buffer=track_cfg.get('track_buffer', 60),
            minimum_matching_threshold=track_cfg.get('match_thresh', 0.9)
        )
        
        # 3. Line Counting Logic (Stateful)
        self.line_zone = None 
        self.counted_ids = set()
        
        # 4. Logging & Telemetry
        log_cfg = inf_cfg.get('logging', {})
        self.class_names_list = list(self.inferencer.class_names.values())
        self.logger = DailyLogger(
            class_names=self.class_names_list,
            log_dir=log_cfg.get('log_dir', 'logs'),
            save_interval=log_cfg.get('save_interval', 3600)
        )

        # 5. Visual Annotators
        self.mask_annotator = sv.MaskAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=0, text_scale=0)

    def init_line(self, width, height, x_percent=0.65):
        """Initializes the counting line based on frame dimensions."""
        line_x = int(width * x_percent)
        self.line_zone = sv.LineZone(
            start=sv.Point(line_x, 0), 
            end=sv.Point(line_x, height),
            triggering_anchors=[sv.Position.BOTTOM_CENTER]
        )

    def process_frame(self, frame: np.ndarray, class_totals: dict):
        """Run detection, tracking, and line-crossing counting on one frame.

        Called once per frame from the inference thread. Thread-safe
        provided it is called from a single thread (the inference loop
        owns it exclusively).

        Args:
            frame: Raw BGR frame as a NumPy array (from OpenCV).
            class_totals: Dict updated **in-place** with crossing counts
                (e.g. ``{'carton': 12, 'polybag': 5}``). Owned by
                ``StreamHandler``; do not pass the same dict from
                multiple threads simultaneously.

        Returns:
            A 3-tuple ``(annotated, detections, event)``:

            - ``annotated``: BGR frame with masks, traces, boxes, labels,
              and the counting line drawn.
            - ``detections``: ``sv.Detections`` object for the current
              frame (tracker IDs assigned).
            - ``event``: ``None``, or ``{"class": str, "id": int}`` when
              a new object crosses the line this frame.

        Note:
            ``counted_ids`` is automatically pruned when it exceeds
            50,000 entries (prevents unbounded memory growth in
            sessions longer than 8–12 hours).
        """
        if self.line_zone is None:
            h, w = frame.shape[:2]
            self.init_line(w, h)

        # 1. Core AI Logic (timed for performance dashboard)
        _t0 = time.perf_counter()
        frame_input = np.ascontiguousarray(frame)
        detections = self.inferencer.predict_frame(frame_input)
        _t1 = time.perf_counter()
        detections = self.tracker.update_with_detections(detections)
        _t2 = time.perf_counter()
        self.last_timing = {
            'forward_ms': (_t1 - _t0) * 1000,
            'tracker_ms': (_t2 - _t1) * 1000,
        }
        
        # 2. Crossing/Trigger Logic
        in_cross, out_cross = self.line_zone.trigger(detections=detections)
        all_crossings = in_cross | out_cross
        
        new_event = None
        for i, crossed in enumerate(all_crossings):
            if crossed and detections.tracker_id is not None:
                t_id = detections.tracker_id[i]
                if t_id not in self.counted_ids:
                    class_id = detections.class_id[i]
                    name = self.inferencer.class_names.get(class_id, "object")
                    class_totals[name] = class_totals.get(name, 0) + 1
                    self.counted_ids.add(t_id)
                    new_event = {"class": name, "id": t_id}

        # Prune counted_ids to prevent unbounded growth over long shifts.
        # ByteTrack IDs are monotonically increasing — old IDs are never reassigned,
        # so it is safe to discard the lower half once the set gets large.
        if len(self.counted_ids) > 50_000:
            sorted_ids = sorted(self.counted_ids)
            self.counted_ids = set(sorted_ids[len(sorted_ids) // 2:])
            logger.info(f"♻️ counted_ids pruned to {len(self.counted_ids)} entries")

        # 3. Update Hourly CSV Log
        self.logger.update(class_totals)

        # 4. Visual Composition
        annotated = frame.copy()
        if detections.mask is not None:
            annotated = self.mask_annotator.annotate(scene=annotated, detections=detections)
        
        annotated = self.trace_annotator.annotate(scene=annotated, detections=detections)
        annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
        
        if detections.tracker_id is not None:
            labels = [f"#{t_id} {self.inferencer.class_names.get(c_id, 'obj')}" 
                      for c_id, t_id in zip(detections.class_id, detections.tracker_id)]
            annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        
        annotated = self.line_annotator.annotate(frame=annotated, line_counter=self.line_zone)
        
        return annotated, detections, new_event

    def cleanup(self, class_totals):
        """Safe shutdown."""
        self.logger.save(class_totals, auto=False)

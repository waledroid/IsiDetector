# src/shared/vision_engine.py
import cv2
import logging
import supervision as sv
import numpy as np
from datetime import datetime
from src.utils.analytics_logger import DailyLogger

logger = logging.getLogger(__name__)

class VisionEngine:
    """
    Unified Vision Logic for Counting, Tracking, and Annotation.
    Separates 'Smart Object Detection' from 'Data Presentation'.
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
        """
        The Heavy-Lifter: Runs Detection, Tracking, and Counting in one pass.
        :param frame: Raw RGB/BGR frame
        :param class_totals: External dictionary to update with counts
        :return: (Annotated Frame, Detection Results, Crossings)
        """
        if self.line_zone is None:
            h, w = frame.shape[:2]
            self.init_line(w, h)

        # 1. Core AI Logic
        frame_input = np.ascontiguousarray(frame)
        detections = self.inferencer.predict_frame(frame_input)
        detections = self.tracker.update_with_detections(detections)
        
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

# src/shared/vision_engine.py
import cv2
import time
import logging
import supervision as sv
import numpy as np
from datetime import datetime
from src.utils.event_logger import EventLogger
from src.shared.dedup_gate import DedupGate
from src.shared.crossing import CrossingDetector

logger = logging.getLogger(__name__)

class VisionEngine:
    """Unified orchestrator for detection, tracking, counting, and annotation.

    The single stateful object used by both the CLI scripts and the
    Flask web app. Wraps a model-agnostic inferencer and adds:

    - **ByteTrack** persistent object tracking (IDs survive occlusion).
    - **LineZone** crossing detection — counts each object exactly once
      per session.
    - **EventLogger** per-crossing CSV rows (auto-rotates at midnight,
      keeps 30 days of history by default).
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
        # ByteTrack params now live under config['bytetrack'] (mode-driven, from
        # isidet/configs/inference/{cpu,gpu}.yaml). Fall back to the legacy
        # config['inference']['tracker'] location if a caller still uses the
        # old train.yaml shape.
        inf_cfg = config.get('inference', {})
        conf_thresh = inf_cfg.get('conf_threshold', 0.3)
        track_cfg = config.get('bytetrack') or inf_cfg.get('tracker', {})

        # 2. Tracking & Core Analytics
        _fr = track_cfg.get('frame_rate')
        _bt_kwargs = dict(
            track_activation_threshold=conf_thresh,
            lost_track_buffer=track_cfg.get('track_buffer', 60),
            minimum_matching_threshold=track_cfg.get('match_thresh', 0.9),
        )
        if _fr:                      # 0/None => let supervision default (30)
            _bt_kwargs['frame_rate'] = int(round(float(_fr)))
        self.tracker = sv.ByteTrack(**_bt_kwargs)
        self._tracker_kwargs = _bt_kwargs    # remembered for live rebuilds
        
        # 3. Line Counting Logic (Stateful)
        self.line_zone = None
        # Dedup: track-ID base (always on) + optional time guard (operator toggle).
        _inf = config.get('inference', {})
        self.dedup = DedupGate(
            time_enabled=bool(_inf.get('dedup_time_enabled', True)),
            interval_ms=int(_inf.get('dedup_interval_ms', 300)),
        )
        # Recall booster — OR-ed with LineZone, shares the dedup/seq path.
        self.count_interpolate = bool(_inf.get('count_interpolate', True))
        self.crossing = CrossingDetector()
        self._emit_seq = 0   # monotonic per-emitted-crossing counter (== UDP seq, == CSV seq)
        
        # 4. Logging & Telemetry — one CSV row per line crossing.
        # The events subdir keeps them separate from legacy snapshot logs
        # and is the dir the /api/chart endpoint reads from.
        log_cfg = inf_cfg.get('logging', {})
        self.class_names_list = list(self.inferencer.class_names.values())
        base_log_dir = log_cfg.get('log_dir', 'logs')
        retention_days = int(log_cfg.get('retention_days', 30))
        self.event_logger = EventLogger(
            log_dir=f"{base_log_dir.rstrip('/')}/events",
            retention_days=retention_days,
        )

        # 5. Visual Annotators
        self.mask_annotator = sv.MaskAnnotator(opacity=0.3)
        self.trace_annotator = sv.TraceAnnotator(thickness=1, trace_length=30)
        self.box_annotator = sv.BoxAnnotator(thickness=1)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.3, text_thickness=1, text_padding=2)
        # display_in_count / display_out_count = False removes the small badge
        # rectangle supervision renders at the line's midpoint. Counts are
        # already visible in the UI footer and /api/stats.
        self.line_annotator = sv.LineZoneAnnotator(
            thickness=1,
            text_thickness=0,
            text_scale=0,
            display_in_count=False,
            display_out_count=False,
        )

        # 6. Line configuration (can be overridden before first frame)
        self.line_orientation = 'vertical'
        self.line_position = 0.5
        self.belt_direction = 'left_to_right'
        self._frame_w = 0
        self._frame_h = 0

    # Belt direction → leading-edge anchor map. The leading edge is the side
    # of the bbox that enters the line zone FIRST given the belt's motion.
    # Using the leading edge maximises the sorter's reaction window.
    _ANCHOR_MAP = {
        ('vertical',   'left_to_right'): sv.Position.CENTER_RIGHT,
        ('vertical',   'right_to_left'): sv.Position.CENTER_LEFT,
        ('horizontal', 'top_to_bottom'): sv.Position.BOTTOM_CENTER,
        ('horizontal', 'bottom_to_top'): sv.Position.TOP_CENTER,
    }

    def swap_inferencer(self, new_inferencer):
        """Replace the model without losing counts, tracker IDs, or line state.

        Rebuilds palette-dependent annotators against the new inferencer's
        class_names so per-class colours reflect the new model's convention
        (YOLO 0-indexed vs RF-DETR 1-indexed). Tracker, dedup gate,
        line_zone, and event_logger are preserved so hot-swap keeps
        counts and per-event history intact.
        """
        self.inferencer = new_inferencer
        self.class_names_list = list(new_inferencer.class_names.values())

        # Rebuild only the annotators whose palette is indexed by class_id.
        # Trace and line annotators don't depend on class_id — leave them.
        self.mask_annotator = sv.MaskAnnotator(opacity=0.3)
        self.box_annotator = sv.BoxAnnotator(thickness=1)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.3, text_thickness=1, text_padding=2,
        )

    def configure_counting(self, count_interpolate=None):
        """Live-toggle the recall recovery without tearing down session state.

        Preserves counts/tracker/line/dedup. Only flips the interpolation flag;
        CrossingDetector latch state is preserved so in-flight tracks keep their
        before/after history across the toggle.
        """
        if count_interpolate is not None:
            self.count_interpolate = bool(count_interpolate)

    def init_line(self, width, height, position=0.5, orientation='vertical',
                   belt_direction=None):
        """Initializes the counting line based on frame dimensions.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            position: Line position as a fraction (0.1–0.9). For vertical
                lines this is the x-offset; for horizontal, the y-offset.
            orientation: ``'vertical'`` (default) or ``'horizontal'``.
            belt_direction: One of ``'left_to_right'``, ``'right_to_left'``,
                ``'top_to_bottom'``, ``'bottom_to_top'``. Determines which
                side of the bbox is the leading edge and therefore the
                trigger anchor. If ``None``, uses ``self.belt_direction``.
        """
        self.line_position = position
        self.line_orientation = orientation
        if belt_direction is not None:
            self.belt_direction = belt_direction
        self._frame_w = width
        self._frame_h = height

        if orientation == 'horizontal':
            line_y = int(height * position)
            start = sv.Point(0, line_y)
            end = sv.Point(width, line_y)
        else:
            line_x = int(width * position)
            start = sv.Point(line_x, 0)
            end = sv.Point(line_x, height)

        # Trigger anchor. Default 'leading_edge': fire on the FRONT of the parcel
        # ("début de l'objet") — the PLC's timing correlation expects the send at the
        # leading edge (size-independent timing), not the centre (which fires later).
        # 'center' (bbox centre) is an opt-in alternative.
        _anchor_name = self.config.get('inference', {}).get('trigger_anchor', 'leading_edge')
        if _anchor_name == 'leading_edge':
            anchor = self._ANCHOR_MAP.get(
                (orientation, self.belt_direction),
                sv.Position.BOTTOM_CENTER,
            )
        else:
            anchor = sv.Position.CENTER

        # Geometry the CrossingDetector needs (axis + line coord + belt "after" side).
        self._trigger_anchor = anchor
        if orientation == 'horizontal':
            self._cross_axis = 1            # compare anchor y
            self._line_coord = float(line_y)
            self._after_is_greater = (self.belt_direction == 'top_to_bottom')
        else:
            self._cross_axis = 0            # compare anchor x
            self._line_coord = float(line_x)
            self._after_is_greater = (self.belt_direction == 'left_to_right')

        self.line_zone = sv.LineZone(
            start=start, end=end,
            triggering_anchors=[anchor],
        )
        logger.info(f"[LINE] {orientation} @ {position:.2f} ({_anchor_name}) "
                    f"for frame {width}x{height}")

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
            - ``events``: list of ``{"class": str, "id": int}`` dicts —
              one entry per new line crossing this frame. Empty list if
              no crossings. Multiple entries if several objects crossed
              in the same frame (close-together on the belt).

        Note:
            The dedup gate's ``counted_ids`` is automatically pruned when it
            exceeds 50,000 entries (prevents unbounded memory growth in
            sessions longer than 8–12 hours).
        """
        # (Re)build the line whenever the incoming frame size changes — this is what
        # makes line_position adapt to an ROI crop. The ROI crop (+ downscale) changes
        # the frame dimensions; without this, the line keeps the pixel geometry from
        # the first/pre-crop frame and no longer sits at line_position of the new crop.
        # Dims are stable for a fixed ROI+source, so this only fires on an actual
        # ROI/source change (no per-frame churn, no crossing-state thrash).
        h, w = frame.shape[:2]
        if self.line_zone is None or (w, h) != (self._frame_w, self._frame_h):
            self.init_line(w, h, self.line_position, self.line_orientation)

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

        # Recall recovery: OR in the frame-gap-tolerant latch on the leading-edge
        # anchor. Same dedup/seq path below, so a crossing caught by both counts once.
        if self.count_interpolate and detections.tracker_id is not None and len(detections):
            anchors = detections.get_anchors_coordinates(self._trigger_anchor)
            ids = [int(t) for t in detections.tracker_id]
            coords = [float(a[self._cross_axis]) for a in anchors]
            recovered = self.crossing.update(ids, coords, self._line_coord,
                                              self._after_is_greater)
            if recovered:
                all_crossings = all_crossings | np.array(
                    [t in recovered for t in ids], dtype=bool)
            self.crossing.forget(keep_ids=ids)

        # Collect EVERY new crossing this frame — the caller publishes one
        # UDP datagram per event so the sorter never misses a trigger when
        # two close-together objects cross in the same frame.
        new_events = []
        now_ms = time.monotonic() * 1000.0   # one clock per frame; within-frame ties share it
        for i, crossed in enumerate(all_crossings):
            if crossed and detections.tracker_id is not None:
                t_id = int(detections.tracker_id[i])
                if self.dedup.should_emit(t_id, now_ms):
                    class_id = int(detections.class_id[i])
                    name = self.inferencer.class_names.get(class_id, "object")
                    class_totals[name] = class_totals.get(name, 0) + 1
                    self.dedup.record(t_id, now_ms)
                    # One monotonic seq per emitted crossing — the SAME value the
                    # UDP publisher sends and the event log records, so the CSV can
                    # be reconciled against the wire. Gap-free per stream/engine.
                    self._emit_seq += 1
                    new_events.append({"class": name, "id": t_id, "seq": self._emit_seq})
                    self.event_logger.log(name, t_id, self._emit_seq)
                elif self.dedup.time_suppressed(t_id, now_ms):
                    logger.info(f"⏱️ dedup-suppressed crossing id={t_id} "
                                f"(<{self.dedup.interval_ms}ms since last emit)")

        # 3. Visual Composition
        # skip_masks / skip_traces are set by stream_handler._apply_render_settings
        # from the mode-driven inference config (cpu.yaml: both True; gpu.yaml: both False).
        # Default to False if the attribute isn't present (defensive — older callers
        # that don't go through _apply_render_settings get the full render).
        annotated = frame.copy()
        if detections.mask is not None and not getattr(self, 'skip_masks', False):
            annotated = self.mask_annotator.annotate(scene=annotated, detections=detections)

        if not getattr(self, 'skip_traces', False):
            annotated = self.trace_annotator.annotate(scene=annotated, detections=detections)
        annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
        
        if detections.tracker_id is not None:
            labels = [f"#{t_id} {self.inferencer.class_names.get(c_id, 'obj')}" 
                      for c_id, t_id in zip(detections.class_id, detections.tracker_id)]
            annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        
        annotated = self.line_annotator.annotate(frame=annotated, line_counter=self.line_zone)
        
        return annotated, detections, new_events

    def cleanup(self, class_totals):
        """Safe shutdown. EventLogger writes on every event, so there is
        nothing to flush here; method kept for API stability."""
        return

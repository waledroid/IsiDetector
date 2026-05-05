import os
import cv2
import time
import threading
import logging
import socket
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


# Import inferencers lazily — rfdetr triggers heavy CUDA init at import time
# which can segfault in Docker if loaded before the app needs it.
# YOLOInferencer, RFDETRInferencer, ONNXInferencer, OpenVINOInferencer,
# TensorRTInferencer are all imported inside start() when needed.
from src.shared.vision_engine import VisionEngine
from isitec_api.performance_monitor import PerformanceMonitor

logger = logging.getLogger("IsiDetector-Web")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class UDPPublisher:
    """Broadcasts line-crossing events via UDP to the sorting machine controller.

    Fires a ~60-byte JSON datagram on every line-crossing event::

        {"class": "carton", "ts": "2026-03-31T14:23:45.312847"}

    A single UDP socket is created at construction and reused for
    every event — no connection overhead per datagram.

    The target can be re-pointed at runtime via :meth:`update_target`
    without restarting the stream (``POST /api/udp``).

    Args:
        host: IP address of the sorting controller. Defaults to
            ``"127.0.0.1"`` (same machine).
        port: UDP port the controller is listening on. Defaults to
            ``9502``.
        enabled: Set ``False`` to disable all publishing without
            removing the publisher.

    Configuration priority (highest wins):

    1. ``POST /api/udp`` at runtime.
    2. ``UDP_HOST`` / ``UDP_PORT`` environment variables.
    3. ``isidet/configs/train.yaml`` → ``inference.udp.host / port``.
    4. Hardcoded defaults (``127.0.0.1:9502``).
    """

    def __init__(self, host="127.0.0.1", port=9502, enabled=True):
        self.host = host
        self.port = port
        self.enabled = enabled
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, class_name, event_id=None):
        """Emit one UDP datagram per line-crossing event.

        Optional ``event_id`` (tracker ID) is included in the payload as
        ``"id"`` so the sorter can dedupe and trace individual objects.
        Omitted when ``None`` — backward-compatible with consumers that
        only read ``class``.

        Returns:
            Trigger-to-wire latency in nanoseconds (JSON encode +
            ``sendto`` syscall), or ``0`` when publishing was skipped or
            failed. The PerformanceMonitor surfaces this as a histogram
            so the automation engineer can see the actual sort-trigger
            budget (p50 / p95 / p99 / max).
        """
        if not self.enabled:
            return 0
        t0 = time.perf_counter_ns()
        msg = {
            "class": str(class_name),
            "ts": datetime.datetime.now().isoformat(),
        }
        if event_id is not None:
            msg["id"] = int(event_id)
        payload = json.dumps(msg).encode()
        try:
            self._sock.sendto(payload, (self.host, self.port))
            elapsed = time.perf_counter_ns() - t0
            logger.debug(f"[UDP] → {self.host}:{self.port} | {payload.decode()} | {elapsed / 1000:.0f} µs")
            return elapsed
        except Exception as e:
            logger.warning(f"[UDP] Send failed: {e}")
            return 0

    def update_target(self, host, port):
        """Retarget to a different host/port at runtime — no socket recreation needed."""
        self.host = host
        self.port = int(port)
        logger.info(f"[UDP] Target updated → {self.host}:{self.port}")

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass

class LiveReader:
    """Background frame-reader thread with automatic reconnection.

    Reads frames from any OpenCV-compatible source (RTSP, USB webcam,
    MP4 file) in a dedicated daemon thread and exposes the latest frame
    via a ``Queue(maxsize=1)`` — the inference thread always gets the
    most recent frame, never a stale one.

    **Live sources** (RTSP, webcam): the hardware throttles the read
    rate naturally; no artificial delay is added.

    **File sources** (MP4, AVI): reads are paced to the video's native
    FPS (read from ``cv2.CAP_PROP_FPS``) to prevent fast-forward
    playback. Falls back to 30 fps if the container header is missing.

    On stream disconnection or read failure, the reader waits 1 second
    and attempts to reconnect automatically — no manual restart needed.

    Args:
        source: An RTSP URL string, integer webcam index, or file path.
    """

    def __init__(self, source):
        self.source = source
        self.cap = None
        self._cap_lock = threading.Lock()   # BUG 3 FIX: guard cap across threads
        self.q = queue.Queue(maxsize=1)
        self.running = True
        # For file sources, pace reads to the video's native FPS to prevent fast-forward.
        # Live sources (RTSP, webcam) are already throttled by the hardware.
        self.is_file = isinstance(source, str) and Path(source).is_file()
        self.frame_delay = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open_capture(self):
        """Open ``self.source`` with sensible defaults + RTSP fallback.

        For RTSP sources, prefer ``rtsp_transport=tcp`` (more reliable on
        noisy site LANs) and **fall back to UDP** if TCP can't open or
        decode a frame. Avoids on-site troubleshooting trips for the few
        cameras / firewalls where TCP is blocked.

        Caps the open timeout at 5 s via ``CAP_PROP_OPEN_TIMEOUT_MSEC``
        so a bad URL fails fast instead of hanging the inference thread
        for 30+ s.
        """
        is_rtsp = (
            isinstance(self.source, str)
            and self.source.lower().startswith(('rtsp://', 'rtspt://'))
        )
        if not is_rtsp:
            return cv2.VideoCapture(self.source)

        def _try_open():
            try:
                return cv2.VideoCapture(
                    self.source, cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000],
                )
            except Exception:
                return cv2.VideoCapture(self.source)

        # Attempt 1: TCP (preferred)
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        cap = _try_open()
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                logger.info("📡 RTSP transport: TCP (preferred)")
                return cap
            logger.warning("📡 RTSP TCP opened but first read failed; trying UDP fallback")
            cap.release()
        else:
            logger.warning("📡 RTSP TCP open failed; trying UDP fallback")

        # Attempt 2: UDP (fallback)
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
        cap = _try_open()
        if cap.isOpened():
            logger.info("📡 RTSP transport: UDP (TCP unavailable on this network)")
        else:
            logger.error("📡 RTSP failed to open on both TCP and UDP")
        # Restore TCP as default so subsequent reconnects retry the
        # preferred transport (in case the network heals).
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        return cap

    def _run(self):
        logger.info(f"🚀 LiveReader: Thread started for source {self.source}")
        while self.running:
            try:
                with self._cap_lock:
                    if self.cap is None or not self.cap.isOpened():
                        self.cap = self._open_capture()
                        if not self.cap.isOpened():
                            self.cap = None
                            logger.error(f"❌ LiveReader: Failed to open {self.source}. Retrying in 2s...")
                            time.sleep(2)
                            continue
                        # Cap OpenCV's internal queue at 1 frame so the reader
                        # always returns the most-recent frame from the network.
                        # Default 5 → up to ~165 ms of stale frames pile up on
                        # RTSP. CPU-only sites can't afford the wasted decode.
                        try:
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                        logger.info(f"✅ LiveReader: Connection established to {self.source}")

                        # Stream info on connect — confirms what the camera is
                        # actually sending (resolution, native fps, codec).
                        # Saves an `ffprobe` round-trip during on-site debugging.
                        try:
                            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                            fps_native = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
                            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC) or 0)
                            codec = bytes(
                                [(fourcc >> shift) & 0xFF for shift in (0, 8, 16, 24)]
                            ).decode(errors='replace').strip('\x00') or '?'
                            logger.info(
                                f"📹 Stream: {w}×{h} @ {fps_native:.0f} fps codec={codec}"
                            )
                        except Exception:
                            pass

                        if self.is_file:
                            fps = self.cap.get(cv2.CAP_PROP_FPS)
                            self.frame_delay = (1.0 / fps) if fps > 0 else (1.0 / 30)
                            logger.info(f"🎞️ LiveReader: File source detected — pacing to {fps:.1f} fps")

                    ret, frame = self.cap.read()
                    if not ret:
                        if self.is_file:
                            # EOF on file — loop cleanly without the 1s stutter or false "disconnect" warning.
                            self.cap.release()
                            self.cap = None
                            continue
                        logger.warning(f"⚠️ LiveReader: Stream disconnected. Retrying...")
                        self.cap.release()
                        self.cap = None
                        time.sleep(1)
                        continue

                if not self.q.empty():
                    try: self.q.get_nowait()
                    except: pass
                self.q.put(frame)

                # Pace to native FPS for file sources (lock NOT held during sleep)
                if self.frame_delay > 0:
                    time.sleep(self.frame_delay)
            except Exception as e:
                logger.error(f"🔥 LiveReader: Error: {e}")
                time.sleep(1)

    def get_frame(self):
        try: return self.q.get(timeout=0.2)
        except: return None

    def stop(self):
        logger.info("🛑 LiveReader: Stopping thread...")
        self.running = False
        with self._cap_lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

class StreamHandler:
    def __init__(self):
        self.running = False
        self.reader = None
        self.lock = threading.Lock()

        # Load Config first — other init steps depend on it
        # After bucket restructure this file is at webapp/isitec_api/stream_handler.py,
        # so parent.parent = webapp/ and parent.parent.parent = repo root. Config
        # lives at <repo>/isidet/configs/train.yaml.
        config_path = Path(__file__).resolve().parent.parent.parent / "isidet" / "configs" / "train.yaml"
        self.config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}

        self.imgsz = self.config.get('image_size', 640)
        self.web_imgsz = min(self.imgsz, 640)  # Safety cap for web throughput

        # Vision state
        self.engine = None
        self.class_totals = {}
        self.last_detected = None
        self.mode_text = ""
        self.language = 'fr'
        self.latest_annotated = None
        self.inf_thread = None
        self.source_is_image = False
        self.frame_ready = threading.Event()
        self._current_source = None  # track for hot-swap detection
        self.roi = None              # (x1,y1,x2,y2) bbox or None — loaded per start()
        # 2:1 JPEG-encode throttle so display FPS ≈ inference FPS / 2 (~12 FPS
        # at 25 FPS inference). Inference, tracking, line crossings, and UDP
        # publish still run on every frame — only the visualization is throttled.
        self._encode_skip_idx = 0

        # Cache the STANDBY frame once — 16:9 aspect ratio
        _standby_h = int(self.web_imgsz * 9 / 16)
        _blank = np.zeros((_standby_h, self.web_imgsz, 3), dtype=np.uint8)
        cv2.putText(_blank, "STANDBY", (int(self.web_imgsz / 4), int(_standby_h / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        _, _buf = cv2.imencode('.jpg', _blank)
        self._standby_frame = _buf.tobytes()

        # UDP Telemetry — config priority (highest wins):
        #   1. settings.json (operator-edited from Settings UI; persists)
        #   2. UDP_HOST / UDP_PORT env vars (compose / .env)
        #   3. train.yaml inference.udp.host / port
        #   4. Hardcoded default 127.0.0.1:9502
        udp_cfg = self.config.get('inference', {}).get('udp', {})
        env_host = os.environ.get('UDP_HOST', udp_cfg.get('host', '127.0.0.1'))
        env_port = int(os.environ.get('UDP_PORT', udp_cfg.get('port', 9502)))
        udp_host, udp_port = env_host, env_port
        try:
            settings_path = Path(__file__).parent / 'settings.json'
            if settings_path.exists():
                with open(settings_path) as f:
                    ui = json.load(f)
                if isinstance(ui.get('udp_host'), str) and ui['udp_host'].strip():
                    udp_host = ui['udp_host'].strip()
                if isinstance(ui.get('udp_port'), int) and 1 <= ui['udp_port'] <= 65535:
                    udp_port = ui['udp_port']
        except Exception:
            pass
        udp_enabled = udp_cfg.get('enabled', True)
        self.publisher = UDPPublisher(host=udp_host, port=udp_port, enabled=udp_enabled)
        logger.info(f"[UDP] Publisher ready → {self.publisher.host}:{self.publisher.port}")

        # Performance Monitor — collects real-time metrics for /api/performance
        self.monitor = PerformanceMonitor()

        # Mode detection + inference-config loading. Mode is one of "cpu"|"gpu",
        # determined from COMPOSE_MODE env var (set by up.sh / docker-compose),
        # falling back to nvidia-smi probe and finally to "cpu" as the safe
        # default. The mode picks which YAML in isidet/configs/inference/
        # gets layered on top of common.yaml.
        mode_info = self._detect_mode()
        self.mode = mode_info['mode']
        self.mode_detected_via = mode_info['detected_via']
        self.inference_config, self.inference_config_files = self._load_inference_config(self.mode)
        logger.info(
            f"[mode] {self.mode} (detected_via: {self.mode_detected_via}) — "
            f"loaded {' + '.join(self.inference_config_files)}"
        )

        # Background preload of the default RF-DETR ONNX. Skipped in CPU mode
        # (RF-DETR isn't supported on CPU at all).
        if self.mode == "gpu":
            threading.Thread(target=self._preload_default_onnx, daemon=True).start()

        # Boot-time auto-start: if the operator ticked "Auto-start on boot"
        # in Settings → Camera, replay the last successful start() so the
        # site PC is operator-ready without anyone clicking Start.
        threading.Thread(target=self._maybe_auto_start, daemon=True).start()

    def _settings_path(self):
        return Path(__file__).parent / 'settings.json'

    def _detect_mode(self):
        """Resolve the operating mode (cpu | gpu) at container boot.

        Priority: COMPOSE_MODE env > nvidia-smi probe > safe fallback (cpu).
        Returns a dict {mode, detected_via} for the /api/mode endpoint.
        """
        env_mode = os.environ.get('COMPOSE_MODE', '').strip().lower()
        if env_mode in ('cpu', 'gpu'):
            return {'mode': env_mode, 'detected_via': 'COMPOSE_MODE env'}
        try:
            import subprocess as _sp
            result = _sp.run(['nvidia-smi'], capture_output=True, timeout=2)
            if result.returncode == 0:
                return {'mode': 'gpu', 'detected_via': 'nvidia-smi probe'}
        except Exception:
            pass
        return {'mode': 'cpu', 'detected_via': 'fallback (no GPU detected)'}

    def _load_inference_config(self, mode: str):
        """Load isidet/configs/inference/common.yaml + the mode-specific yaml.
        Returns (merged_config_dict, list_of_loaded_filenames).
        """
        cfg_dir = Path(__file__).resolve().parent.parent.parent / 'isidet' / 'configs' / 'inference'
        common_path = cfg_dir / 'common.yaml'
        mode_path = cfg_dir / f'{mode}.yaml'

        merged = {}
        loaded = []

        def deep_merge(base, overlay):
            for k, v in overlay.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    deep_merge(base[k], v)
                else:
                    base[k] = v

        for label, path in [('common.yaml', common_path), (f'{mode}.yaml', mode_path)]:
            try:
                if path.exists():
                    with open(path) as f:
                        data = yaml.safe_load(f) or {}
                    deep_merge(merged, data)
                    loaded.append(label)
                else:
                    logger.warning(f"[mode] inference config not found: {path}")
            except Exception as e:
                logger.warning(f"[mode] failed to load {path}: {e}")

        return merged, loaded

    def _persist_last_used(self, model_type: str, weights: str):
        """Write the last successful (model_type, weights) back to settings.json
        so a subsequent container boot can replay them via _maybe_auto_start.
        """
        try:
            path = self._settings_path()
            current = {}
            if path.exists():
                with open(path) as f:
                    current = json.load(f) or {}
            current['last_model_type'] = str(model_type or '')
            current['last_weights'] = str(weights or '')
            with open(path, 'w') as f:
                json.dump(current, f, indent=2)
        except Exception as e:
            logger.warning(f"[auto-start] Could not persist last-used selection: {e}")

    def _load_roi(self):
        """Read roi_enabled + roi_points from settings.json into self.roi.
        Always sets self.roi to either a 4-tuple (x1,y1,x2,y2) bbox or None.
        Any error → None + log; never raises.
        """
        self.roi = None
        try:
            path = self._settings_path()
            if not path.exists():
                return
            with open(path) as f:
                ui = json.load(f) or {}
            if not ui.get('roi_enabled'):
                return
            pts = ui.get('roi_points')
            if not (isinstance(pts, list) and len(pts) == 4):
                return
            xs = [int(p[0]) for p in pts]
            ys = [int(p[1]) for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if x2 > x1 and y2 > y1:
                self.roi = (x1, y1, x2, y2)
                logger.info(f"[ROI] Active crop: x=[{x1},{x2}] y=[{y1},{y2}] "
                            f"({x2-x1}×{y2-y1})")
        except Exception as e:
            logger.warning(f"[ROI] Could not load — falling back to full frame: {e}")
            self.roi = None

    def get_raw_snapshot(self):
        """Return the latest pre-crop, pre-resize, pre-annotation frame as JPEG.
        Used by the Live-Inference Set-ROI configurator.
        """
        if not self.running or self.reader is None:
            return None
        try:
            frame = self.reader.get_frame()
            if frame is None:
                return None
            ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return buf.tobytes() if ok else None
        except Exception as e:
            logger.warning(f"[snapshot] failed: {e}")
            return None

    def _maybe_auto_start(self):
        """If auto_start is enabled and we have a saved camera + last-used
        model, kick off start() once the HTTP server is up. No-op otherwise.
        """
        # Wait for the ASGI server to bind. 5 s is enough on this hardware
        # and short enough that the operator sees the stream come up while
        # the kiosk Chrome window is still finishing its first paint.
        time.sleep(5.0)
        try:
            path = self._settings_path()
            if not path.exists():
                return
            with open(path) as f:
                ui = json.load(f) or {}
            if not ui.get('auto_start'):
                return
            rtsp_url = (ui.get('rtsp_url') or '').strip()
            model_type = (ui.get('last_model_type') or '').strip()
            weights = (ui.get('last_weights') or '').strip()
            if not (rtsp_url and model_type and weights):
                logger.info("[auto-start] Enabled but missing rtsp_url / last_model_type / "
                            "last_weights — skipping. Click Start once to record them.")
                return
            logger.info(f"[auto-start] Replaying last session: model_type={model_type} weights={weights}")
            ok, msg = self.start(source="", model_type=model_type, weights=weights)
            if ok:
                logger.info(f"[auto-start] {msg}")
            else:
                logger.warning(f"[auto-start] failed: {msg}")
        except Exception as e:
            logger.warning(f"[auto-start] error: {e}")

    def _preload_default_onnx(self):
        try:
            settings_path = Path(__file__).parent / 'settings.json'
            if not settings_path.exists():
                return
            with open(settings_path) as f:
                settings = json.load(f)
            path = settings.get('rfdetr_weights')
            if not path or not str(path).lower().endswith('.onnx'):
                return
            abs_path = Path(path)
            if not abs_path.is_absolute():
                # Paths in settings.json are relative to the repo root.
                # webapp/isitec_api/stream_handler.py → parent.parent.parent = repo root.
                abs_path = Path(__file__).resolve().parent.parent.parent / path
            if not abs_path.exists():
                return
            from src.inference.onnx_inferencer import preload_onnx
            preload_onnx(str(abs_path))
        except Exception as e:
            logger.warning(f"RF-DETR ONNX preload skipped: {e}")

    def set_language(self, lang):
        self.language = lang

    def set_belt_status(self, active: bool):
        with self.lock:
            self.monitor.belt_active = active

    def set_udp_target(self, host, port):
        self.publisher.update_target(host, int(port))

    def get_udp_target(self):
        return {"host": self.publisher.host, "port": self.publisher.port, "enabled": self.publisher.enabled}

    def set_line_config(self, orientation=None, position=None, belt_direction=None):
        """Update line orientation / position / belt direction live.

        Re-initializes the LineZone so the new trigger anchor (derived
        from orientation × belt_direction) takes effect immediately
        without restarting the stream.
        """
        with self.lock:
            engine = self.engine
            if not engine:
                return
            if orientation is not None:
                engine.line_orientation = orientation
            if position is not None:
                engine.line_position = max(0.1, min(0.9, float(position)))
            if belt_direction is not None:
                engine.belt_direction = belt_direction
            if engine.line_zone is not None:
                engine.init_line(engine._frame_w, engine._frame_h,
                                 engine.line_position, engine.line_orientation,
                                 belt_direction=engine.belt_direction)

    def get_line_config(self):
        with self.lock:
            engine = self.engine
            if engine:
                return {
                    "orientation": engine.line_orientation,
                    "position": engine.line_position,
                    "belt_direction": engine.belt_direction,
                }
        return {"orientation": "vertical", "position": 0.5, "belt_direction": "left_to_right"}

    def _apply_line_settings(self, engine):
        """Load line settings from settings.json and apply to engine."""
        settings_path = Path(__file__).parent / 'settings.json'
        try:
            if settings_path.exists():
                with open(settings_path) as f:
                    settings = json.load(f)
                engine.line_orientation = settings.get('line_orientation', 'vertical')
                engine.line_position = settings.get('line_position', 0.5)
                engine.belt_direction = settings.get('belt_direction', 'left_to_right')
        except Exception:
            pass

    def _apply_render_settings(self, engine):
        """Apply render-perf flags from the mode-driven inference config.

        Sourced from inference_config['render'] (cpu.yaml: skip_masks/traces=true,
        gpu.yaml: skip_masks/traces=false). Not operator-tunable — these are
        hardware-class optimization knobs.
        """
        try:
            render_cfg = (self.inference_config or {}).get('render') or {}
            engine.skip_masks = bool(render_cfg.get('skip_masks', False))
            engine.skip_traces = bool(render_cfg.get('skip_traces', False))
            logger.info(
                f"VisionEngine render: skip_masks={engine.skip_masks} "
                f"skip_traces={engine.skip_traces}"
            )
        except Exception as e:
            logger.warning(f"[render] could not apply render settings: {e}")

    def get_stats(self):
        with self.lock:
            stats = {
                "is_running": self.running,
                "counts": self.class_totals,
                "last_detected": self.last_detected
            }
            return sanitize_for_json(stats)

    def get_performance(self) -> dict:
        """Return a full performance snapshot for ``GET /api/performance``.

        Builds on :meth:`PerformanceMonitor.get_snapshot` and injects
        live counting rate and running status which require access to
        ``StreamHandler`` state.
        """
        snapshot = self.monitor.get_snapshot()

        with self.lock:
            is_running = self.running
            totals = dict(self.class_totals)

        snapshot['session']['is_running'] = is_running

        elapsed = (time.time() - self.monitor.session_start) if self.monitor.session_start else 0
        snapshot['counting']['totals'] = totals
        if elapsed > 0:
            snapshot['counting']['rate_per_hour'] = {
                k: round(v / elapsed * 3600, 1) for k, v in totals.items()
            }
        else:
            snapshot['counting']['rate_per_hour'] = {}

        snapshot['counting']['status'] = self.monitor._status_counting(totals, is_running)

        return sanitize_for_json(snapshot)

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

    def _resolve_default_weights(self, model_type: str) -> str | None:
        """Resolve default model weights: settings.json → auto-discover → None."""
        import json as _json

        # 1. Try settings.json
        settings_path = Path(__file__).parent / 'settings.json'
        try:
            if settings_path.exists():
                with open(settings_path) as f:
                    settings = _json.load(f)
                is_detr = 'rfdetr' in model_type.lower()
                key = 'rfdetr_weights' if is_detr else 'yolo_weights'
                saved = settings.get(key, '')
                if saved and Path(saved).exists():
                    logger.info(f"Using saved default weights: {saved}")
                    return saved
        except Exception:
            pass

        # 2. Auto-discover weights — prioritize by hardware
        is_detr = 'rfdetr' in model_type.lower()
        has_gpu = self.monitor.has_gpu

        if is_detr:
            search_dirs = [Path('isidet/models/rfdetr')]
        else:
            search_dirs = [Path('isidet/runs/segment/models/yolo'), Path('isidet/models/yolo')]

        # Priority order: GPU prefers native/ONNX, CPU prefers OpenVINO
        if has_gpu:
            if is_detr:
                ext_priority = ['.pth', '.onnx', '.xml']
            else:
                ext_priority = ['.pt', '.onnx', '.xml']
        else:
            ext_priority = ['.xml', '.onnx', '.pth', '.pt']

        all_files = []
        for d in search_dirs:
            if d.exists():
                for ext in ext_priority:
                    all_files.extend(d.rglob(f'*{ext}'))

        if all_files:
            # Sort by: extension priority first, then newest within same priority
            def sort_key(p):
                ext = p.suffix.lower()
                priority = ext_priority.index(ext) if ext in ext_priority else 99
                return (priority, -p.stat().st_mtime)

            best = min(all_files, key=sort_key)
            logger.info(f"Auto-discovered weights: {best} ({'GPU' if has_gpu else 'CPU'} priority)")
            return str(best)

        # 3. Nothing found
        return None

    def _tune_annotators(self, engine):
        """Override VisionEngine annotators for web-resolution frames.

        The shared VisionEngine (src/shared/) uses hardcoded annotation
        sizes calibrated for full-resolution frames. At web_imgsz (416px),
        masks are too faint and the line marker is oversized. This method
        re-creates the annotators with resolution-appropriate settings
        without touching the shared code used by Flask.
        """
        sz = self.web_imgsz
        # Scale factor relative to 1080p baseline
        scale = sz / 1080

        engine.mask_annotator = sv.MaskAnnotator(opacity=0.5)
        engine.trace_annotator = sv.TraceAnnotator(
            thickness=max(1, round(2 * scale)),
            trace_length=30,
        )
        engine.box_annotator = sv.BoxAnnotator(
            thickness=max(1, round(2 * scale)),
        )
        engine.label_annotator = sv.LabelAnnotator(
            text_scale=max(0.25, round(0.5 * scale, 2)),
            text_thickness=max(1, round(2 * scale)),
            text_padding=max(2, round(4 * scale)),
        )
        engine.line_annotator = sv.LineZoneAnnotator(
            thickness=max(1, round(2 * scale)),
            text_thickness=0,
            text_scale=0,
            display_in_count=False,
            display_out_count=False,
        )
        logger.info(f"🎨 Annotators tuned for {sz}px (scale={scale:.2f}, mask_opacity=0.5)")

    def _build_engine(self, model_type, weights, imgsz, conf_thresh, device):
        """Build an inferencer by resolving weights and dispatching by file extension.

        Returns:
            ``(base_engine, mode_text)`` on success.

        Raises:
            ValueError: If weights not found or format incompatible.
            Exception: If model loading fails.
        """
        final_conf = conf_thresh if conf_thresh is not None else self.config.get('inference', {}).get('conf_threshold', 0.3)
        has_gpu = self.monitor.has_gpu
        device_label = "GPU" if has_gpu else "CPU"

        # OpenVINO + ByteTrack tunables come from the mode-driven inference config
        # (isidet/configs/inference/{common,cpu,gpu}.yaml), NOT from settings.json.
        ov_cfg = (self.inference_config or {}).get('openvino') or {}
        cpu_threads = ov_cfg.get('inference_num_threads')  # may be None on GPU mode

        model_path = weights
        if not model_path:
            model_path = self._resolve_default_weights(model_type)
            if model_path is None:
                raise ValueError(f"No model weights found for {model_type}. Go to Settings and select a model, or upload weights to the models/ folder.")

        ext = Path(model_path).suffix.lower()
        in_docker = Path('/.dockerenv').exists()

        # Mode-driven model allowlist. CPU mode rejects anything that isn't
        # an OpenVINO IR or ONNX (no .pt / .pth / .engine on CPU). GPU mode
        # accepts everything.
        model_cfg = (self.inference_config or {}).get('model') or {}
        allowed_exts = model_cfg.get('allowed_extensions')
        if allowed_exts and ext not in allowed_exts:
            raise ValueError(
                f"Model extension '{ext}' is not supported in {self.mode.upper()} mode. "
                f"Allowed in this mode: {', '.join(allowed_exts)}. "
                f"On CPU mode, use OpenVINO (.xml) or ONNX (.onnx) — re-export via the office workstation's compress.sh if needed."
            )
        allowed_families = model_cfg.get('allowed_families') or []
        if 'rfdetr' not in allowed_families and model_type == 'rfdetr':
            raise ValueError(
                f"RF-DETR is not supported in {self.mode.upper()} mode. "
                f"Switch to a YOLO model in Settings → Model 1."
            )

        if ext == '.engine' and not has_gpu:
            raise ValueError("TensorRT engines require an NVIDIA GPU. Use an OpenVINO (.xml) or ONNX (.onnx) model instead.")

        if ext == '.engine':
            from src.inference.tensorrt_inferencer import TensorRTInferencer
            base_engine = TensorRTInferencer(model_path=model_path, conf_threshold=final_conf, device=device, imgsz=imgsz)
            mode_text = f"TensorRT • {device_label}"
        elif ext == '.xml':
            from src.inference.openvino_inferencer import OpenVINOInferencer
            base_engine = OpenVINOInferencer(
                model_path=model_path, conf_threshold=final_conf,
                device=device, imgsz=imgsz, cpu_threads=cpu_threads,
                performance_hint=ov_cfg.get('performance_hint', 'LATENCY'),
                num_streams=ov_cfg.get('num_streams', 1),
            )
            mode_text = f"OpenVINO • CPU"
        elif ext == '.onnx':
            from src.inference.onnx_inferencer import ONNXInferencer
            onnx_device = device if device else ("cuda" if has_gpu else "cpu")
            base_engine = ONNXInferencer(model_path=model_path, conf_threshold=final_conf, device=onnx_device, imgsz=imgsz)
            mode_text = f"ONNX • {device_label}"
        elif ext == '.pth':
            if in_docker:
                from src.inference.remote_rfdetr_inferencer import RemoteRFDETRInferencer
                base_engine = RemoteRFDETRInferencer(model_path=model_path, conf_threshold=final_conf, device=device, imgsz=imgsz)
                mode_text = f"RF-DETR Remote • {device_label}"
            else:
                from src.inference.rfdetr_inferencer import RFDETRInferencer
                base_engine = RFDETRInferencer(model_path=model_path, conf_threshold=final_conf, device=device, imgsz=imgsz)
                mode_text = f"RF-DETR • {device_label}"
        else:  # .pt
            from src.inference.yolo_inferencer import YOLOInferencer
            base_engine = YOLOInferencer(model_path=model_path, conf_threshold=final_conf, device=device, imgsz=imgsz)
            mode_text = f"YOLO • {device_label}"

        return base_engine, mode_text

    def start(self, source, model_type, weights, device=None, imgsz=None, conf_thresh=None):
        """Start or hot-swap an inference session.

        If a stream is already running on the same source, the model is
        swapped without tearing down the reader or inference thread.
        Otherwise, a full restart is performed.

        Args:
            source: Video source — RTSP URL, integer webcam index, or
                path to an uploaded MP4/image file.
            model_type: ``'yolo'`` or ``'rfdetr'``.
            weights: Path to model weights (``.pt``, ``.pth``, ``.onnx``,
                ``.xml``, ``.engine``).
            device: ``None`` (auto), ``"cpu"``, or ``"cuda"``.
            imgsz: Inference image size (also sets web output size).
            conf_thresh: Confidence threshold override.

        Returns:
            ``(bool, str)`` — success flag and message.
        """
        # Empty / None source → fall back to the saved RTSP URL from settings.
        # The Live Inference landing page's "📡 Site Camera" button submits
        # source="" so the URL itself stays in Settings → Camera and isn't
        # editable on the landing page (avoiding URL-typing mistakes on site).
        if not source:
            try:
                settings_path = Path(__file__).parent / 'settings.json'
                if settings_path.exists():
                    with open(settings_path) as f:
                        ui_settings = json.load(f)
                    saved = ui_settings.get('rtsp_url', '').strip()
                    if saved:
                        source = saved
                        logger.info(f"▶ Using saved RTSP URL from settings: {source}")
                    else:
                        return False, ("No source provided and no rtsp_url is "
                                       "configured in Settings → Camera.")
            except Exception as e:
                return False, f"Could not read saved rtsp_url: {e}"

        # Normalize source for comparison
        src_val = int(source) if str(source).isdigit() else source

        # Check if we can hot-swap (same source, reader alive, thread running)
        can_hot_swap = (
            self.running
            and self.reader is not None
            and self._current_source == src_val
        )

        if can_hot_swap:
            # ── Hot-swap: keep reader, thread, tracker, counts, line — swap model only ─
            try:
                self.web_imgsz = imgsz or self.config.get('image_size', 416)
                base_engine, mode_text = self._build_engine(model_type, weights, imgsz, conf_thresh, device)
            except ValueError as e:
                return False, str(e)
            except Exception as e:
                logger.error(f"Hot-swap failed: {e}")
                return False, f"Model Error: {str(e)}"

            with self.lock:
                old_inferencer = self.engine.inferencer if self.engine else None
                self.engine.swap_inferencer(base_engine)
                # swap_inferencer rebuilt the palette annotators with defaults —
                # re-apply our web-resolution tuning on top.
                self._tune_annotators(self.engine)
                # Re-read render flags on every Start so a Settings toggle
                # (e.g. skip_masks) takes effect even if the source is unchanged.
                self._apply_render_settings(self.engine)
                # Backfill class_totals with any NEW class names the new model exposes
                # (without zeroing existing counts for classes that still apply).
                for name in base_engine.class_names.values():
                    self.class_totals.setdefault(name, 0)
                self.mode_text = mode_text

            # Free the old inferencer's resources — logger + tracker stay.
            del old_inferencer

            self.monitor.start_session(model_type=model_type)
            self._load_roi()
            logger.info(f"🔄 Model switched to {mode_text} (hot-swap — counts preserved)")
            self._persist_last_used(model_type, weights)
            return True, f"Model switched to {mode_text}."

        # ── Full restart: different source or not running ────────────────
        self.stop()

        with self.lock:
            if self.engine:
                del self.engine
                self.engine = None

            try:
                self.web_imgsz = imgsz or self.config.get('image_size', 416)
                base_engine, mode_text = self._build_engine(model_type, weights, imgsz, conf_thresh, device)
                # Merge mode-driven inference_config (bytetrack thresholds etc.)
                # under self.config so VisionEngine sees both the train.yaml
                # bits (logging) and the runtime mode bits.
                ve_config = {**self.config, **(self.inference_config or {})}
                self.engine = VisionEngine(inferencer=base_engine, config=ve_config)
                self._tune_annotators(self.engine)
                self._apply_line_settings(self.engine)
                self._apply_render_settings(self.engine)
                self.mode_text = mode_text
            except ValueError as e:
                return False, str(e)
            except Exception as e:
                logger.error(f"Failed to load AI engine: {e}")
                return False, f"Model Error: {str(e)}"

            self.class_totals = {name: 0 for name in base_engine.class_names.values()}
            self.source_is_image = str(source).lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            self.monitor.start_session(model_type=model_type)

            self._current_source = src_val
            self.reader = LiveReader(src_val)

            self.running = True
            self.inf_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.inf_thread.start()

            self._load_roi()
            logger.info(f"✅ Web App: Processing started ({self.mode_text})")
            self._persist_last_used(model_type, weights)
            return True, "Processing started."

    def stop(self):
        with self.lock:
            if not self.running: return True, "Already stopped."
            self.running = False
            self._current_source = None

            inf_thread = self.inf_thread
            reader = self.reader
            engine = self.engine
            counts = self.class_totals.copy()
            self.monitor.save_session_summary(counts)

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

                # BUG 2 FIX: publisher is app-lifetime (created in __init__), not session-lifetime.
                # Closing it here would kill the socket for any new session that starts before
                # this cleanup thread finishes. Publisher is only closed on full app shutdown.

                logger.info("🛑 Web App: Session Cleanup Finished (Thread Joined & CSV Saved)")
            except Exception as e:
                logger.error(f"🔥 Error during cleanup: {e}")

        cleanup_thread = threading.Thread(target=_cleanup, daemon=False)
        cleanup_thread.start()
        cleanup_thread.join(timeout=5.0)
        return True, "Session stopped."

    def _inference_loop(self):
        """Inference thread: pulls frames, runs AI, pushes annotated JPEG.

        Designed for long-running (12 h+) stability:

        - **Heartbeat log** every ~5 min (9000 frames at 30 fps) so the
          log confirms the thread is alive.
        - **CUDA cache flush** every ~60 s (1800 frames) returns cached
          VRAM to the allocator, preventing slow memory creep.
        - **Exponential backoff** on consecutive errors (0.1 s → 5 s
          cap): sustained GPU errors back off gracefully rather than
          flooding logs at 10 errors/s.
        - Resets error counter on every successful frame.
        """
        logger.info("🎬 Inference Thread: Started")
        frame_count = 0
        consecutive_errors = 0
        consecutive_drops = 0

        try:
            while self.running:
                with self.lock:
                    reader = self.reader
                    engine = self.engine

                if not reader or not engine:
                    time.sleep(0.01)
                    continue

                frame = reader.get_frame()
                if frame is None:
                    consecutive_drops += 1
                    self.monitor.track_frame_drop()
                    # Backoff on sustained drops: 1ms → 10ms → 50ms → 200ms cap
                    drop_sleep = min(0.001 * (2 ** min(consecutive_drops, 8)), 0.2)
                    time.sleep(drop_sleep)
                    continue
                consecutive_drops = 0

                try:
                    # Belt ROI crop — runs BEFORE the pre-engine downscale so
                    # the resize works on a smaller region and the model sees
                    # parcels at higher pixel density. Numpy slice = ~zero cost.
                    # Any failure latches ROI off for the rest of the session
                    # so a bad config can't spam the log every frame.
                    if self.roi is not None:
                        try:
                            x1, y1, x2, y2 = self.roi
                            h, w = frame.shape[:2]
                            x1c = max(0, min(x1, w))
                            y1c = max(0, min(y1, h))
                            x2c = max(x1c + 1, min(x2, w))
                            y2c = max(y1c + 1, min(y2, h))
                            cropped = frame[y1c:y2c, x1c:x2c]
                            if cropped.size == 0:
                                raise ValueError(f"empty crop for {w}×{h} with roi {self.roi}")
                            frame = cropped
                        except Exception as roi_err:
                            logger.warning(f"[ROI] Crop failed — disabling for this session: {roi_err}")
                            self.roi = None  # latch off; never retry mid-session

                    # Downscale ONCE — aspect-ratio preserved, fit within web_imgsz
                    fh, fw = frame.shape[:2]
                    if max(fh, fw) > self.web_imgsz:
                        scale = self.web_imgsz / max(fh, fw)
                        frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)))

                    _t_start = time.perf_counter()
                    annotated, detections, new_events = engine.process_frame(frame, self.class_totals)
                    _t_done = time.perf_counter()

                    stage_ms = dict(getattr(engine, 'last_timing', {}))
                    stage_ms['total_ms'] = (_t_done - _t_start) * 1000
                    self.monitor.track_frame(
                        latency_ms=stage_ms['total_ms'],
                        stage_ms=stage_ms,
                        detections=detections,
                    )
                    self.monitor.notify_no_counts(self.class_totals)

                    # One UDP datagram PER crossing — two close-together
                    # objects in the same frame now trigger two sort gates.
                    for event in new_events:
                        ts = datetime.datetime.now().isoformat()
                        with self.lock:
                            self.last_detected = {"class": event['class'], "time": ts, "id": event['id']}
                        latency_ns = self.publisher.publish(event['class'], event_id=event['id'])
                        self.monitor.track_udp_publish(latency_ns=latency_ns)
                        self.monitor.track_crossing()

                    # Throttle: encode every 2nd inference frame so display CPU
                    # is halved. Skipped frames re-serve the prior JPEG via the
                    # WS sender / MJPEG generator. frame_ready only fires on
                    # real encodes.
                    if self._encode_skip_idx % 2 == 0:
                        _, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                        self.latest_annotated = buffer.tobytes()
                        self.frame_ready.set()
                    self._encode_skip_idx += 1

                    consecutive_errors = 0
                    frame_count += 1

                    if frame_count % 9000 == 0:
                        logger.info(f"💓 Inference heartbeat — frame {frame_count:,} | counts: {self.class_totals}")
                        self.monitor.heartbeat()

                    # Periodic memory cleanup every ~60s
                    if frame_count % 1800 == 0:
                        import gc
                        gc.collect()
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except ImportError:
                            pass

                except Exception as e:
                    consecutive_errors += 1
                    is_oom = 'out of memory' in str(e).lower()
                    self.monitor.track_error(is_oom=is_oom)
                    backoff = min(0.1 * (2 ** (consecutive_errors - 1)), 5.0)
                    logger.error(f"Inference Loop Error (#{consecutive_errors}): {e} — retrying in {backoff:.1f}s")
                    time.sleep(backoff)

        except Exception as e:
            logger.critical(f"🔥 Inference thread crashed: {e}", exc_info=True)
            self.monitor.track_error(is_oom=False)
        finally:
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
                frame_bytes = self._standby_frame  # cached at __init__, no per-tick allocation
                try:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'Content-Length: ' + f"{len(frame_bytes)}".encode() + b'\r\n\r\n' +
                           frame_bytes + b'\r\n')
                except Exception:
                    return  # client disconnected — exit generator cleanly
                time.sleep(1)
                continue

            try:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + f"{len(frame_bytes)}".encode() + b'\r\n\r\n' +
                       frame_bytes + b'\r\n')
            except Exception:
                return  # client disconnected — exit generator cleanly

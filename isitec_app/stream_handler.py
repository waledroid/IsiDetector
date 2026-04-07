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


# Import inferencers from the parent directory
from src.inference.yolo_inferencer import YOLOInferencer
from src.inference.rfdetr_inferencer import RFDETRInferencer
from src.shared.vision_engine import VisionEngine
from isitec_app.performance_monitor import PerformanceMonitor

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
    3. ``configs/train.yaml`` → ``inference.udp.host / port``.
    4. Hardcoded defaults (``127.0.0.1:9502``).
    """

    def __init__(self, host="127.0.0.1", port=9502, enabled=True):
        self.host = host
        self.port = port
        self.enabled = enabled
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, class_name):
        if not self.enabled:
            return
        payload = json.dumps({
            "class": str(class_name),
            "ts": datetime.datetime.now().isoformat()   # microsecond precision
        }).encode()
        try:
            self._sock.sendto(payload, (self.host, self.port))
            logger.debug(f"[UDP] → {self.host}:{self.port} | {payload.decode()}")
        except Exception as e:
            logger.warning(f"[UDP] Send failed: {e}")

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

    def _run(self):
        logger.info(f"🚀 LiveReader: Thread started for source {self.source}")
        while self.running:
            try:
                with self._cap_lock:
                    if self.cap is None or not self.cap.isOpened():
                        self.cap = cv2.VideoCapture(self.source)
                        if not self.cap.isOpened():
                            self.cap = None
                            logger.error(f"❌ LiveReader: Failed to open {self.source}. Retrying in 2s...")
                            time.sleep(2)
                            continue
                        logger.info(f"✅ LiveReader: Connection established to {self.source}")
                        if self.is_file:
                            fps = self.cap.get(cv2.CAP_PROP_FPS)
                            self.frame_delay = (1.0 / fps) if fps > 0 else (1.0 / 30)
                            logger.info(f"🎞️ LiveReader: File source detected — pacing to {fps:.1f} fps")

                    ret, frame = self.cap.read()
                    if not ret:
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
        config_path = Path(__file__).resolve().parent.parent / "configs/train.yaml"
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
        self.language = 'en'
        self.latest_annotated = None
        self.inf_thread = None
        self.source_is_image = False
        self.frame_ready = threading.Event()

        # Cache the STANDBY frame once — avoids recreating it on every idle generate_frames() tick
        _blank = np.zeros((self.web_imgsz, self.web_imgsz, 3), dtype=np.uint8)
        cv2.putText(_blank, "STANDBY", (int(self.web_imgsz / 3), int(self.web_imgsz / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        _, _buf = cv2.imencode('.jpg', _blank)
        self._standby_frame = _buf.tobytes()

        # UDP Telemetry — priority: env var > train.yaml > default
        udp_cfg = self.config.get('inference', {}).get('udp', {})
        udp_host = os.environ.get('UDP_HOST', udp_cfg.get('host', '127.0.0.1'))
        udp_port = int(os.environ.get('UDP_PORT', udp_cfg.get('port', 9502)))
        udp_enabled = udp_cfg.get('enabled', True)
        self.publisher = UDPPublisher(host=udp_host, port=udp_port, enabled=udp_enabled)
        logger.info(f"[UDP] Publisher ready → {self.publisher.host}:{self.publisher.port}")

        # Performance Monitor — collects real-time metrics for /api/performance
        self.monitor = PerformanceMonitor()

    def set_language(self, lang):
        self.language = lang

    def set_udp_target(self, host, port):
        self.publisher.update_target(host, int(port))

    def get_udp_target(self):
        return {"host": self.publisher.host, "port": self.publisher.port, "enabled": self.publisher.enabled}

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

    def start(self, source, model_type, weights, device=None, imgsz=None, conf_thresh=None):
        """Start an inference session.

        Stops any running session first (safe to call when idle).
        Loads the selected AI model, creates a :class:`LiveReader`
        for the source, and starts the inference thread.

        Args:
            source: Video source — RTSP URL, integer webcam index, or
                path to an uploaded MP4/image file.
            model_type: ``'yolo'`` or ``'rfdetr'``. Determines which
                inferencer class is loaded.
            weights: Path to the model weights file (``.pt``, ``.pth``,
                or ``.onnx``). Falls back to the latest checkpoint if
                the path does not exist.
            device: ``None`` (auto — GPU if available), ``"cpu"``
                (force CPU), or ``"cuda"`` (force GPU).

        Returns:
            A ``(bool, str)`` tuple: ``(True, "Processing started.")``
            on success, ``(False, error_message)`` on failure.
        """
        # BUG 1 FIX: stop() acquires self.lock too — calling it inside the lock causes deadlock.
        # stop() is safe to call when not running (returns early), so call it first, outside lock.
        self.stop()

        with self.lock:
            if self.engine:
                del self.engine
                self.engine = None

            try:
                # Use provided conf_thresh or fallback to config
                final_conf = conf_thresh if conf_thresh is not None else self.config.get('inference', {}).get('conf_threshold', 0.3)
                
                is_detr = 'rfdetr' in model_type.lower() or (weights and 'rfdetr' in weights.lower())
                device_label = "CPU" if device == "cpu" else "GPU"

                if is_detr:
                    model_path = weights or "models/rfdetr/25-03-2026_0043/checkpoint_best_ema.pth"
                    if not Path(model_path).exists():
                        found = list(Path("runs/segment/models/rfdetr").rglob("*.onnx"))
                        if found: model_path = str(found[0])
                    base_engine = RFDETRInferencer(model_path=model_path, conf_threshold=final_conf, device=device, imgsz=imgsz)
                    self.mode_text = f"MODE 2 (DETR • {device_label})"
                else:
                    model_path = weights or "runs/segment/models/yolo/25-03-20262/weights/best.pt"
                    if not Path(model_path).exists():
                        found_pt = list(Path("runs/segment/models/yolo").rglob("*.pt"))
                        if found_pt:
                            model_path = str(found_pt[0])
                        else:
                            found_onnx = list(Path("runs/segment/models/yolo").rglob("*.onnx"))
                            if found_onnx: model_path = str(found_onnx[0])

                    base_engine = YOLOInferencer(model_path=model_path, conf_threshold=final_conf, device=device, imgsz=imgsz)
                    self.mode_text = f"MODE 1 (YOLO • {device_label})"
                
                self.engine = VisionEngine(inferencer=base_engine, config=self.config)
            except Exception as e:
                logger.error(f"Failed to load AI engine: {e}")
                return False, f"Model Error: {str(e)}"

            self.class_totals = {name: 0 for name in base_engine.class_names.values()}
            self.source_is_image = str(source).lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            self.monitor.start_session(model_type=model_type)
            
            # Launch Background Reader for Zero-Lag
            src_val = int(source) if str(source).isdigit() else source
            self.reader = LiveReader(src_val)
            
            # Launch Inference Thread
            self.running = True
            self.inf_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.inf_thread.start()
            
            logger.info(f"✅ Web App: Processing started ({self.mode_text})")
            return True, "Processing started."

    def stop(self):
        with self.lock:
            if not self.running: return True, "Already stopped."
            self.running = False
            
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

        threading.Thread(target=_cleanup, daemon=True).start()
        return True, "Stopping session..."

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

        while self.running:
            with self.lock:
                reader = self.reader
                engine = self.engine

            if not reader or not engine:
                time.sleep(0.01)
                continue

            frame = reader.get_frame()
            if frame is None:
                self.monitor.track_frame_drop()
                time.sleep(0.001)
                continue

            try:
                _t_start = time.perf_counter()
                annotated, detections, event = engine.process_frame(frame, self.class_totals)
                _t_done = time.perf_counter()

                stage_ms = dict(getattr(engine, 'last_timing', {}))
                stage_ms['total_ms'] = (_t_done - _t_start) * 1000
                self.monitor.track_frame(
                    latency_ms=stage_ms['total_ms'],
                    stage_ms=stage_ms,
                    detections=detections,
                )
                self.monitor.notify_no_counts(self.class_totals)

                if event:
                    ts = datetime.datetime.now().isoformat()
                    with self.lock:
                        self.last_detected = {"class": event['class'], "time": ts}
                    self.publisher.publish(event['class'])
                    self.monitor.track_udp_publish()
                    self.monitor.track_crossing()

                annotated = self.draw_isi_ui(annotated, self.class_totals, self.mode_text)

                # Downsample for web throughput if needed
                h, w = annotated.shape[:2]
                if w > self.web_imgsz or h > self.web_imgsz:
                    annotated = cv2.resize(annotated, (self.web_imgsz, self.web_imgsz))

                # Quality 50 is much lighter for 30fps MJPEG
                _, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                self.latest_annotated = buffer.tobytes()
                self.frame_ready.set()

                consecutive_errors = 0  # reset backoff on successful frame
                frame_count += 1

                # Heartbeat every ~5 min (at 30fps) so log confirms the thread is alive
                if frame_count % 9000 == 0:
                    logger.info(f"💓 Inference heartbeat — frame {frame_count:,} | counts: {self.class_totals}")
                    self.monitor.heartbeat()

                # Periodic CUDA cache flush every ~60s — returns cached VRAM to the allocator
                if frame_count % 1800 == 0:
                    import torch, gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                consecutive_errors += 1
                is_oom = 'out of memory' in str(e).lower()
                self.monitor.track_error(is_oom=is_oom)
                # Exponential backoff: 0.1s → 0.2s → 0.4s … capped at 5s
                backoff = min(0.1 * (2 ** (consecutive_errors - 1)), 5.0)
                logger.error(f"Inference Loop Error (#{consecutive_errors}): {e} — retrying in {backoff:.1f}s")
                time.sleep(backoff)

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

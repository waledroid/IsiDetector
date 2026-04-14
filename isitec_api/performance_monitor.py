import time
import threading
import logging
import statistics
from collections import deque
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Collects real-time inference pipeline metrics for the Performance dashboard.

    Designed to be instantiated once in ``StreamHandler.__init__()`` and kept
    alive for the lifetime of the Flask process. Each new inference session
    calls ``start_session()`` to reset per-session counters while the
    rolling windows accumulate over the current session.

    Metric collection hooks (call these from ``stream_handler.py``):

    - :meth:`start_session` — reset state at stream start
    - :meth:`track_frame` — called every inference frame
    - :meth:`track_frame_drop` — called when the reader queue is empty
    - :meth:`track_error` — called in the inference except block
    - :meth:`track_udp_publish` — called after each UDP datagram
    - :meth:`track_csv_write` — called after each DailyLogger save
    - :meth:`track_crossing` — called when a line-crossing event fires
    - :meth:`heartbeat` — called at the existing 9000-frame heartbeat point

    GPU stats are read via ``pynvml`` if available; silently returns
    ``None`` fields otherwise so the dashboard degrades gracefully.
    """

    # ── Status thresholds ─────────────────────────────────────────────────────

    _THRESHOLDS = {
        'throughput': {
            'fps':        {'yellow': 20,   'red': 10},     # below value → worse
            'latency_ms': {'yellow': 50,   'red': 100},    # above value → worse
            'frame_drops':{'yellow': 1,    'red': 20},     # above value → worse
        },
        'detection': {
            'avg_confidence':  {'green_min': 0.75, 'yellow_min': 0.55},
            'low_conf_rate':   {'yellow': 0.10,    'red': 0.25},        # above → worse
        },
        'tracking': {
            'id_ratio': {'yellow': 2.0, 'red': 5.0},  # above → worse
        },
        'hardware': {
            'vram_pct':     {'yellow': 70, 'red': 85},
            'gpu_util_pct': {'yellow': 80, 'red': 90},
            'gpu_temp_c':   {'yellow': 75, 'red': 85},
        },
    }

    def __init__(self):
        self.lock = threading.Lock()
        self.session_start: float | None = None
        self._model_type: str = ''

        # Rolling windows — stats computed over the last N frames
        self._fps_ts    = deque(maxlen=60)   # perf_counter timestamps for FPS
        self._latency   = deque(maxlen=100)  # total ms per frame
        self._forward   = deque(maxlen=100)  # forward+preproc ms
        self._tracker   = deque(maxlen=100)  # tracker update ms
        self._conf      = deque(maxlen=500)  # per-detection confidence scores
        self._det_count = deque(maxlen=100)  # detections per frame
        self._coverage  = deque(maxlen=200)  # mask/bbox area ratios

        # Per-session counters (reset by start_session)
        self.belt_active       = True
        self.frame_drops       = 0
        self.error_count       = 0
        self.cuda_oom_count    = 0
        self.udp_published     = 0
        self.csv_writes_ok     = 0
        self.csv_writes_failed = 0
        self._track_ids        = set()  # all unique ByteTrack IDs seen
        self.total_crossings   = 0
        self._last_heartbeat   = None
        self._no_count_since   = None   # timestamp when counts first stalled

        # Baseline snapshot (captured at session start, before model load)
        self._baseline_vram_mb:  float | None = None
        self._baseline_ram_mb:   float | None = None
        self._baseline_gpu_temp: int   | None = None
        self._baseline_gpu_util: int   | None = None

        # pynvml
        self._nvml_handle = None
        self._init_nvml()

        # sessions log path (relative to this file)
        self._sessions_path = Path(__file__).parent / 'logs' / 'sessions.json'

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml_handle = None

    @property
    def has_gpu(self) -> bool:
        """True if an NVIDIA GPU is available."""
        if self._nvml_handle is not None:
            return True
        self._init_nvml()
        return self._nvml_handle is not None

    def start_session(self, model_type: str = ''):
        """Reset all per-session counters. Call at the start of each stream."""
        with self.lock:
            # Capture baseline resource usage before model loads
            gpu = self._get_gpu()
            ram = self._get_ram_usage()
            self._baseline_vram_mb  = gpu['vram_used_mb']  if gpu else None
            self._baseline_ram_mb   = ram['used_mb']       if ram else None
            self._baseline_gpu_temp = gpu['gpu_temp_c']    if gpu else None
            self._baseline_gpu_util = gpu['gpu_util_pct']  if gpu else None

            self.session_start    = time.time()
            self._model_type      = model_type
            self.belt_active      = True
            self.frame_drops      = 0
            self.error_count      = 0
            self.cuda_oom_count   = 0
            self.udp_published    = 0
            self.csv_writes_ok    = 0
            self.csv_writes_failed = 0
            self._track_ids       = set()
            self.total_crossings  = 0
            self._last_heartbeat  = time.time()
            self._no_count_since  = None
            self._fps_ts.clear()
            self._latency.clear()
            self._forward.clear()
            self._tracker.clear()
            self._conf.clear()
            self._det_count.clear()
            self._coverage.clear()

    # ── Metric hooks ──────────────────────────────────────────────────────────

    def track_frame(self, latency_ms: float, stage_ms: dict, detections):
        """Record one processed frame. Called from ``_inference_loop`` after
        ``engine.process_frame()``.

        Args:
            latency_ms: Total time for ``process_frame()`` in milliseconds.
            stage_ms: Dict with optional keys ``forward_ms``, ``tracker_ms``
                from ``engine.last_timing``.
            detections: ``sv.Detections`` object (may be empty).
        """
        with self.lock:
            now = time.perf_counter()
            self._fps_ts.append(now)
            self._latency.append(latency_ms)
            self._last_heartbeat = time.time()

            if stage_ms.get('forward_ms') is not None:
                self._forward.append(stage_ms['forward_ms'])
            if stage_ms.get('tracker_ms') is not None:
                self._tracker.append(stage_ms['tracker_ms'])

            # Confidence + detection density
            if detections is not None and hasattr(detections, 'confidence') and detections.confidence is not None:
                for c in detections.confidence:
                    self._conf.append(float(c))
                self._det_count.append(len(detections.confidence))
            else:
                self._det_count.append(0)

            # Mask coverage (proxy for segmentation quality)
            if (detections is not None
                    and hasattr(detections, 'mask') and detections.mask is not None
                    and hasattr(detections, 'xyxy') and detections.xyxy is not None):
                for mask, box in zip(detections.mask, detections.xyxy):
                    mask_area = float(mask.sum())
                    x1, y1, x2, y2 = box
                    bbox_area = max(1.0, float((x2 - x1) * (y2 - y1)))
                    self._coverage.append(mask_area / bbox_area)

            # Unique tracker IDs (proxy for track fragmentation)
            if (detections is not None
                    and hasattr(detections, 'tracker_id') and detections.tracker_id is not None):
                self._track_ids.update(int(i) for i in detections.tracker_id)

    def track_frame_drop(self):
        with self.lock:
            self.frame_drops += 1

    def track_error(self, is_oom: bool = False):
        with self.lock:
            self.error_count += 1
            if is_oom:
                self.cuda_oom_count += 1

    def track_udp_publish(self):
        with self.lock:
            self.udp_published += 1

    def track_csv_write(self, success: bool):
        with self.lock:
            if success:
                self.csv_writes_ok += 1
            else:
                self.csv_writes_failed += 1

    def track_crossing(self):
        with self.lock:
            self.total_crossings += 1
            self._no_count_since = None  # reset stall timer

    def heartbeat(self):
        with self.lock:
            self._last_heartbeat = time.time()

    def notify_no_counts(self, class_totals: dict):
        """Called each frame to detect counting stalls.

        If the stream is running but all counts are 0 (or unchanged since
        last crossing), record when the stall started for status computation.
        """
        with self.lock:
            total = sum(class_totals.values()) if class_totals else 0
            if total == 0 and self.total_crossings == 0:
                if self._no_count_since is None:
                    self._no_count_since = time.time()
            else:
                self._no_count_since = None

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def get_snapshot(self) -> dict:
        """Return a complete performance snapshot with computed status fields.

        Thread-safe. Called by ``StreamHandler.get_performance()``.

        Returns:
            Dict with keys: ``session``, ``throughput``, ``detection``,
            ``tracking``, ``hardware``, ``counting``, ``sessions``.
            Every group includes a ``"status"`` key: ``"green"``,
            ``"yellow"``, or ``"red"``.
        """
        with self.lock:
            now = time.time()

            # ── Session ──────────────────────────────────────────────────────
            uptime_s = round(now - self.session_start, 1) if self.session_start else 0
            hb_age   = round(now - self._last_heartbeat, 1) if self._last_heartbeat else None
            session  = {
                'uptime_s':       uptime_s,
                'uptime_fmt':     self._fmt_uptime(uptime_s),
                'error_count':    self.error_count,
                'cuda_oom_count': self.cuda_oom_count,
                'heartbeat_age_s': hb_age,
            }
            session['status'] = self._status_session(session)

            # ── Throughput ───────────────────────────────────────────────────
            fps     = self._calc_fps()
            lat_avg = self._avg(self._latency)
            fwd_avg = self._avg(self._forward)
            trk_avg = self._avg(self._tracker)
            throughput = {
                'fps':         round(fps, 1)     if fps     is not None else None,
                'latency_ms':  round(lat_avg, 1) if lat_avg is not None else None,
                'forward_ms':  round(fwd_avg, 1) if fwd_avg is not None else None,
                'tracker_ms':  round(trk_avg, 1) if trk_avg is not None else None,
                'frame_drops': self.frame_drops,
            }
            throughput['status'] = self._status_throughput(throughput)

            # ── Detection Quality ────────────────────────────────────────────
            avg_conf     = self._avg(self._conf)
            low_conf_ct  = sum(1 for c in self._conf if c < 0.60)
            low_conf_r   = round(low_conf_ct / max(1, len(self._conf)), 3) if self._conf else None
            avg_det      = self._avg(self._det_count)
            avg_cov      = self._avg(self._coverage)
            detection = {
                'avg_confidence':  round(avg_conf, 3)  if avg_conf  is not None else None,
                'low_conf_rate':   low_conf_r,
                'avg_detections':  round(avg_det, 1)   if avg_det   is not None else None,
                'mask_coverage':   round(avg_cov, 2)   if avg_cov   is not None else None,
            }
            detection['status'] = self._status_detection(detection)

            # ── Tracking ─────────────────────────────────────────────────────
            total_ids = len(self._track_ids)
            crossings = self.total_crossings
            id_ratio  = round(total_ids / crossings, 2) if crossings > 0 else None
            tracking = {
                'total_unique_ids': total_ids,
                'total_crossings':  crossings,
                'id_ratio':         id_ratio,
            }
            tracking['status'] = self._status_tracking(tracking)

            # ── Hardware ─────────────────────────────────────────────────────
            gpu   = self._get_gpu()
            ram   = self._get_ram_usage()
            cpu   = self._get_cpu_info()
            hardware = {
                'has_gpu':       gpu is not None,
                # GPU fields (None if no GPU)
                'vram_used_mb':  gpu['vram_used_mb']  if gpu else None,
                'vram_total_mb': gpu['vram_total_mb'] if gpu else None,
                'vram_pct':      gpu['vram_pct']      if gpu else None,
                'gpu_util_pct':  gpu['gpu_util_pct']  if gpu else None,
                'gpu_temp_c':    gpu['gpu_temp_c']    if gpu else None,
                # CPU fields (always available)
                'cpu_pct':       cpu['cpu_pct']       if cpu else None,
                'cpu_freq_mhz':  cpu['cpu_freq_mhz']  if cpu else None,
                'cpu_temp_c':    cpu['cpu_temp_c']    if cpu else None,
                'cpu_cores':     cpu['cpu_cores']     if cpu else None,
                # RAM
                'ram_used_mb':   ram['used_mb']       if ram else None,
                'ram_total_mb':  ram['total_mb']      if ram else None,
                'ram_pct':       ram['pct']           if ram else None,
                # Deltas from pre-inference baseline
                'vram_delta_mb':    self._delta(gpu, 'vram_used_mb', self._baseline_vram_mb),
                'ram_delta_mb':     self._delta(ram, 'used_mb',      self._baseline_ram_mb),
                'temp_delta_c':     self._delta(gpu, 'gpu_temp_c',   self._baseline_gpu_temp),
                'gpu_util_delta_pct': self._delta(gpu, 'gpu_util_pct', self._baseline_gpu_util),
            }
            hardware['status'] = self._status_hardware(hardware)

            # counting status is set by get_performance() which has class_totals
            counting = {'status': 'green'}

            sessions = self._load_sessions()

        return {
            'session':    session,
            'throughput': throughput,
            'detection':  detection,
            'tracking':   tracking,
            'hardware':   hardware,
            'counting':   counting,
            'sessions':   sessions,
        }

    # ── Session history ───────────────────────────────────────────────────────

    def save_session_summary(self, class_totals: dict, model_type: str = ''):
        """Append an end-of-session summary to ``logs/sessions.json``.

        Called from ``StreamHandler.stop()`` before cleanup.
        """
        if not self.session_start:
            return
        try:
            self._sessions_path.parent.mkdir(parents=True, exist_ok=True)
            sessions = self._load_sessions(limit=None)

            duration_s = time.time() - self.session_start
            fps_snap   = self._calc_fps()
            avg_conf   = self._avg(self._conf)
            crossings  = self.total_crossings
            total_ids  = len(self._track_ids)

            entry = {
                'date':         __import__('datetime').datetime.now().strftime('%d-%m-%Y %H:%M'),
                'model':        model_type or self._model_type or '—',
                'duration_h':   round(duration_s / 3600, 2),
                'fps':          round(fps_snap, 1) if fps_snap else None,
                'avg_confidence': round(avg_conf, 3) if avg_conf else None,
                'id_ratio':     round(total_ids / crossings, 2) if crossings > 0 else None,
                'counts':       dict(class_totals),
                'baseline': {
                    'vram_mb':  self._baseline_vram_mb,
                    'ram_mb':   self._baseline_ram_mb,
                    'gpu_temp': self._baseline_gpu_temp,
                },
            }
            sessions.insert(0, entry)
            sessions = sessions[:10]  # keep last 10 sessions

            with open(self._sessions_path, 'w') as f:
                json.dump(sessions, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session summary: {e}")

    def _load_sessions(self, limit: int | None = 5) -> list:
        try:
            if self._sessions_path.exists():
                with open(self._sessions_path) as f:
                    data = json.load(f)
                return data[:limit] if limit else data
        except Exception:
            pass
        return []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _calc_fps(self) -> float | None:
        if len(self._fps_ts) < 2:
            return None
        elapsed = self._fps_ts[-1] - self._fps_ts[0]
        return (len(self._fps_ts) - 1) / elapsed if elapsed > 0 else None

    @staticmethod
    def _avg(q) -> float | None:
        return statistics.mean(q) if q else None

    @staticmethod
    def _delta(current: dict | None, key: str, baseline: float | None):
        """Return current − baseline, or None if either side is missing."""
        if current is None or baseline is None:
            return None
        val = current.get(key)
        if val is None:
            return None
        return round(val - baseline, 1)

    @staticmethod
    def _fmt_uptime(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}h {m:02d}m {s:02d}s"

    def _get_gpu(self) -> dict | None:
        if self._nvml_handle is None:
            self._init_nvml() # try again if it failed
        if self._nvml_handle is None:
            return None
        try:
            import pynvml
            mem  = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            temp = pynvml.nvmlDeviceGetTemperature(self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
            return {
                'vram_used_mb':  round(mem.used  / 1024**2),
                'vram_total_mb': round(mem.total / 1024**2),
                'vram_pct':      round(mem.used / mem.total * 100, 1),
                'gpu_util_pct':  util.gpu,
                'gpu_temp_c':    temp,
            }
        except Exception as e:
            logger.error(f"nvml error: {e}")
            self._nvml_handle = None
            return None

    @staticmethod
    def _get_ram_usage() -> dict | None:
        try:
            import psutil
            vm = psutil.virtual_memory()
            return {
                'used_mb': round((vm.total - vm.available) / 1024**2, 1),
                'total_mb': round(vm.total / 1024**2, 1),
                'pct': vm.percent
            }
        except Exception:
            return None

    @staticmethod
    def _get_cpu_info() -> dict | None:
        """CPU utilization, frequency, and temperature (Linux/Windows)."""
        try:
            import psutil
            cpu_pct = psutil.cpu_percent(interval=None)
            freq = psutil.cpu_freq()
            cpu_freq_mhz = round(freq.current) if freq else None

            # CPU temperature — Linux only via sensors
            cpu_temp = None
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Try common sensor names
                    for key in ('coretemp', 'k10temp', 'cpu_thermal', 'cpu-thermal'):
                        if key in temps and temps[key]:
                            cpu_temp = round(temps[key][0].current, 1)
                            break
                    # Fallback: first available sensor
                    if cpu_temp is None:
                        for entries in temps.values():
                            if entries:
                                cpu_temp = round(entries[0].current, 1)
                                break

            return {
                'cpu_pct': cpu_pct,
                'cpu_freq_mhz': cpu_freq_mhz,
                'cpu_temp_c': cpu_temp,
                'cpu_cores': psutil.cpu_count(logical=True),
            }
        except Exception:
            return None

    # ── Status computation ────────────────────────────────────────────────────

    @staticmethod
    def _worst(statuses: list) -> str:
        if 'red'    in statuses: return 'red'
        if 'yellow' in statuses: return 'yellow'
        return 'green'

    def _status_session(self, d: dict) -> str:
        s = []
        if d['cuda_oom_count'] > 0:               s.append('red')
        elif d['error_count'] > 5:                s.append('red')
        elif d['error_count'] > 0:                s.append('yellow')
        else:                                      s.append('green')

        hb = d['heartbeat_age_s']
        if hb is not None:
            if hb > 30:   s.append('red')
            elif hb > 10: s.append('yellow')
            else:          s.append('green')
        return self._worst(s)

    def _status_throughput(self, d: dict) -> str:
        s = []
        fps = d['fps']
        if fps is not None:
            if fps < 10:   s.append('red')
            elif fps < 20: s.append('yellow')
            else:           s.append('green')

        lat = d['latency_ms']
        if lat is not None:
            if lat > 100:  s.append('red')
            elif lat > 50: s.append('yellow')
            else:           s.append('green')

        drops = d['frame_drops']
        # drops accumulate over whole session, so thresholds should be high
        if drops > 1000: s.append('red')
        elif drops > 100: s.append('yellow')
        else:            s.append('green')

        return self._worst(s) if s else 'green'

    def _status_detection(self, d: dict) -> str:
        s = []
        conf = d['avg_confidence']
        if conf is not None:
            if conf < 0.55:   s.append('red')
            elif conf < 0.75: s.append('yellow')
            else:              s.append('green')

        lcr = d['low_conf_rate']
        if lcr is not None:
            if lcr > 0.25:   s.append('red')
            elif lcr > 0.10: s.append('yellow')
            else:             s.append('green')

        return self._worst(s) if s else 'green'

    def _status_tracking(self, d: dict) -> str:
        ratio = d['id_ratio']
        if ratio is None:    return 'green'
        if ratio > 5.0:      return 'red'
        if ratio > 2.0:      return 'yellow'
        return 'green'

    def _status_hardware(self, d: dict) -> str:
        s = []
        thresholds = [
            ('vram_pct',    (70, 85)),
            ('gpu_util_pct',(80, 90)),
            ('gpu_temp_c',  (75, 85)),
            ('cpu_pct',     (80, 95)),
            ('cpu_temp_c',  (75, 90)),
            ('ram_pct',     (85, 95)),
        ]
        for key, (w, r) in thresholds:
            v = d.get(key)
            if v is None: continue
            if v > r:   s.append('red')
            elif v > w: s.append('yellow')
            else:        s.append('green')
        return self._worst(s) if s else 'green'

    def _status_counting(self, class_totals: dict, is_running: bool) -> str:
        if not is_running:
            return 'green'
        if not self.session_start:
            return 'green'
        if not getattr(self, 'belt_active', True):
            return 'green'
        elapsed = time.time() - self.session_start
        total = sum(class_totals.values()) if class_totals else 0
        if total == 0 and elapsed > 600:   # 10 min with no counts
            return 'red'
        if total == 0 and elapsed > 300:   # 5 min with no counts
            return 'yellow'
        return 'green'

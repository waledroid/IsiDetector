"""Digital-output (relay pulse) publisher — the "wire" alternative to UDP.

On every line-crossing event the inference loop calls
:meth:`DigitalOutPublisher.pulse(class_name)`. The class is mapped to a
relay channel (``{"carton": 1, "polybag": 2}``) and a worker thread drives
that channel ON for ``pulse_ms`` then OFF — a plain electrical pulse the
sorter PLC reads as a digital input, exactly like a photocell.

Why a worker thread: a USB/serial or Modbus write can take 1–5 ms and a
pulse must be *held* for tens of ms. Neither belongs on the inference
loop. Pulses are serialised back-to-back (never overlapped) so two
crossings in one frame produce two distinct pulses on the same channel.

Drivers (selected by ``driver`` key):

- ``serial``     — USB relay boards (and Ethernet ones via pyserial URLs
                   like ``socket://10.0.0.5:4196``). Protocols: ``numato``
                   (``relay on 0\\r``), ``lcus`` (CH340 ``A0 01 01 A2``
                   hex boards), ``custom`` (user templates).
- ``modbus_tcp`` — Ethernet relay modules / PLC coils. Write Single Coil
                   (FC 05) over raw TCP — no library needed.
- ``sim``        — no hardware; records pulses. For demos and CI.

The publisher is app-lifetime (like ``UDPPublisher``) and fully
re-configurable at runtime via :meth:`configure` — toggling ``enabled``
in the Settings page takes effect on the next crossing, no restart.
"""
from __future__ import annotations

import codecs
import logging
import queue
import socket
import struct
import threading
import time
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

DEFAULTS = {
    "enabled": False,
    "driver": "sim",                 # sim | serial | modbus_tcp
    "device": "/dev/ttyUSB0",        # serial path / pyserial URL, or host[:port] for modbus
    "pulse_ms": 50,
    "map": {"carton": 1, "polybag": 2},
    "serial_baud": 9600,
    "serial_protocol": "numato",     # numato | lcus | custom
    "on_cmd": "relay on {ch0}\\r",   # custom protocol templates ({ch} 1-based, {ch0} 0-based)
    "off_cmd": "relay off {ch0}\\r",
    "modbus_unit": 1,
    "modbus_offset": 0,              # coil address = channel - 1 + offset
}

_RETRY_OPEN_S = 2.0      # back-off between failed device opens
_MAX_QUEUE = 64          # bounded: a dead device must never grow memory unbounded


# ── Drivers ──────────────────────────────────────────────────────────────────

class _Driver:
    name = "base"

    def open(self) -> None: ...
    def set(self, channel: int, on: bool) -> None: ...
    def close(self) -> None: ...


class SimDriver(_Driver):
    """No hardware — records ``(ts, channel, on)`` in a bounded deque."""
    name = "sim"

    def __init__(self, cfg: dict):
        self.events: deque = deque(maxlen=200)

    def open(self) -> None:
        pass

    def set(self, channel: int, on: bool) -> None:
        self.events.append((time.perf_counter(), channel, on))

    def close(self) -> None:
        pass


class SerialDriver(_Driver):
    """USB / Ethernet relay boards speaking a serial line protocol."""
    name = "serial"

    def __init__(self, cfg: dict):
        self.device = str(cfg["device"])
        self.baud = int(cfg.get("serial_baud", 9600))
        self.protocol = str(cfg.get("serial_protocol", "numato")).lower()
        self.on_tpl = str(cfg.get("on_cmd", DEFAULTS["on_cmd"]))
        self.off_tpl = str(cfg.get("off_cmd", DEFAULTS["off_cmd"]))
        self._ser = None

    def open(self) -> None:
        import serial  # pyserial — deploy dependency
        # serial_for_url handles both real ttys and socket:// / rfc2217:// URLs
        self._ser = serial.serial_for_url(
            self.device, baudrate=self.baud, timeout=1.0, write_timeout=1.0)

    @staticmethod
    def _render(tpl: str, channel: int) -> bytes:
        text = tpl.format(ch=channel, ch0=channel - 1,
                          chx=f"{channel:02X}", ch0x=f"{channel - 1:02X}")
        # allow "\r", "\x41" style escapes in the template
        return codecs.decode(text, "unicode_escape").encode("latin-1")

    def frame(self, channel: int, on: bool) -> bytes:
        if self.protocol == "numato":
            return f"relay {'on' if on else 'off'} {channel - 1}\r".encode()
        if self.protocol == "lcus":
            # CH340 "LCUS-x" boards: A0 <ch> <01|00> <checksum = sum & 0xFF>
            val = 1 if on else 0
            return bytes([0xA0, channel, val, (0xA0 + channel + val) & 0xFF])
        return self._render(self.on_tpl if on else self.off_tpl, channel)

    def set(self, channel: int, on: bool) -> None:
        if self._ser is None:
            raise RuntimeError("serial device not open")
        self._ser.write(self.frame(channel, on))
        self._ser.flush()

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None


class ModbusTcpDriver(_Driver):
    """Write Single Coil (FC 05) over Modbus/TCP — Ethernet relay modules, PLC coils."""
    name = "modbus_tcp"

    def __init__(self, cfg: dict):
        dev = str(cfg["device"])
        host, _, port = dev.partition(":")
        self.host = host or "127.0.0.1"
        self.port = int(port) if port else 502
        self.unit = int(cfg.get("modbus_unit", 1))
        self.offset = int(cfg.get("modbus_offset", 0))
        self._sock: socket.socket | None = None
        self._tid = 0

    def open(self) -> None:
        self._sock = socket.create_connection((self.host, self.port), timeout=1.0)
        self._sock.settimeout(1.0)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def frame(self, channel: int, on: bool) -> bytes:
        self._tid = (self._tid + 1) & 0xFFFF
        addr = channel - 1 + self.offset
        pdu = struct.pack(">BHH", 0x05, addr, 0xFF00 if on else 0x0000)
        mbap = struct.pack(">HHHB", self._tid, 0, len(pdu) + 1, self.unit)
        return mbap + pdu

    def set(self, channel: int, on: bool) -> None:
        if self._sock is None:
            raise RuntimeError("modbus socket not open")
        self._sock.sendall(self.frame(channel, on))
        # FC05 echoes the request (12 bytes). Read it so we *know* the coil
        # was written — this is the built-in ack the wire path otherwise lacks.
        buf = b""
        while len(buf) < 12:
            chunk = self._sock.recv(12 - len(buf))
            if not chunk:
                raise ConnectionError("modbus peer closed")
            buf += chunk
        if buf[7] & 0x80:
            raise RuntimeError(f"modbus exception 0x{buf[8]:02X}")

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None


_DRIVERS = {"sim": SimDriver, "serial": SerialDriver, "modbus_tcp": ModbusTcpDriver}


# ── Publisher ────────────────────────────────────────────────────────────────

class DigitalOutPublisher:
    """App-lifetime relay-pulse publisher. See module docstring."""

    def __init__(self, cfg: dict | None = None):
        self._lock = threading.RLock()
        self._q: queue.Queue = queue.Queue(maxsize=_MAX_QUEUE)
        self._driver: _Driver | None = None
        self._connected = False
        self._last_open_attempt = 0.0
        self._last_open_warn = -1e9
        self._cfg = dict(DEFAULTS)
        self._cfg["map"] = dict(DEFAULTS["map"])
        # counters / telemetry
        self.fired = 0
        self.errors = 0
        self.dropped = 0
        self.last_error: str | None = None
        self.last_channel: int | None = None
        self.last_class: str | None = None
        self.last_ts: str | None = None
        self._latency_us: deque = deque(maxlen=200)   # enqueue → ON written
        self.recent: deque = deque(maxlen=50)          # (ts, class, channel, ok)
        self._ch_last: dict = {}                       # channel → (monotonic ts of last ON, class)
        self._ch_fired: dict = {}                      # channel → pulses fired
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="dio-worker", daemon=True)
        self._worker.start()
        if cfg:
            self.configure(cfg)

    # ── configuration ────────────────────────────────────────────────────
    @staticmethod
    def normalize(cfg: dict) -> dict:
        """Coerce/validate a raw settings dict → clean config. Raises ValueError."""
        out = dict(DEFAULTS)
        out["map"] = dict(DEFAULTS["map"])
        if "enabled" in cfg:
            out["enabled"] = bool(cfg["enabled"])
        if "driver" in cfg:
            d = str(cfg["driver"]).lower().strip()
            if d not in _DRIVERS:
                raise ValueError(f"dio_driver must be one of {sorted(_DRIVERS)}")
            out["driver"] = d
        if "device" in cfg:
            v = cfg["device"]
            if not isinstance(v, str) or not (0 < len(v.strip()) <= 255):
                raise ValueError("dio_device must be a non-empty string ≤ 255 chars")
            out["device"] = v.strip()
        if "pulse_ms" in cfg:
            n = int(cfg["pulse_ms"])
            if not (5 <= n <= 5000):
                raise ValueError("dio_pulse_ms must be 5-5000")
            out["pulse_ms"] = n
        if "map" in cfg:
            m = cfg["map"]
            if not isinstance(m, dict) or not m:
                raise ValueError("dio_map must be a non-empty {class: channel} object")
            clean = {}
            for k, v in m.items():
                ch = int(v)
                if not (0 <= ch <= 255):
                    raise ValueError("dio_map channels must be 0-255 (0 = not wired)")
                clean[str(k)] = ch
            out["map"] = clean
        if "serial_baud" in cfg:
            b = int(cfg["serial_baud"])
            if b not in (1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200):
                raise ValueError("dio_serial_baud must be a standard baud rate")
            out["serial_baud"] = b
        if "serial_protocol" in cfg:
            p = str(cfg["serial_protocol"]).lower().strip()
            if p not in ("numato", "lcus", "custom"):
                raise ValueError("dio_serial_protocol must be numato | lcus | custom")
            out["serial_protocol"] = p
        for k in ("on_cmd", "off_cmd"):
            if k in cfg:
                v = cfg[k]
                if not isinstance(v, str) or len(v) > 128:
                    raise ValueError(f"dio_{k} must be a string ≤ 128 chars")
                out[k] = v
        if "modbus_unit" in cfg:
            u = int(cfg["modbus_unit"])
            if not (0 <= u <= 255):
                raise ValueError("dio_modbus_unit must be 0-255")
            out["modbus_unit"] = u
        if "modbus_offset" in cfg:
            o = int(cfg["modbus_offset"])
            if not (0 <= o <= 65535):
                raise ValueError("dio_modbus_offset must be 0-65535")
            out["modbus_offset"] = o
        return out

    @staticmethod
    def from_settings(ui: dict) -> dict:
        """Extract the ``dio_*`` keys of settings.json into a raw config dict."""
        raw = {}
        for k in DEFAULTS:
            sk = f"dio_{k}"
            if sk in ui:
                raw[k] = ui[sk]
        return raw

    def configure(self, cfg: dict) -> None:
        """Apply a (raw or normalized) config live. Reopens the driver only
        if driver / device / connection params changed."""
        new = self.normalize(cfg)
        with self._lock:
            reopen = any(new[k] != self._cfg.get(k) for k in
                         ("driver", "device", "serial_baud", "serial_protocol",
                          "on_cmd", "off_cmd", "modbus_unit", "modbus_offset"))
            was_enabled = self._cfg.get("enabled")
            self._cfg = new
            if reopen or (not new["enabled"] and was_enabled):
                self._close_driver()
            if reopen or (new["enabled"] and not was_enabled):
                self._last_open_attempt = 0.0   # allow immediate reconnect
                self.last_error = None          # stale error belongs to the old device
        state = "ON" if new["enabled"] else "OFF"
        logger.info(f"[DIO] {state} · driver={new['driver']} device={new['device']} "
                    f"pulse={new['pulse_ms']}ms map={new['map']}")

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.get("enabled"))

    @property
    def config(self) -> dict:
        with self._lock:
            c = dict(self._cfg)
            c["map"] = dict(c["map"])
            return c

    # ── publishing ───────────────────────────────────────────────────────
    def pulse(self, class_name: str) -> bool:
        """Queue one pulse for ``class_name``. Returns True if queued.
        Non-blocking; never raises. Called from the inference loop."""
        if not self.enabled:
            return False
        ch = self._cfg["map"].get(str(class_name), 0)
        if not ch:
            return False   # class not wired
        return self._enqueue(ch, str(class_name))

    def test_pulse(self, channel: int) -> bool:
        """Queue a pulse on an explicit channel (Settings "Test" button).
        Works even when disabled so the automaticien can commission the wire
        before switching the feature on."""
        ch = int(channel)
        if not (1 <= ch <= 255):
            raise ValueError("channel must be 1-255")
        with self._lock:
            self._last_open_attempt = 0.0
        return self._enqueue(ch, "test")

    def _enqueue(self, ch: int, cls: str) -> bool:
        try:
            self._q.put_nowait((time.perf_counter(), ch, cls))
            return True
        except queue.Full:
            self.dropped += 1
            self.last_error = "queue full — device too slow or dead"
            return False

    # ── worker ───────────────────────────────────────────────────────────
    def _open_driver(self) -> bool:
        now = time.monotonic()
        if self._driver is not None and self._connected:
            return True
        if now - self._last_open_attempt < _RETRY_OPEN_S:
            return False
        self._last_open_attempt = now
        try:
            with self._lock:
                cfg = dict(self._cfg)
            drv = _DRIVERS[cfg["driver"]](cfg)
            drv.open()
            self._driver, self._connected = drv, True
            self.last_error = None
            logger.info(f"[DIO] connected · {cfg['driver']} {cfg['device']}")
            return True
        except Exception as e:  # noqa: BLE001 — surfaced in state(), never fatal
            self._connected = False
            self._driver = None
            self.last_error = f"open failed: {e}"
            self.errors += 1
            # Rate-limit: first failure, then once every 30 s — an unplugged
            # board must not flood the log at the 2 s retry cadence.
            if now - self._last_open_warn >= 30.0:
                self._last_open_warn = now
                logger.warning(f"[DIO] {self.last_error} (retrying every {_RETRY_OPEN_S:.0f}s)")
            return False

    def _close_driver(self) -> None:
        drv, self._driver, self._connected = self._driver, None, False
        if drv is not None:
            try:
                drv.close()
            except Exception:  # noqa: BLE001
                pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                t_enq, ch, cls = self._q.get(timeout=0.25)
            except queue.Empty:
                # Idle: keep the device connected while enabled so the first
                # crossing never pays the connect cost, and a board that was
                # unplugged/re-plugged is picked up again (2 s back-off).
                if self.enabled and not self._connected:
                    self._open_driver()
                continue
            ok = False
            try:
                if not self._open_driver():
                    self.errors += 1
                    continue          # last_error already set by _open_driver; finally() logs it
                drv = self._driver
                drv.set(ch, True)
                t_on = time.perf_counter()
                self._ch_last[ch] = (time.monotonic(), cls)
                self._ch_fired[ch] = self._ch_fired.get(ch, 0) + 1
                self._latency_us.append((t_on - t_enq) * 1e6)
                width = self._cfg["pulse_ms"] / 1000.0
                # hold the pulse — sleep only the remainder so the write cost
                # doesn't stretch the width
                remaining = width - (time.perf_counter() - t_on)
                if remaining > 0:
                    time.sleep(remaining)
                drv.set(ch, False)
                ok = True
                self.fired += 1
                self.last_error = None
                self.last_channel, self.last_class = ch, cls
                self.last_ts = datetime.now().isoformat()
            except Exception as e:  # noqa: BLE001
                self.errors += 1
                self.last_error = f"pulse ch{ch} failed: {e}"
                logger.warning(f"[DIO] {self.last_error}")
                self._close_driver()
            finally:
                self.recent.append((datetime.now().isoformat(), cls, ch, ok))
        self._close_driver()

    # ── telemetry ────────────────────────────────────────────────────────
    @staticmethod
    def _pct(d, p):
        if not d:
            return None
        s = sorted(d)
        return round(s[min(len(s) - 1, int(len(s) * p / 100))], 1)

    def state(self) -> dict:
        cfg = self.config
        with self._lock:
            lat = list(self._latency_us)
        return {
            "enabled": cfg["enabled"],
            "driver": cfg["driver"],
            "device": cfg["device"],
            "pulse_ms": cfg["pulse_ms"],
            "map": cfg["map"],
            "connected": bool(self._connected),
            "fired": self.fired,
            "errors": self.errors,
            "dropped": self.dropped,
            "queue_depth": self._q.qsize(),
            "last_error": self.last_error,
            "last_channel": self.last_channel,
            "last_class": self.last_class,
            "last_ts": self.last_ts,
            "p50_us": self._pct(lat, 50),
            "p95_us": self._pct(lat, 95),
            "max_us": round(max(lat), 1) if lat else None,
            "recent": list(self.recent)[-10:],
            "status": self._status(),
        }

    def channels(self) -> dict:
        """Public, lightweight per-channel view for the Live page LEDs:
        ``{"1": {"class": "carton", "fired": 12, "age_ms": 143}, ...}``.
        Channels come from the class map (so an idle channel still shows),
        ``age_ms`` is time since the last ON edge (None = never)."""
        cfg = self.config
        now = time.monotonic()
        out = {}
        for cls, ch in cfg["map"].items():
            if not ch:
                continue
            last = self._ch_last.get(ch)
            out[str(ch)] = {
                "class": cls,
                "fired": self._ch_fired.get(ch, 0),
                "age_ms": int((now - last[0]) * 1000) if last else None,
            }
        return {"enabled": cfg["enabled"], "connected": bool(self._connected),
                "driver": cfg["driver"], "pulse_ms": cfg["pulse_ms"], "channels": out}

    def _status(self) -> str:
        """Traffic light for the dashboard: green (off, or on+connected and
        last pulse OK), yellow (on+connected but last pulse failed), red (on
        but device unreachable)."""
        if not self.enabled:
            return "green"
        if not self._connected:
            return "red" if self.last_error else "yellow"   # yellow = connecting
        return "yellow" if self.last_error else "green"

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=2.0)
        self._close_driver()

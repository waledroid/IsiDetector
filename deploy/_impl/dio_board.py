#!/usr/bin/env python3
"""On-screen relay board — the far end of the "wire", without hardware.

  dio_board.py serial 0.0.0.0 4196   # acts like an Ethernet-serial relay (Numato ASCII / LCUS hex)
  dio_board.py modbus 0.0.0.0 1502   # acts like a Modbus/TCP relay (FC05 coils, echo = ack)

Point the app at it (Settings → Sorter (electrical pulse) → device
``socket://<this-ip>:4196`` or ``<this-ip>:1502``) and every relay ON/OFF
prints here with the measured pulse width — the wire equivalent of the UDP
receiver window. Also the acceptance test for a real board: identical bytes.
"""
import socket, struct, sys, time, threading, re, os

mode, host, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
log = open(os.environ.get("DIO_BOARD_LOG", "/dev/null"), "a", buffering=1)
on_at = {}
def emit(ch, on, extra=""):
    t = time.perf_counter(); wall = time.strftime("%H:%M:%S") + f".{int((time.time()%1)*1000):03d}"
    if on:
        on_at[ch] = t; line = f"{wall}  CH{ch} \033[92m● ON \033[0m {extra}"
    else:
        w = (t - on_at.pop(ch, t)) * 1000; line = f"{wall}  CH{ch} ○ OFF  width={w:.1f} ms  {extra}"
    print(line, flush=True); log.write(line + "\n")

def serve_serial(conn, peer):
    print(f"[serial] client {peer}", flush=True); log.write(f"client {peer}\n")
    buf = b""
    while True:
        d = conn.recv(256)
        if not d: break
        buf += d
        while b"\r" in buf:
            frame, buf = buf.split(b"\r", 1)
            m = re.match(rb"relay (on|off) (\d+)", frame)
            if m: emit(int(m.group(2)) + 1, m.group(1) == b"on", f"raw={frame!r} from={peer[0]}")
            else: print("??", frame, flush=True)
        # LCUS 4-byte hex frames
        while len(buf) >= 4 and buf[0] == 0xA0:
            emit(buf[1], buf[2] == 1, f"raw={buf[:4].hex()} from={peer[0]}"); buf = buf[4:]
    print(f"[serial] bye {peer}", flush=True)

def serve_modbus(conn, peer):
    print(f"[modbus] client {peer}", flush=True); log.write(f"client {peer}\n")
    while True:
        h = b""
        while len(h) < 12:
            c = conn.recv(12 - len(h))
            if not c: return
            h += c
        tid, pid, ln, uid, fc, addr, val = struct.unpack(">HHHBBHH", h)
        if fc == 5:
            emit(addr + 1, val == 0xFF00, f"unit={uid} coil={addr} from={peer[0]}")
            conn.sendall(h)          # FC05 echo = ack
        else:
            conn.sendall(h[:7] + bytes([fc | 0x80, 0x01]))

srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((host, port)); srv.listen(4)
print(f"[relay board · {mode}] listening on {host}:{port} — waiting for pulses (Ctrl-C to quit)", flush=True)
while True:
    c, peer = srv.accept(); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    threading.Thread(target=serve_serial if mode == "serial" else serve_modbus, args=(c, peer), daemon=True).start()

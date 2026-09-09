# Digital Output — relay pulse on a wire

The **electrical** trigger that runs alongside the UDP datagram. On every line
crossing the system drives a relay channel ON for `dio_pulse_ms` then OFF —
a plain digital input for the sorter PLC, exactly like a photocell.

| | UDP datagram | Relay pulse (this page) |
|---|---|---|
| What the PLC gets | `{"class","id","ts"}` JSON | a contact closure on one input |
| Needs | IP, port, subnet, firewall | a wire |
| Carries | class + tracker ID + timestamp | class only (one channel per class) |
| Fails when | network misconfigured / changed | board unplugged (dashboard shows red) |

Use **both**: the pulse is the trigger the machine can't miss; the datagram is
the information the pulse can't carry.

---

## Settings (dev-gated, apply immediately on Save)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `dio_enabled` | bool | `false` | Master toggle. Takes effect on the **next crossing** — no stream restart. |
| `dio_driver` | `serial` \| `modbus_tcp` \| `sim` | `sim` | Board type. `sim` = no hardware (demo / dry-run). |
| `dio_device` | string | `/dev/ttyUSB0` | Serial: tty path **or** pyserial URL `socket://IP:PORT` (Ethernet-serial relay). Modbus: `IP[:502]`. |
| `dio_pulse_ms` | int 5–5000 | `50` | Pulse width. Must be **shorter than the minimum gap between two parcels** or pulses merge. 50 ms is safe at 1 m/s. |
| `dio_map` | `{class: channel}` | `{"carton":1,"polybag":2}` | Channel per class (1-based). `0` = not wired. |
| `dio_serial_protocol` | `numato` \| `lcus` \| `custom` | `numato` | `numato`: `relay on 0\r`. `lcus`: CH340 boards `A0 01 01 A2`. `custom`: templates below. |
| `dio_serial_baud` | int | `9600` | Standard rates only. |
| `dio_on_cmd` / `dio_off_cmd` | string | `relay on {ch0}\r` | Custom templates. Placeholders `{ch}` (1-based), `{ch0}` (0-based), `{chx}`/`{ch0x}` (hex). Escapes `\r \n \xNN` allowed. |
| `dio_modbus_unit` | int | `1` | Modbus unit ID. |
| `dio_modbus_offset` | int | `0` | Coil address = channel − 1 + offset. |

### Seeing the pulse without hardware

- **Live page → "Sorter outputs" LEDs**: `UDP` flashes on every datagram, `WIRE` shows
  the relay state and one LED per channel (`CH1 CARTON`, `CH2 POLYBAG`) that lights on
  each pulse — like the LEDs on a real relay board (held ~0.7 s for visibility; the real
  pulse is `dio_pulse_ms`).
- **`./io.sh board`** — an on-screen relay board (serial or Modbus emulator). Point the
  app at `socket://<ip>:4196` (or `<ip>:1502` for Modbus) and every `CH1 ● ON … ○ OFF
  width=50 ms` prints in the terminal — the wire equivalent of the UDP receiver window.
- **`./io.sh test <ch>`**, **`./io.sh on|off`**, **`./io.sh status`** — commissioning
  from the shell (asks the dev password, or `DEV_PASSWORD=…`).

### Test buttons

**⚡ Test carton / ⚡ Test polybag** fire one pulse on that channel — **even while
`dio_enabled` is OFF** — so the automaticien can check the wiring and the PLC
input before enabling. Also available as `POST /api/dio/test {"channel": 1}`.

---

## Docker: passing a USB relay into the container

Ethernet relays (`socket://…`, Modbus) need nothing. USB boards need device
passthrough — `up.sh` adds `deploy/docker-compose.dio.yml` automatically when
`/dev/ttyUSB0` exists or `DIO_DEVICE` is set:

```bash
DIO_DEVICE=/dev/ttyUSB0 ./up.sh       # explicit
./up.sh                               # auto if /dev/ttyUSB0 exists
```

Pin the device name across reboots with a udev rule (else it may come back as
`ttyUSB1`):

```bash
# /etc/udev/rules.d/99-isitec-relay.rules  (find idVendor/idProduct with lsusb)
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="relay0"
# then: DIO_DEVICE=/dev/relay0 ./up.sh
```

---

## Dashboard: "Relay Pulse (wire)" group

| Light | Meaning |
|---|---|
| Green | OFF, or ON and device connected |
| Yellow | ON, errors seen (last error shown) |
| Red | ON but device unreachable |

Rows: state, driver/device, pulse width, pulses fired, errors (+dropped),
queue depth, enqueue→ON p50/p95, last pulse, last error. Footer shows
`DIO → ch1 CARTON @ hh:mm:ss` next to the UDP indicator.

Timing model: pulses are serialised by a worker thread — two crossings in one
frame produce two back-to-back pulses on the same channel, never overlapped.
Enqueue→ON is typically < 1 ms on serial/USB, ~1–2 ms on Modbus (with ack).

---

## Wiring notes for the automaticien

- One relay channel per class → one PLC digital input per class.
- Use the relay's **NO** contact; PLC input sees a `dio_pulse_ms` closure per parcel.
- Prefer solid-state relays for longevity at parcel rates (no contact wear).
- The Modbus driver reads the FC05 echo, so a Modbus pulse is **acknowledged**;
  serial boards are fire-and-forget (watch the dashboard *errors* row).

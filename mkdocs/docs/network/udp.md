# UDP Sorter Protocol

How IsiDetector tells the sorting machine that a parcel just crossed the counting line.

On every line crossing, the web container fires **one UDP datagram** to the sorter controller. The datagram is small (~70 bytes), fire-and-forget, and sent the instant the parcel's leading edge hits the line.

## The datagram

Each crossing sends one JSON object:

```json
{"class": "carton", "seq": 17, "id": 42, "ts": "2026-03-31T14:23:45.312847"}
```

| Field | Type | Meaning |
|---|---|---|
| `class` | string | What crossed — `carton` or `polybag`. Act on this. |
| `seq` | int | Gap-free counter. Increments by exactly 1 per datagram sent. |
| `id` | int | ByteTrack tracker ID of the object. Use for dedup/tracing only. |
| `ts` | string | Send time, ISO format with microseconds. |

!!! note "Two close parcels = two datagrams"
    The line fires per crossing, not per frame. If two parcels cross in the same frame, the sorter gets two datagrams — one per object.

!!! warning "`seq` and `id` are not the same — don't confuse them"
    - **`seq`** is gap-free by design. Use it to detect lost datagrams: if you receive `...15, 16, 18...`, datagram `17` was lost on the network.
    - **`id`** is the tracker ID and is **NOT** sequential. ByteTrack assigns IDs to every tracked object, including ones that never cross the line, so gaps in `id` are completely normal and do **not** mean a datagram was lost.

    `id` is optional (omitted in rare legacy paths). A consumer that reads only `class` keeps working.

## Default target

The default destination is:

```
10.0.0.1:9502
```

This is the canonical site PLC address. Change it for your site via the priority chain below — do not edit source.

## Where the target comes from (priority)

Highest wins:

| Priority | Source | How to set | Takes effect |
|---|---|---|---|
| 1 (highest) | Live API | `POST /api/udp {"host": "...", "port": ...}` | Immediately, no restart |
| 2 | Environment | `UDP_HOST` / `UDP_PORT` (compose `.env`) | On container restart |
| 3 | YAML | `isidet/configs/train.yaml` → `inference.udp.host/port` | On container restart |
| 4 (lowest) | Default | built-in | `10.0.0.1:9502` |

!!! tip "Use the Settings UI, not source edits"
    The Settings → Camera panel writes `udp_host` / `udp_port` and live-retargets the running stream (priority 1, persisted). That is the normal way to point a site PC at its sorter. See [Counting & Line Settings](../settings/counting.md).

## Minimal consumer (sorter side)

Drop-in receiver for the automation engineer. Blocks until a parcel crosses — no polling.

```python
import socket, json

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 9502))          # listen on the sorter's UDP port

last_seq = None
while True:
    data, _ = sock.recvfrom(1024)     # blocks until an event arrives
    event = json.loads(data)

    # Optional: detect lost datagrams via the gap-free seq
    if last_seq is not None and event["seq"] != last_seq + 1:
        print(f"lost {event['seq'] - last_seq - 1} datagram(s)")
    last_seq = event["seq"]

    trigger_sort_gate(event["class"])  # act on "carton" or "polybag"
```

## Verify egress from the site PC

Before going live, confirm the site PC can actually send to the sorter. `net.sh test` runs a live UDP egress probe to the configured target:

```bash
./net.sh test
```

This sends a probe datagram and reports whether it left the host toward `10.0.0.1:9502` (or your configured target). For NIC freeze, gateway, and full network diagnostics see [Network Setup](network.md).

!!! note "UDP is fire-and-forget"
    A successful egress probe means the datagram left the PC — it does not confirm the sorter received it. Confirm receipt on the controller side using the consumer snippet above, watching `seq` advance.

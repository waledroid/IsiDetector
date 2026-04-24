# Industrial Analytics & Logging

The `EventLogger` provides a robust telemetry layer for industrial parcel sorting. Every line-crossing is recorded as its own row, so the operator dashboard and downstream BI tools can slice counts by second, hour, or day without losing resolution.

---

## The EventLogger

:material-file-code: **Source**: `isidet/src/utils/event_logger.py`

### How it Works
The logger is designed to be "zero-maintenance." It automatically handles:

1. **Per-event rows** — one CSV append per line-crossing, written from `VisionEngine.process_frame()` the moment a new ByteTrack ID crosses the line.
2. **Daily rotation** — files are named `events_YYYY-MM-DD.csv`; the first write past midnight opens a new file.
3. **Rolling retention** — files older than `retention_days` (default 30) are deleted on init and on every midnight rollover, so `isidet/logs/events/` never grows unbounded.
4. **Thread safety** — a single internal lock serialises writes and rollover checks, so the inference thread and any diagnostic thread can log concurrently.

### CSV Structure
Each row is one event. The file opens in Excel / BI tools with no extra parsing:

| ts | class | id |
|---|---|---|
| 2026-04-23T09:00:14.312847 | carton | 42 |
| 2026-04-23T09:00:15.118201 | polybag | 43 |
| 2026-04-23T09:00:16.882310 | carton | 44 |

`ts` is ISO-8601 with microsecond precision — the same timestamp the UDP datagram carries, so sorter telemetry can be correlated one-to-one with logged events.

---

## Configuration
You can control the behavior of the analytics layer via `isidet/configs/train.yaml`:

```yaml
inference:
  logging:
    log_dir: "isidet/logs"  # events land in <log_dir>/events/
    retention_days: 30      # files older than this are pruned on rollover
```

!!! info "Why event-level?"
    The previous hourly-snapshot design stored cumulative counts, which made the `/api/chart` endpoint reconstruct per-bucket deltas by diffing rows — fragile across midnight resets and opaque for audits. Event-level storage turns the chart into a plain bucket-count, and gives compliance/audit a row-per-parcel ledger for free. SSD write amplification stays low: ~60 bytes per event, and crossings are a few per second at peak — orders of magnitude under any modern drive's endurance.

---

## API Reference

::: src.utils.event_logger.EventLogger

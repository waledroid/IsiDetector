# Analytics & Event Logs

Every line-crossing is recorded to a daily CSV file; the dashboard chart and report endpoints read directly from those files.

---

## Event CSV

### Location

```
isidet/logs/events/events_YYYY-MM-DD.csv
```

Inside the container this resolves to `/opt/isitec/isidet/logs/events/`. On the host it is bind-mounted there from the repo checkout.

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `ts` | ISO-8601 (microseconds) | Timestamp of the crossing — same value sent in the UDP datagram |
| `class` | string | `carton` or `polybag` |
| `id` | integer | ByteTrack tracker ID; empty if unavailable |
| `seq` | integer | Monotonic per-stream datagram counter; empty on legacy rows |

Example rows:

```
ts,class,id,seq
2026-06-03T08:14:32.481203,carton,17,1
2026-06-03T08:14:33.107841,polybag,18,2
2026-06-03T08:14:33.882110,carton,19,3
```

The `seq` value is the same counter sent over UDP. Use it to reconcile logged events against what the sorter received. See [UDP sorting broadcast](../network/udp.md) for the wire format.

!!! note "Legacy compatibility"
    Older files may have only three columns (`ts,class,id`). All read paths tolerate missing `seq` and treat it as empty.

### Rotation and retention

- A new file is created on the first write past midnight.
- Files older than **30 days** are deleted automatically — on container start and on every midnight rollover.
- No manual cleanup is needed.

---

## Dashboard chart — `/api/chart`

The **Chart** tab in the UI calls this endpoint. You can also query it directly:

| `period` | Window | Bucket size |
|----------|--------|-------------|
| `live` | Current session counts (in-memory) | — |
| `24h` | Last 24 hours | 1 hour |
| `7d` | Last 7 days | 1 day |
| `30d` | Last 30 days | 1 day |

```bash
# Example — last 7 days from the host
curl http://localhost:9501/api/chart?period=7d
```

Response shape:

```json
{
  "status": "success",
  "view": "timeseries",
  "buckets": ["Mon 27", "Tue 28", "…"],
  "series": {
    "carton":  [142, 198, 211, 176, 203, 187, 95],
    "polybag": [67,  91,  84,  79,  88,  74,  41]
  }
}
```

---

## Report endpoint — `/api/report`

Produces a summary over a chosen period: totals, class mix, peak hour, throughput, and session stats.

```bash
curl "http://localhost:9501/api/report?period=today"
curl "http://localhost:9501/api/report?period=7d"
curl "http://localhost:9501/api/report?period=custom&from=2026-05-01&to=2026-05-31"
```

| `period` value | Window |
|----------------|--------|
| `today` | Current calendar day |
| `yesterday` | Previous calendar day |
| `7d` | Last 7 days |
| `30d` | Last 30 days |
| `custom` | Requires `from=YYYY-MM-DD` and `to=YYYY-MM-DD` |

Response includes:

- `counts` — per-class totals and grand total
- `mix_pct` — percentage split between carton and polybag
- `peak` — busiest hour (`{ "hour": "2026-06-03 09:00", "events": 87 }`)
- `throughput_per_hour` — average crossings per hour over the window
- `avg_fps`, `total_runtime_h`, `sessions_count` — from the session log

---

## Export to CSV — `/api/events/export`

Download a date-range slice of the event log as a CSV file. Both `from` and `to` are required.

```bash
# Download all events for May 2026
curl -O "http://localhost:9501/api/events/export?from=2026-05-01&to=2026-05-31"
```

The downloaded file is named `events_<from>_to_<to>.csv` and has the same four-column format as the on-disk logs (`ts,class,id,seq`). The `seq` column lets you cross-check against the sorter's received datagram log.

!!! tip "Opening in Excel"
    The file is UTF-8 CSV with ISO-8601 timestamps. In Excel, use **Data → From Text/CSV** and set the delimiter to comma. The `ts` column imports correctly as a datetime when formatted as `yyyy-mm-ddThh:mm:ss.000000`.

---

## Reading the files directly

If you need to inspect logs outside the container:

```bash
# List available event files
ls isidet/logs/events/

# Count today's crossings by class
grep ',carton,'  isidet/logs/events/events-$(date +%F).csv | wc -l
grep ',polybag,' isidet/logs/events/events-$(date +%F).csv | wc -l

# Copy a log out of the container
docker compose exec web cat /opt/isitec/isidet/logs/events/events_2026-06-03.csv \
  > /tmp/events_2026-06-03.csv
```

!!! warning "Do not delete log files while the stream is running"
    The inference thread holds the file open in append mode. Deleting a day's file mid-session will cause that day's events to be lost. Stop the stream first, or wait for midnight rollover.

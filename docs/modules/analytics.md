# Industrial Analytics & Logging

The `DailyLogger` provides a robust telemetry layer for industrial parcel sorting. It ensures that every detection is recorded accurately for long-term auditing and business intelligence.

---

## The DailyLogger

:material-file-code: **Source**: `src/utils/analytics_logger.py`

### How it Works
The logger is designed to be "zero-maintenance." It automatically handles:

1. **Daily Rotation**: Every day at midnight, it creates a new CSV file (e.g., `report_27-03-2026.csv`).
2. **Hourly Snapshots**: Instead of writing to disk every millisecond (which kills SSD life), it takes a snapshot of the totals according to your `save_interval` (Default: 1 Hour).
3. **Graceful Persistence**: Even if the system crashes or is stopped manually, the logger captures the final state before closing.

### CSV Structure
The generated reports are standard CSV files that can be opened in Excel, Google Sheets, or any BI tool (PowerBI/Tableau).

| Time | CARTON | POLYBAG |
|---|---|---|
| 09:00:00 | 45 | 12 |
| 10:00:00 | 112 | 34 |
| 11:00:00 | 258 | 89 |

---

## Configuration
You can control the behavior of the analytics layer via `configs/train.yaml`:

```yaml
inference:
  logging:
    enabled: true
    save_interval: 3600    # Time in seconds (3600 = 1 Hour)
    log_dir: "logs"        # Destination folder
```

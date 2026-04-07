import csv
import time
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class DailyLogger:
    """Hourly CSV data logger with automatic midnight file rotation.

    Appends one row per save interval to a daily CSV file named
    ``report_DD-MM-YYYY.csv``. Rotates to a new file automatically
    when the date changes, enabling long multi-day sessions.

    CSV format::

        Time,carton,polybag
        08:00:00,142,87
        09:00:00,298,165

    Args:
        class_names: List of class name strings
            (e.g. ``['carton', 'polybag']``). Becomes the CSV header.
        log_dir: Directory for CSV files. Created if it does not exist.
            Defaults to ``"logs"``.
        save_interval: Seconds between auto-saves. Defaults to
            ``3600`` (hourly). Reduce for higher-frequency snapshots.
    """

    def __init__(self, class_names: list, log_dir: str = "logs", save_interval: int = 3600):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval = save_interval 
        self.last_save_time = time.time()
        self.current_date = datetime.now().strftime("%d-%m-%Y")
        
        # Define CSV headers: Time + all class names
        self.headers = ["Time"] + [name for name in class_names]

    def _get_filepath(self) -> Path:
        """Returns the current day's CSV file, updating the date if midnight passes."""
        new_date = datetime.now().strftime("%d-%m-%Y")
        if new_date != self.current_date:
            self.current_date = new_date
            logger.info(f"🌒 Midnight Rollover: Starting new log for {self.current_date}")
            
        return self.log_dir / f"report_{self.current_date}.csv"

    def update(self, counts: dict):
        """Check elapsed time and save if the interval has passed.

        Designed to be called on every frame — the check is a single
        ``time.time()`` comparison so overhead is negligible at 30 fps.

        Args:
            counts: Current session totals dict
                (e.g. ``{'carton': 42, 'polybag': 18}``).
        """
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            self.save(counts, auto=True)
            self.last_save_time = current_time

    def save(self, counts: dict, auto: bool = False):
        """Write the current counts as a timestamped row to the CSV file.

        Handles midnight date rollover automatically. Writes the CSV
        header on first write to a new file. Errors are logged but
        never re-raised — a failed save does not crash the session.

        Args:
            counts: Session totals dict to snapshot.
            auto: ``True`` when called by the interval timer (logged as
                "Auto-Snapshot"); ``False`` for a manual final save on
                session stop.
        """
        filepath = self._get_filepath()
        file_exists = filepath.exists()
        
        # Format the row data
        timestamp = datetime.now().strftime("%H:%M:%S")
        row = {"Time": timestamp}
        for name in self.headers[1:]:
            row[name] = counts.get(name, 0)

        try:
            with open(filepath, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                
                # Write header if it's a brand new file
                if not file_exists:
                    writer.writeheader()
                    
                writer.writerow(row)
                
            trigger_type = "⏳ Auto-Snapshot" if auto else "🛑 Final Stream Save"
            logger.info(f"💾 {trigger_type} to {filepath.name}: {counts}")
            
        except Exception as e:
            logger.error(f"⚠️ Failed to write to log: {e}")

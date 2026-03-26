import csv
import time
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class DailyLogger:
    """
    Industrial Data Logger for IsiDetector.
    Handles hourly snapshots, daily file rotation, and graceful exits.
    """
    def __init__(self, class_names: list, log_dir: str = "logs", save_interval: int = 3600):
        """
        :param class_names: List of strings (e.g., ['carton', 'polybag'])
        :param log_dir: Folder to save the CSV reports
        :param save_interval: Seconds between auto-saves (Default: 3600 = 1 hour)
        """
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
        """Called every frame. Triggers a save only if the interval has passed."""
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            self.save(counts, auto=True)
            self.last_save_time = current_time

    def save(self, counts: dict, auto: bool = False):
        """Forces a save to the CSV file. Usually called on exit or by update()."""
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

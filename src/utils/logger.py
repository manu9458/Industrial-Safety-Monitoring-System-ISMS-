import csv
import os
import datetime
import logging

class ActivityLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self._ensure_log_directory()
        self._init_csv()
        
        # Setup standard logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.console = logging.getLogger("IndustrialMonitor")

    def _ensure_log_directory(self):
        directory = os.path.dirname(self.log_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def _init_csv(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Event Type", "Details", "Count"])

    def log_event(self, count, event_type, details=""):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write to CSV
        try:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, event_type, details, count])
        except IOError as e:
            self.console.error(f"Failed to write to CSV: {e}")

    def info(self, message):
        self.console.info(message)

    def warning(self, message):
        self.console.warning(message)

    def error(self, message):
        self.console.error(message)

import csv
import os
import datetime

class ActivityLogger:
    def __init__(self, log_file='activity_log.csv'):
        self.log_file = log_file
        self.init_logger()

    def init_logger(self):
        """Initializes the CSV log file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Motion Magnitude (Area)", "Status"])

    def log(self, area, status):
        """Saves event data to CSV."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, area, status])

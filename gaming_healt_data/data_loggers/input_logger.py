import csv
import time
import threading

class InputLogger:
    def __init__(self, filename):
        self.filename = filename
        self.csv_file = open(self.filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Time (ms)', 'Device', 'Action', 'Details'])

    def close(self):
        self.csv_file.close()

    def log(self, device, action, details):
        time_ms = int(time.time() * 1000)  # Get time in milliseconds
        self.csv_writer.writerow([time_ms, device, action, details])

import csv
from datetime import datetime
import threading

class InputLogger:
    def __init__(self, filename):
        self.filename = filename
        self.csv_file = open(self.filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Time', 'Device', 'Action', 'Details'])

    def close(self):
        self.csv_file.close()

    def log(self, device, action, details):
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.csv_writer.writerow([time, device, action, details])

import pandas as pd
import numpy as np

class EKGAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.ekg_data = None

    def load_data(self):
        self.ekg_data = pd.read_csv(self.file_path, header=None, names=["Timestamp", "HeartSignal"])

    def calculate_bpm(self, threshold=1.5):
        peaks = self.ekg_data.loc[self.ekg_data["HeartSignal"] > threshold]
        time_diff = np.diff(peaks["Timestamp"])
        bpm = 60 / (time_diff / 1000)
        return bpm

    def calculate_moving_avg_bpm(self, window_size=10):
        bpm = self.calculate_bpm()
        moving_avg_bpm = bpm.rolling(window=window_size).mean()
        return moving_avg_bpm

    def analyze(self, threshold=1.5, window_size=10):
        self.load_data()
        moving_avg_bpm = self.calculate_moving_avg_bpm(window_size=window_size)
        return moving_avg_bpm
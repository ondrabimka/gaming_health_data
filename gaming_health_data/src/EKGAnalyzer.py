# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Literal
import ast
from scipy import signal
from sklearn.preprocessing import StandardScaler

@pd.api.extensions.register_dataframe_accessor("EKG")
class EKGAnalyzer:
    """
    A class for analyzing EKG (Electrocardiogram) data.

    Parameters:
    -----------
    file_path : str
        The path to the CSV file containing EKG data.

    Attributes:
    -----------
    ekg_data : pandas.DataFrame
        The EKG data loaded from the CSV file.

    Methods:
    --------
    calculate_bpm(threshold=1.5)
        Calculates the beats per minute (BPM) from the EKG data.

    calculate_moving_avg_bpm(window_size=10)
        Calculates the moving average of the BPM.

    analyze(threshold=1.5, window_size=10)
        Analyzes the EKG data by calculating the moving average of the BPM.
    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if "HeartSignal" not in obj.columns:
            raise AttributeError("HeartSignal column is missing")
        if "Timestamp" not in obj.columns:
            raise AttributeError("Timestamp column is missing")
        
    # sensor type AD8232 or Polar H10
    @staticmethod
    def read_file(file_path, sensor_type: Literal["AD8232", "PolarH10"] = "AD8232", **kwargs):

        """
        Reads EKG data from a CSV file captured by the AD8232 or Polar H10 sensor.

        Parameters:
        -----------
        file_path : str
            The path to the CSV file containing EKG data.

        sensor_type : str
            The type of the sensor used to capture the EKG data. It can be either "AD8232" or "PolarH10".

        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        ekg_data : pandas.DataFrame
            The EKG data loaded from the CSV file.
        """

        assert sensor_type in ["AD8232", "PolarH10"], "Invalid sensor type"

        if sensor_type == "AD8232":
            return EKGAnalyzer.read_file_AD8232(file_path, **kwargs)
        elif sensor_type == "PolarH10":
            return EKGAnalyzer.read_file_polar_h10(file_path, **kwargs)
        else:
            raise ValueError("Invalid sensor type")
        

    @classmethod
    def read_files(cls, files, sensor_type: Literal["AD8232", "PolarH10"] = "AD8232", as_list=False, **kwargs):

        """
        Reads multiple EKG data files captured by the specified sensor type.
        
        Parameters:
        -----------
        files : list of str
            List of file paths to the EKG data files, or directory containing the files.
            
        sensor_type : str
            The type of the sensor used to capture the EKG data. It can be either "AD8232" or "PolarH10".

        **kwargs : dict
            Additional keyword arguments for the read_file method.

        Returns:
        --------
        ekg_data : pandas.DataFrame
            The combined EKG data loaded from the specified files.
        """
        
        if isinstance(files, str):
            files = [files]
        
        ekg_data_list = []
        for file in files:
            ekg_data = EKGAnalyzer.read_file(file, sensor_type=sensor_type, **kwargs)
            ekg_data_list.append(ekg_data)

        if as_list:
            return ekg_data_list

        return pd.concat(ekg_data_list, ignore_index=True)     
    

    @staticmethod
    def read_file_polar_h10(file_path):

        """
        Reads EKG data from a CSV file captured by the Polar H10 sensor.

        Parameters:
        -----------
        file_path : str
            The path to the CSV file containing EKG data.

        Returns:
        --------
        ekg_data : pandas.DataFrame
            The EKG data loaded from the CSV file.
        """

        # Read the file
        with open(file_path, "r") as f: 
            lines = f.readlines()

        parsed_data = []
        for line in lines:
            entry = ast.literal_eval(line.strip())  # Convert to tuple
            parsed_data.append(entry)

        ekg_data = pd.DataFrame(parsed_data, columns=["Signal Type", "Timestamp", "Values"])
        ekg_data["Timestamp"] = pd.to_datetime(ekg_data["Timestamp"], unit="ns")
        
        signal = []
        for samples in ekg_data["Values"]:
            signal.extend(samples)
        
        signal_mV = [x * 1e-3 for x in signal]
        t = [x / 130.0 for x in range(len(signal_mV))]

        # create new df from the signal and t
        transformed_data = pd.DataFrame({"Timestamp": t, "HeartSignal": signal_mV})
        transformed_data.measure_start_date = ekg_data["Timestamp"].iloc[0]
        transformed_data.measure_end_date = ekg_data["Timestamp"].iloc[-1]
        return transformed_data

    @staticmethod
    def read_file_AD8232(file_path, to_seconds=True, solve_time_reset=True, move_to_zero=True):

        """
        Reads EKG data from a CSV file captured by the AD8232 sensor.
        
        Parameters:
        -----------
        file_path : str
            The path to the CSV file containing EKG data.
            
        to_seconds : bool, optional
            Whether to convert the timestamps to seconds. Default is True.
        
        solve_time_reset : bool, optional
            Whether to solve the time reset issue. Default is True.
            
        move_to_zero : bool, optional
            Whether to move the timestamps to zero. Default is True.
            
        Returns:
        --------
        ekg_data : pandas.DataFrame
            The EKG data loaded from the CSV file.     
        """

        # TODO: To separate methods
        ekg_data = pd.read_csv(file_path, header=None, names=["Timestamp", "HeartSignal"])
        if to_seconds:
            ekg_data['Timestamp'] = ekg_data['Timestamp'] / 1e6
        if solve_time_reset:
            ekg_data['adjusted_timestamp'] = ekg_data['Timestamp']
            offset = 0
            # Iterate over the rows
            for i in range(1, len(ekg_data)):
                # If the current timestamp is less than the previous one, update 'offset'
                if ekg_data.loc[i, 'Timestamp'] < ekg_data.loc[i-1, 'Timestamp']:
                    offset = ekg_data.loc[i-1, 'adjusted_timestamp']
                # Add 'offset' to 'adjusted_timestamp'
                ekg_data.loc[i, 'adjusted_timestamp'] = ekg_data.loc[i, 'Timestamp'] + offset
            ekg_data.drop(columns=['Timestamp'], inplace=True)
            ekg_data.rename(columns={'adjusted_timestamp': 'Timestamp'}, inplace=True)
        if move_to_zero:
            ekg_data['Timestamp'] = ekg_data['Timestamp'] - ekg_data['Timestamp'].iloc[0]
        return ekg_data

    def calculate_bpm(self):
        """
        Calculates the beats per minute (BPM) from the EKG data.
        Original timestamps are in microseconds, so the window size should be in microseconds.

        Returns:
        --------
        bpm : numpy.ndarray
            An array of BPM values calculated from the EKG data.
        """
        bpm = 60 / np.diff(self._obj["Timestamp"].iloc[self.beats])
        return bpm

    def calculate_moving_avg_bpm(self, window_size=10):
        """
        Calculates the moving average of the beats per minute (BPM).

        Parameters:
        -----------
        window_size : int, optional
            The size of the moving average window. Default is 10.

        Returns:
        --------
        moving_avg_bpm : pandas.Series
            A series of moving average BPM values.
        """
        bpm = pd.Series(self.calculate_bpm())
        moving_avg_bpm = bpm.rolling(window=window_size).mean()
        return moving_avg_bpm

    def analyze(self, threshold=1.5, window_size=10):
        """
        Analyzes the EKG data by calculating the moving average of the beats per minute (BPM).

        Parameters:
        -----------
        threshold : float, optional
            The threshold value for detecting heartbeats. Default is 1.5.
        window_size : int, optional
            The size of the moving average window. Default is 10.

        Returns:
        --------
        moving_avg_bpm : pandas.Series
            A series of moving average BPM values.
        """
        moving_avg_bpm = self.calculate_moving_avg_bpm(window_size=window_size)
        return moving_avg_bpm
    
    def plot_ekg_data(self):
        """
        Plots the EKG data.
        """
        fig = go.Figure()
        fig.add_scatter(x=self._obj['Timestamp'], y=self._obj['HeartSignal'], mode='lines', name='Raw')
        fig.add_scatter(x=self._obj['Timestamp'].iloc[self.peaks], y=self._obj['HeartSignal'].iloc[self.peaks], mode='markers', name='Peaks')
        fig.add_scatter(x=self._obj['Timestamp'].iloc[self.low_peaks], y=self._obj['HeartSignal'].iloc[self.low_peaks], mode='markers', name='Low Peaks')
        fig.add_scatter(x=self._obj['Timestamp'].iloc[self.beats], y=self._obj['HeartSignal'].iloc[self.beats], mode='markers', name='Beats', opacity=0.5)
        fig.show()

    def plot_moving_avg_bpm(self, window_size=10):
        """
        Plots the moving average of the BPM.

        Parameters:
        -----------
        window_size : int, optional
            The size of the moving average window. Default is 10.
        """
        moving_avg_bpm = self.calculate_moving_avg_bpm(window_size=window_size)
        fig = go.Figure()
        fig.add_scatter(x=self._obj['Timestamp'].iloc[self.beats], y=moving_avg_bpm, mode='lines', name='Moving Average BPM')
        fig.update_layout(title='Moving Average BPM', xaxis_title='Time (s)', yaxis_title='BPM')
        fig.show()

    @property
    def peaks(self):
        """
        Finds the peaks in the EKG data.

        Returns:
        --------
        peaks : numpy.ndarray
            An array of peak indices.
        """
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(self._obj["HeartSignal"], distance=40, height=(self.signal_center + 0.4), prominence=0.4)
        return peaks
    
    @property
    def low_peaks(self):
        """
        Finds the low peaks in the EKG data.

        Returns:
        --------
        low_peaks : numpy.ndarray
            An array of low peak indices.
        """
        from scipy.signal import find_peaks

        low_peaks, _ = find_peaks(-self._obj["HeartSignal"], distance=40, height=(- self.signal_center + 0.3), prominence=0.3)
        return low_peaks
    
    @property
    def signal_center(self):
        """
        Finds the center of the EKG signal.

        Returns:
        --------
        signal_center : float
            The center of the EKG signal.
        """
        return np.mean(self._obj["HeartSignal"])
    
    @property
    def beats(self, threshold=40):
        """
        Finds the beats in the EKG data.

        Parameters:
        -----------
        threshold : int, optional
            The threshold value for detecting beats. Default is 5.

        Returns:
        --------
        beats : numpy.ndarray
            An array of beat indices.
        """
        combined_peaks = np.sort(np.concatenate((self.peaks, self.low_peaks)))
        beats = []

        i = 0
        while i < len(combined_peaks):
            current_peak = combined_peaks[i]
            beats.append(current_peak)
            while i < len(combined_peaks) and combined_peaks[i] <= current_peak + threshold:
                i += 1
        return beats  


    @property
    def measurement_length(self):
        """
        Calculates the length of the EKG measurement.

        Returns:
        --------
        measurement_length : float
            The length of the EKG measurement in minutes.
        """
        return str(round(self._obj["Timestamp"].iloc[-1] / 60, 2) + " minutes")
    
    @property
    def rr_intervals(self):
        """
        Calculates R-R intervals (time between consecutive beats) in milliseconds.
        
        Returns:
        --------
        rr_intervals : numpy.ndarray
            Array of R-R intervals in milliseconds.
        """
        beat_times = self._obj["Timestamp"].iloc[self.beats]
        rr_intervals_sec = np.diff(beat_times)
        return rr_intervals_sec * 1000  # Convert to milliseconds
    
    @property
    def mean_heart_rate(self):
        """
        Calculates the mean heart rate in beats per minute.
        
        Returns:
        --------
        mean_hr : float
            Mean heart rate in BPM.
        """
        rr_intervals_ms = self.rr_intervals
        if len(rr_intervals_ms) == 0:
            return 0.0
        mean_rr_ms = np.mean(rr_intervals_ms)
        return 60000 / mean_rr_ms  # Convert from ms to BPM
    
    @property
    def rmssd(self):
        """
        Calculates RMSSD (Root Mean Square of Successive Differences) - a key HRV metric.
        
        Returns:
        --------
        rmssd : float
            RMSSD value in milliseconds.
        """
        rr_intervals_ms = self.rr_intervals
        if len(rr_intervals_ms) < 2:
            return 0.0
        successive_diffs = np.diff(rr_intervals_ms)
        return np.sqrt(np.mean(successive_diffs ** 2))
    
    @property
    def sdnn(self):
        """
        Calculates SDNN (Standard Deviation of NN intervals) - another HRV metric.
        
        Returns:
        --------
        sdnn : float
            SDNN value in milliseconds.
        """
        rr_intervals_ms = self.rr_intervals
        if len(rr_intervals_ms) == 0:
            return 0.0
        return np.std(rr_intervals_ms)
    
    def calculate_frequency_domain_hrv(self):
        """
        Calculates frequency domain HRV metrics including LF/HF ratio.
        
        Returns:
        --------
        dict : Dictionary containing LF, HF, and LF/HF ratio.
        """
        
        rr_intervals_ms = self.rr_intervals
        if len(rr_intervals_ms) < 10:
            return {"LF": 0.0, "HF": 0.0, "LF_HF_ratio": 0.0}
        
        # Resample RR intervals to get evenly spaced time series
        beat_times = self._obj["Timestamp"].iloc[self.beats[:-1]]  # Exclude last beat since we have n-1 intervals
        
        # Interpolation for evenly spaced samples
        fs = 4  # 4 Hz sampling rate
        time_new = np.arange(beat_times.iloc[0], beat_times.iloc[-1], 1/fs)
        rr_interp = np.interp(time_new, beat_times, rr_intervals_ms)
        
        # Calculate power spectral density
        freqs, psd = signal.welch(rr_interp, fs=fs, nperseg=len(rr_interp)//4)
        
        # Define frequency bands
        lf_band = (freqs >= 0.04) & (freqs <= 0.15)  # Low frequency
        hf_band = (freqs >= 0.15) & (freqs <= 0.4)   # High frequency
        
        lf_power = np.trapz(psd[lf_band], freqs[lf_band])
        hf_power = np.trapz(psd[hf_band], freqs[hf_band])
        
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0
        
        return {
            "LF": lf_power,
            "HF": hf_power,
            "LF_HF_ratio": lf_hf_ratio
        }
    
    @property
    def rhythm_type(self):
        """
        Analyzes the rhythm pattern of the heart.
        
        Returns:
        --------
        rhythm : str
            Classification of heart rhythm.
        """
        rr_intervals_ms = self.rr_intervals
        if len(rr_intervals_ms) < 5:
            return "Insufficient data"
        
        # Calculate coefficient of variation
        cv = np.std(rr_intervals_ms) / np.mean(rr_intervals_ms)
        mean_hr = self.mean_heart_rate
        
        # Classify rhythm based on HR and variability
        if mean_hr < 60:
            rhythm = "Bradycardia"
        elif mean_hr > 100:
            rhythm = "Tachycardia"
        elif cv < 0.05:
            rhythm = "Regular"
        elif cv < 0.15:
            rhythm = "Sinus rhythm"
        else:
            rhythm = "Irregular"
        
        return rhythm
    
    @property
    def arrhythmia_detected(self):
        """
        Detects potential arrhythmias based on RR interval analysis.
        
        Returns:
        --------
        arrhythmia : bool
            True if potential arrhythmia is detected.
        """
        rr_intervals_ms = self.rr_intervals
        if len(rr_intervals_ms) < 5:
            return False
        
        # Check for significant outliers in RR intervals
        q75, q25 = np.percentile(rr_intervals_ms, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        outliers = (rr_intervals_ms < lower_bound) | (rr_intervals_ms > upper_bound)
        outlier_percentage = np.sum(outliers) / len(rr_intervals_ms)
        
        # Check for very high variability
        cv = np.std(rr_intervals_ms) / np.mean(rr_intervals_ms)
        
        # Detect potential arrhythmia
        return outlier_percentage > 0.05 or cv > 0.2
    
    @property
    def stress_index(self):
        """
        Calculates a simple stress index based on HRV metrics.
        
        Returns:
        --------
        stress_index : float
            Stress index (higher values indicate more stress).
        """
        rmssd = self.rmssd
        mean_hr = self.mean_heart_rate
        
        # Simple stress index: higher HR and lower HRV indicate stress
        if rmssd == 0:
            return 100.0
        
        stress_index = (mean_hr / 70) * (50 / rmssd) * 100
        return min(stress_index, 100.0)  # Cap at 100
    
    def get_heart_analysis_summary(self):
        """
        Creates a comprehensive summary of heart analysis metrics.
        
        Returns:
        --------
        summary : str
            Formatted summary string with key heart metrics.
        """
        hr_mean = self.mean_heart_rate
        rmssd = self.rmssd
        sdnn = self.sdnn
        freq_metrics = self.calculate_frequency_domain_hrv()
        lfhf = freq_metrics["LF_HF_ratio"]
        rhythm_type = self.rhythm_type
        arrhythmia_flag = "Yes" if self.arrhythmia_detected else "No"
        stress_idx = self.stress_index
        measurement_duration = round(self._obj["Timestamp"].iloc[-1] / 60, 1)
        
        summary = (
            f"Heart Rate Analysis Summary:\n"
            f"Mean heart rate: {hr_mean:.1f} bpm, "
            f"HRV (RMSSD): {rmssd:.1f} ms, "
            f"SDNN: {sdnn:.1f} ms, "
            f"LF/HF ratio: {lfhf:.2f}, "
            f"Rhythm: {rhythm_type}, "
            f"Arrhythmia detected: {arrhythmia_flag}, "
            f"Stress index: {stress_idx:.1f}/100, "
            f"Duration: {measurement_duration} min"
        )
        
        return summary

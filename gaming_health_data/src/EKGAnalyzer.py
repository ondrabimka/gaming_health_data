# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
        
    @staticmethod
    def from_file(file_path, to_seconds=True, solve_time_reset=True, move_to_zero=True):

        """
        Reads EKG data from a CSV file.
        
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
        bpm = 60 / np.diff(self._obj["Timestamp"].iloc[self.low_peaks]) * 1e6
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
        fig.add_scatter(x=self._obj["Timestamp"].iloc[self.low_peaks] / 1e6, y=moving_avg_bpm, mode='lines', name='Moving Average BPM')
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
        peaks, _ = find_peaks(self._obj["HeartSignal"], distance=10, height=(self.signal_center + 0.1), prominence=0.3)
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

        low_peaks, _ = find_peaks(-self._obj["HeartSignal"], distance=10, height=(- self.signal_center + 0.1), prominence=0.3)
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

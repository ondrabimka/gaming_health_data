# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

@pd.api.extensions.register_dataframe_accessor("Keyboard")
class KeyboardAnalyzer:
    """
    A class for analyzing keyboard data.

    Parameters:
    -----------
    file_path : str
        The path to the CSV file containing keyboard data.

    Attributes:
    -----------
    keyboard_data : pandas.DataFrame
        The keyboard data loaded from the CSV file.

    Methods:
    --------
    analyze()
        Analyzes the keyboard data by calculating the typing speed.
    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """
        Validate the input object.

        Parameters:
        -----------
        obj : pandas.DataFrame
            The object to validate.

        Raises:
        -------
        TypeError
            If the input object is not a pandas DataFrame.
        """
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("KeyboardAnalyzer only works with pandas DataFrames.")

    @staticmethod
    def from_file(file_path, solve_time_reset=True, move_to_zero=True):
        """
        Reads keyboard data from a CSV file.

        Parameters:
        -----------
        file_path : str
            The path to the CSV file containing keyboard data.

        Returns:
        --------
        pandas.DataFrame
            The keyboard data.
        """
        keyboard_data = pd.read_csv(file_path)
        keyboard_data.rename(columns={"Time (ms)": "Timestamp"}, inplace=True)

        if solve_time_reset:
            keyboard_data['adjusted_timestamp'] = keyboard_data['Timestamp']
            offset = 0
            # Iterate over the rows
            for i in range(1, len(keyboard_data)):
                # If the current timestamp is less than the previous one, update 'offset'
                if keyboard_data.loc[i, 'Timestamp'] < keyboard_data.loc[i-1, 'Timestamp']:
                    offset = keyboard_data.loc[i-1, 'adjusted_timestamp']
                # Add 'offset' to 'adjusted_timestamp'
                keyboard_data.loc[i, 'adjusted_timestamp'] = keyboard_data.loc[i, 'Timestamp'] + offset
            keyboard_data.drop(columns=['Timestamp'], inplace=True)
            keyboard_data.rename(columns={'adjusted_timestamp': 'Timestamp'}, inplace=True)
        if move_to_zero:
            keyboard_data['Timestamp'] = keyboard_data['Timestamp'] - keyboard_data['Timestamp'].iloc[0]

        return keyboard_data

    def analyze(self):
        """
        Analyzes the keyboard data by calculating the typing speed.

        Returns:
        --------
        typing_speed : float
            The typing speed in words per minute.
        """
        pass

    def calculate_presses_per_second(self, window_size=5):
    
        """
        Calculate the number of keyboard per second using a moving window.

        Returns:
        --------
        pandas.Series
            The clicks per second calculated using the moving window.
        """

        click_times = self.press_data['Timestamp']
        click_diff = click_times.diff()
        click_diff = click_diff / 1000  # convert to seconds
        click_diff = 1 / click_diff # convert to clicks per second
        click_diff = click_diff.rolling(window=window_size).mean()
        return click_diff
    
    def plot_clicks_per_second(self, window_size=5):
        """
        Plot the number of clicks per second.
        """
        click_diff = self.calculate_presses_per_second(window_size)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.press_data['Timestamp'], y=click_diff, mode='markers', name='Clicks per second'))
        fig.update_layout(title='Clicks per second', xaxis_title='Time', yaxis_title='Clicks per second')
        fig.show()

    def plot_key_presses_histogram(self):
        """
        Plot a histogram of the key presses.
        """
        pressed_keys = self.press_data[['Details']]
        pressed_keys['Details'] = pressed_keys['Details'].str.replace('key: ', '')
        fig = px.histogram(pressed_keys, x='Details')
        fig.show()


    def calculate_total_press_time(self):
        """
        Calculate the total press time for each key.

        Returns:
        --------
        pandas.DataFrame
            The total press time for each key.
        """
        self._obj['key'] = self._obj['Details'].str.split(': ').str[1]

        total_press_time = {}

        for key in self._obj['key'].unique():
            key_presses = self._obj[(self._obj['key'] == key) & (self._obj['Action'] == 'Press')]
            key_releases = self._obj[(self._obj['key'] == key) & (self._obj['Action'] == 'Release')]
            durations = key_releases['Timestamp'].values - key_presses['Timestamp'].values
            total_press_time[key] = durations.sum()

        result_df = pd.DataFrame(list(total_press_time.items()), columns=['key', 'TotalPressTime'])
        return result_df.sort_values('TotalPressTime', ascending=False)
    
    @property
    def press_data(self):
        """
        Extract the press data from the keyboard data.

        Returns:
        --------
        pandas.DataFrame
            The press data.
        """
        return self._obj[self._obj['Action'] == 'Press']
    
    @property
    def release_data(self):
        """
        Extract the release data from the keyboard data.

        Returns:
        --------
        pandas.DataFrame
            The release data.
        """
        return self._obj[self._obj['Action'] == 'Release']

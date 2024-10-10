# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

@pd.api.extensions.register_dataframe_accessor("Mouse")
class MouseAnalyzer:
    """
    A class for analyzing mouse data.

    Parameters:
    -----------
    file_path : str
        The path to the CSV file containing mouse data.

    Attributes:
    -----------
    mouse_data : pandas.DataFrame
        The mouse data loaded from the CSV file.

    Methods:
    --------
    calculate_speed()
        Calculates the speed of the mouse pointer.

    calculate_acceleration()
        Calculates the acceleration of the mouse pointer.

    analyze()
        Analyzes the mouse data by calculating the speed and acceleration.
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
            raise TypeError("MouseAnalyzer only works with pandas DataFrames.")

    @staticmethod
    def from_file(file_path, move_to_zero=True):
        """
        Reads mouse data from a CSV file.

        Parameters:
        -----------
        file_path : str
            The path to the CSV file containing mouse data.

        move_to_zero : bool, optional
            Whether to move the timestamps to zero. Default is True.

        Returns:
        --------
        pandas.DataFrame
            The mouse data loaded from the CSV file.
        """
        mouse_data = pd.read_csv(file_path)
        mouse_data.rename(columns={'Time (ms)': 'Timestamp'}, inplace=True)
        if move_to_zero:
            mouse_data['Timestamp'] = mouse_data['Timestamp'] - mouse_data['Timestamp'].min()
        return mouse_data

    def calculate_speed(self):
        """
        Calculate the speed of the mouse pointer.

        Returns:
        --------
        pandas.Series
            The speed of the mouse pointer.
        """
        dx = self._obj['X'].diff()
        dy = self._obj['Y'].diff()
        dt = self._obj['Timestamp'].diff


    @property
    def click_data(self):
        """
        Extract the click data from the mouse data.

        Returns:
        --------
        pandas.DataFrame
            The click data.
        """
        return self._obj[self._obj['Action'] == 'Click']
    
    @property
    def move_data(self):
        """
        Extract the move data from the mouse data.

        Returns:
        --------
        pandas.DataFrame
            The move data.
        """
        move_data = self._obj[self._obj['Action'] == 'Move']
        # split details column into X and Y columns
        move_data[['X', 'Y']] = move_data['Details'].str.split(',', expand=True)
        move_data['X'] = move_data['X'].str.extract(r'(\d+)').astype(int)
        move_data['Y'] = move_data['Y'].str.extract(r'(\d+)').astype(int)
        return move_data
    
    def calculate_clicks_per_second(self, window_size=5):
        """
        Calculate the number of clicks per second using a moving window.

        Returns:
        --------
        pandas.Series
            The clicks per second calculated using the moving window.
        """
        click_times = self.click_data['Timestamp']
        click_diff = click_times.diff()
        click_diff = click_diff / 1000  # convert to seconds
        click_diff = 1 / click_diff # convert to clicks per second
        click_diff = click_diff.rolling(window=window_size).mean()
        return click_diff
    
    def calculate_clicks_by_second(self, include_empty_seconds=True):
        """
        Calculate the number of clicks that occurred in each second of the recording.

        Parameters
        ----------
        include_empty_seconds: bool
            Whether to include seconds where no clicks occurred in the output.

        Returns
        -------
        pandas Series
            A Series where the index is the second number and the values are the number of clicks that occurred in that second.
        """
        click_data_df = self.click_data.copy()
        click_data_df['second'] = click_data_df.Timestamp // 1000
        clicks_by_second = click_data_df.groupby('second').size() 

        if include_empty_seconds:
            all_seconds = np.arange(click_data_df['second'].min(), click_data_df['second'].max() + 1)
            clicks_by_second = clicks_by_second.reindex(all_seconds, fill_value=0)
            
        return clicks_by_second
    
    def plot_clicks_per_second(self, window_size=5):
        """
        Plot the number of clicks per second using a moving window.

        Returns:
        --------
        plotly.graph_objects.Figure
            The plot showing the clicks per second.
        """
        click_diff = self.calculate_clicks_per_second(window_size)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.click_data['Timestamp'] / 1000, y=click_diff, mode='markers'))
        fig.update_layout(title='Clicks per Second', xaxis_title='Time (ms)', yaxis_title='Clicks per Second')
        return fig
    
    def plot_clicks_by_second(self, include_empty_seconds=True):
        """
        Plot the number of clicks that occurred in each second of the recording.

        Returns:
        --------
        plotly.graph_objects.Figure
            The plot showing the clicks by second.
        """
        clicks_by_second = self.calculate_clicks_by_second(include_empty_seconds)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=clicks_by_second.index, y=clicks_by_second))
        fig.update_layout(title='Clicks by Second', xaxis_title='Second', yaxis_title='Number of Clicks')
        return fig
    
    def plot_mouse_path(self):
        """
        Plot the path of the mouse pointer as 3d plot in time.

        Returns:
        --------
        plotly.graph_objects.Figure
            The plot showing the path of the mouse pointer.
        """
        fig = px.scatter(self.move_data, x='X', y='Y', color='Timestamp')
        return fig
        

# %%
mouse_log = pd.read_csv("gaming_health_data/recorded_data/mouse_log_28_04_2024.csv")
mouse_log.rename(columns={'Time (ms)': 'Timestamp'}, inplace=True)
mouse_log['Timestamp'] = mouse_log['Timestamp'] - mouse_log['Timestamp'].min() # move times to zero

# %%
mouse_log.Mouse.plot_clicks_per_second(window_size=2)
# %%
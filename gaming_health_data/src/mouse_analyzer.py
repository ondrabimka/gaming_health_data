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
        pass

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
    
    def calculate_clicks_per_second(self, window_size=100):
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
    
    def plot_clicks_per_second(self, **kwargs):
        """
        Plot the number of clicks per second using a moving window.

        Returns:
        --------
        plotly.graph_objects.Figure
            The plot showing the clicks per second.
        """
        click_diff = self.calculate_clicks_per_second(**kwargs)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.click_data['Timestamp'], y=click_diff, mode='lines'))
        fig.update_layout(title='Clicks per Second', xaxis_title='Time (ms)', yaxis_title='Clicks per Second')
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
mouse_log = pd.read_csv("C:/Users/Admin/Desktop/embedded_code/gaming_health_data/gaming_health_data/recorded_data/mouse_log_28_04_2024.csv")
mouse_log.rename(columns={'Time (ms)': 'Timestamp'}, inplace=True)
mouse_log['Timestamp'] = mouse_log['Timestamp'] - mouse_log['Timestamp'].min() # move times to zero

# %%
mouse_log.Mouse.plot_clicks_per_second(window_size=3)
mouse_log.Mouse.plot_mouse_path()
# %%
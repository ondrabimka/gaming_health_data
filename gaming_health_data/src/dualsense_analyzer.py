# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime


@pd.api.extensions.register_dataframe_accessor("dualsense")
class DualSenseAnalyzer:
    """
    A class for analyzing DualSense controller data.

    Parameters:
    -----------
    file_path : str
        The path to the CSV file containing DualSense controller data.

    Attributes:
    -----------
    dualsense_data : pandas.DataFrame
        The DualSense controller data loaded from the CSV file.

    Methods:
    --------
    analyze()
        Analyzes the DualSense controller data.
    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self.record_start_datetime = pd.to_datetime(self._obj["Timestamp"].min(), unit='s')

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
            raise TypeError("DualSenseAnalyzer only works with pandas DataFrames.")

    @classmethod
    def read_file(cls, file_path, move_to_zero=True, map_buttons=True):
        """
        Reads DualSense controller data from a CSV file.

        Parameters:
        -----------
        file_path : str
            The path to the CSV file containing DualSense controller data.

        Returns:
        --------
        pandas.DataFrame
            The DualSense controller data.
        """
        dualsense_data = pd.read_csv(file_path)

        cls.measure_start_date = dualsense_data["Timestamp"].iloc[0]
        cls.measure_end_date = dualsense_data["Timestamp"].iloc[-1]

        # map buttons to names
        if map_buttons:
            dualsense_data["Button/Axis"] = dualsense_data["Button/Axis"].map({
            "Axis 0": "Left Stick Horizontal", # left is negative, right is positive
            "Axis 1": "Left Stick Vertical", # down is positive, up is negative
            "Axis 2": "Right Stick Horizontal", # left is negative, right is positive
            "Axis 3": "Right Stick Vertical", # down is positive, up is negative
            "Axis 4": "L2",
            "Axis 5": "R2",
            "Button 0": "Cross",
            "Button 1": "Circle",
            "Button 2": "Square",
            "Button 3": "Triangle",
            "Button 4": "Record",
            "Button 5": "PS Button",
            "Button 6": "Options",  
            "Button 7": "L3",
            "Button 8": "R3",
            "Button 9": "L1",
            "Button 10": "R1",
            "Button 11": "Up",
            "Button 12": "Down",
            "Button 13": "Left",
            "Button 14": "Right",
            "Button 15": "Touchpad",
            "Button 16": "Microphone",
        })

        if move_to_zero:
            dualsense_data["Timestamp"] = dualsense_data["Timestamp"] - dualsense_data["Timestamp"].min()

        return dualsense_data
    

    def plot_right_stick_movement(self):
        """
        Plots the movement of the right stick on the DualSense controller.
        """

        # Merge horizontal and vertical data on timestamp.
        right_stick_data = pd.merge(
            self.right_stick_horizontal[['Timestamp', 'Value']].rename(columns={'Value': 'x'}),
            self.right_stick_vertical[['Timestamp', 'Value']].rename(columns={'Value': 'y'}),
            on='Timestamp',
            how='outer'
        )

        # fillna with 0
        right_stick_data.fillna(0, inplace=True)

        # Create scatter plot
        fig = px.scatter(right_stick_data, x='x', y='y',
                         title='Right Stick Movement',
                         labels={'x': 'Horizontal Position', 'y': 'Vertical Position'})

        # Add circular boundary to represent stick limits
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines', 
                                line=dict(color='gray', dash='dash'),
                                name='Stick Boundary'))

        # Set equal axes and range
        fig.update_layout(
            xaxis_range=[-1.1, 1.1],
            yaxis_range=[-1.1, 1.1],
            xaxis=dict(scaleanchor='y', scaleratio=1),
        )

        fig.show()


    def plot_button_presses_histogram(self):
        """
        Plots a histogram of button presses on the DualSense controller.
        """

        # remove axis 1-3
        axis_to_remove = ["Left Stick Vertical", "Left Stick Horizontal", "Right Stick Vertical", "Right Stick Horizontal"]
        button_presses = self._obj[~self._obj["Button/Axis"].isin(axis_to_remove)]
        
        # Create histogram
        fig = px.histogram(button_presses, x='Button/Axis', title='Button Presses',
                           labels={'Button/Axis': 'Button', 'count': 'Frequency'})
        
        fig.show()


    def transform_to_press_release(self, button_name = None, press_threshold = 0.008):
        """
        Transform button data to press/release events by removing intermediate readings.
        
        Parameters:
        -----------
        button_name : str
            The name of the button to analyze
        press_threshold : float
            Time threshold in seconds to consider consecutive readings as part of the same press
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing only press and release events with columns:
            ['Timestamp', 'Button/Axis', 'Value', 'Event']
        """

        if button_name:
            button_data = self.get_button_df(button_name).copy()
        else:
            button_data = self._obj.copy()
        
        if len(button_data) == 0:
            return pd.DataFrame()
        
        button_data = button_data.sort_values('Timestamp')
        button_data['time_diff'] = button_data['Timestamp'].diff()
        button_data['Event'] = 'intermediate'
        
        button_data.loc[
            (button_data['time_diff'] > press_threshold) | 
            (button_data['time_diff'].isna()), 
            'Event'
        ] = 'press'
        
        button_data.loc[
            (button_data['time_diff'].shift(-1) > press_threshold) | 
            (button_data.index == button_data.index[-1]), 
            'Event'
        ] = 'release'
        
        result = button_data[button_data['Event'].isin(['press', 'release'])].copy()
        result = result.drop('time_diff', axis=1)
        
        return result
    
    def calculate_presses_per_second(self, window_size=5):
    
        """
        Calculate the number of keyboard per second using a moving window.

        Parameters:
        -----------
        window_size : int, optional
            The size of the moving window. Default is 5.

        Returns:
        --------
        pandas.Series
            The clicks per second calculated using the moving window.
        """

        click_times = self.press_data['Timestamp']
        click_diff = click_times.diff()
        click_diff = click_diff # / 1000
        click_diff = 1 / click_diff # convert to clicks per second
        click_diff = click_diff.rolling(window=window_size).mean()
        return click_diff
    
    def get_stats_for_button(self, button_name, press_threshold=0.008) -> dict:
        """
        Get statistics for a specific button.

        Parameters:
        -----------
        button_name : str
            The name of the button to analyze

        press_threshold : float
            Time threshold in seconds to consider consecutive readings as part of the same press
        
        Returns:
        --------
        dict
            A dictionary containing the following statistics:
            - Press Count
            - Total Press Time
            - Average Time Between Presses
            - Average Press Duration
        """
        button_df = self.transform_to_press_release(button_name, press_threshold)

        if len(button_df) == 0:
            return {
                "Press Count": 0,
                "Total Press Time": 0,
                "Average Time Between Presses": 0,
                "Average Press Duration": 0
            }

        press_count = button_df[button_df['Event'] == 'press'].shape[0]
        button_df['time_diff'] = button_df['Timestamp'].diff()
        avg_time_between_presses = button_df['time_diff'].mean()

        # Get press times and release times 
        press_times = button_df[button_df['Event'] == 'press']['Timestamp'].values
        release_times = button_df[button_df['Event'] == 'release']['Timestamp'].values

        print(f"Calculating stats for {button_name}:")
        print(len(press_times), len(release_times))
        # Calculate durations as release time minus press time
        press_durations = release_times - press_times
        avg_press_duration = press_durations.mean()
        total_press_time = press_durations.sum()

        return {
            "Press Count": press_count,
            "Total Press Time": total_press_time,
            "Average Time Between Presses": avg_time_between_presses,
            "Average Press Duration": avg_press_duration
        }

    
    def get_all_button_stats(self, press_threshold=0.008, ignore_sticks=True) -> dict:
        """
        Get statistics for all buttons.

        Parameters:
        -----------
        press_threshold : float
            Time threshold in seconds to consider consecutive readings as part of the same press
        
        Returns:
        --------
        dict
            A dictionary containing statistics for each button.
        """
        button_names = self._obj["Button/Axis"].unique()

        if ignore_sticks:
            button_names = [name for name in button_names if name not in ["Left Stick Horizontal", "Left Stick Vertical", "Right Stick Horizontal", "Right Stick Vertical"]] # ignore axis 0-3

        stats = {}
        
        for button_name in button_names:
            print(f"Calculating stats for {button_name}")
            stats[button_name] = self.get_stats_for_button(button_name, press_threshold)
        
        return stats
    

    def calculate_presses_by_second(self, include_empty_seconds=True):
        """
        Calculate the number of key presses that occurred in each second of the recording.

        Parameters
        ----------
        press_data_df: pandas DataFrame
            The DataFrame containing the key press data.

        include_empty_seconds: bool
            Whether to include seconds where no key presses occurred in the output.

        Returns
        -------
        pandas Series
            A Series where the index is the second number and the values are the number of key presses that occurred in that second.
        """
        press_data_df = self.press_data.copy()
        press_data_df['second'] = press_data_df.Timestamp // 1
        presses_by_second = press_data_df.groupby('second').size()

        if include_empty_seconds:
            full_index = pd.RangeIndex(start=0, stop=(press_data_df.Timestamp.max() // 1 + 1))  # // 1000) + 1)
            presses_by_second = presses_by_second.reindex(full_index, fill_value=0)

        return presses_by_second
    
    def plot_presses_per_second(self, window_size:int=5):
        """
        Plot the number of clicks per second using a moving window.
        """
        click_diff = self.calculate_presses_per_second(window_size)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.press_data['Timestamp'] / 1, y=click_diff, mode='markers', name='Clicks per second'))
        fig.update_layout(title='Clicks per second', xaxis_title='Time', yaxis_title='Clicks per second')
        fig.show()


    def plot_presses_by_second(self, include_empty_seconds=True):
        """
        Plot the number of key presses that occurred in each second of the recording.
        """
        presses_by_second = self.calculate_presses_by_second(include_empty_seconds)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=presses_by_second.index, y=presses_by_second, mode='markers', name='Key presses per second'))
        fig.update_layout(title='Key presses per second', xaxis_title='Time', yaxis_title='Key presses per second')
        fig.show()

    def plot_histograms(self, to_plot=['Press Count', 'Total Press Time', 'Average Time Between Presses', 'Average Press Duration']):
        """
        Plot bar charts for button statistics.
        """
        stats = self.get_all_button_stats()
        
        for stat in to_plot:
            # Create arrays for x and y values
            buttons = []
            values = []
            for button_name, button_stats in stats.items():
                buttons.append(button_name)
                values.append(button_stats[stat])
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(name=stat, x=buttons, y=values)
            ])
            
            # Update layout
            fig.update_layout(
                title=f'{stat} by Button',
                xaxis_title='Button',
                yaxis_title=stat,
                xaxis_tickangle=45  # Rotate x labels for better readability
            )
            fig.show()

    @property
    def press_data(self):
        """
        Returns the press data DataFrame.
        """
        press_release = self.transform_to_press_release()
        # keep only buttons
        press_release = press_release[press_release["Button/Axis"].isin(["Cross", "Circle", "Square", "Triangle", "L1", "R1", "L2", "R2"])]
        # only press and release events
        press_release = press_release[press_release["Event"].isin(["press"])]
        return press_release
    
    @property
    def left_stick_horizontal(self):
        return self._obj[self._obj["Button/Axis"] == "Left Stick Horizontal"]
    
    @property
    def left_stick_vertical(self):
        return self._obj[self._obj["Button/Axis"] == "Left Stick Vertical"]
    
    @property
    def right_stick_horizontal(self):
        return self._obj[self._obj["Button/Axis"] == "Right Stick Horizontal"]
    
    @property
    def right_stick_vertical(self):
        return self._obj[self._obj["Button/Axis"] == "Right Stick Vertical"]
    
    def get_button_df(self, button_name):
        return self._obj[self._obj["Button/Axis"] == button_name]

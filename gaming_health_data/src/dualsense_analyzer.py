import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Data example:
# Timestamp,Button/Axis,Value
# 1738949557.3960457,Axis 4,-1.0
# 1738949557.3960457,Axis 5,-1.0
# 1738949557.4070418,Axis 4,-1.0
# 1738949557.4070418,Axis 5,-1.0
# 1738949557.4180398,Axis 4,-1.0
# 1738949557.4180398,Axis 5,-1.0
# 1738949557.4280405,Axis 4,-1.0
# 1738949557.4280405,Axis 5,-1.0
# 1738949557.4380531,Axis 4,-1.0
# 1738949557.4380531,Axis 5,-1.0
# 1738949557.4490898,Axis 4,-1.0
# 1738951225.3713202,Button 9,1
# 1738951225.3713202,Axis 0,-0.694
# 1738951225.3713202,Axis 1,-0.867
# 1738951225.3713202,Axis 2,0.176
# 1738951225.3713202,Axis 4,-1.0
# 1738951225.3713202,Axis 5,-1.0
# 1738951225.382859,Button 9,1

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

        self.name_to_button = {
            "Axis 4": "R3",
            "Axis 5": "L3",
            "Button 0": "Cross",
            "Button 1": "Circle",
            "Button 2": "Square",
            "Button 3": "Triangle",
            "Button 4": "L1",
            "Button 5": "R1",
            "Button 6": "L2",
            "Button 7": "R2",
            "Button 8": "Share",
            "Button 9": "L1",

            "Button 12": "Down",
            "Button 14": "Right",


        }
    # vide is 7 sec +

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
    def from_file(cls, file_path, solve_time_reset=True, move_to_zero=True):
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
        cls.record_start_datetime = pd.to_datetime(dualsense_data["Timestamp"].min(), unit='s')

        if solve_time_reset:
            dualsense_data["Timestamp"] = pd.to_datetime(dualsense_data["Timestamp"])
            dualsense_data["Timestamp"] = (dualsense_data["Timestamp"] - dualsense_data["Timestamp"].min()).dt.total_seconds()

        if move_to_zero:
            dualsense_data["Timestamp"] = dualsense_data["Timestamp"] - dualsense_data["Timestamp"].min()

        return dualsense_data
    

    @property
    def left_stick_df(self):
        return self._obj[self._obj["Button/Axis"] == "Left Stick"]
    
    @property
    def right_stick_df(self):
        return self._obj[self._obj["Button/Axis"] == "Right Stick"]
    
    @property
    def button_df(self, button_name):
        button_number = [key for key, value in self.name_to_button.items() if value == button_name][0]
        return self._obj[self._obj["Button/Axis"] == button_number]
    

# %% covnert 1738949557.3960457 to time
import pandas as pd
# read /Users/obimka/Desktop/Zabafa/gaming_health_data/gaming_health_data/recorded_data/PS/controller_inputs_07_02_2025.csv and convert timestamp to time
df = pd.read_csv('/Users/obimka/Desktop/Zabafa/gaming_health_data/gaming_health_data/recorded_data/PS/controller_inputs_07_02_2025.csv')
# move to zero
df["Timestamp"] = df["Timestamp"] - df["Timestamp"].min()
# convert to seconds
df

# %%
df
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
    def from_file(file_path):
        pass

    def analyze(self):
        """
        Analyzes the keyboard data by calculating the typing speed.

        Returns:
        --------
        typing_speed : float
            The typing speed in words per minute.
        """
        pass


# %%
from gaming_health_data.src.utils import DATA_DIR

keyboard_data = pd.read_csv(DATA_DIR / "keyboard_log_28_04_2024.csv")
keyboard_data.rename(columns={"Time (ms)": "Timestamp"}, inplace=True)
# %%
# move to zero
keyboard_data["Timestamp"] = keyboard_data["Timestamp"] - keyboard_data["Timestamp"].min()

# %%
for i in range(len(keyboard_data) - 1000):
    part = keyboard_data[keyboard_data["Timestamp"].between(keyboard_data["Timestamp"].iloc[i], keyboard_data["Timestamp"].iloc[i + 1000])]
    print(len(part))

# %%
def calculate_presses_per_second(keyboard_data):
    """
    Calculate the number of key presses per second.

    Parameters:
    -----------
    keyboard_data : pandas.DataFrame
        The keyboard data.

    Returns:
    --------
    pandas.Series
        The number of key presses per second.
    """
    for i in range(len(keyboard_data) - 1000):
        keyboard_data["Timestamp"].iloc[i] = keyboard_data["Timestamp"].iloc[i + 1000] - keyboard_data["Timestamp"].iloc[i]
        
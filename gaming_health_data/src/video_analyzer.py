import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

@pd.api.extensions.register_dataframe_accessor("Video")
class VideoAnalyzer:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._validate()

    def _validate(self):
        if "action" not in self._obj.columns:
            raise AttributeError("action column is missing")
        if "timestamp" not in self._obj.columns:
            raise AttributeError("timestamp column is missing")

    @staticmethod    
    def from_file(file_path):
        return pd.read_csv(file_path)
    
    def plot_timeline(self):

        """Plots a timeline of the actions in the video."""

        df = self._obj.copy()
        # Convert timestamp to datetime only if it is in the format HH:MM:SS
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S')

        # Split the actions into start and end
        df['action_type'] = df['action'].apply(lambda x: x.split('_')[-1])
        df['action'] = df['action'].apply(lambda x: '_'.join(x.split('_')[:-1]))

        # Create a new DataFrame to store start and end times
        events = []

        for action in df['action'].unique():
            action_df = df[df['action'] == action]
            starts = action_df[action_df['action_type'] == 'start'].reset_index(drop=True)
            ends = action_df[action_df['action_type'] == 'end'].reset_index(drop=True)
            
            for i in range(len(starts)):
                start_time = starts.at[i, 'timestamp']
                end_time = ends.at[i, 'timestamp'] if i < len(ends) else start_time
                events.append({
                    'action': action,
                    'start': start_time,
                    'end': end_time
                })

        events_df = pd.DataFrame(events)

        fig = px.timeline(events_df, x_start="start", x_end="end", y="action", color="action")
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_layout(title="Action Timeline", xaxis_title="Time", yaxis_title="Action")
        fig.show()

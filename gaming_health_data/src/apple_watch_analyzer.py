# %%
import pandas as pd 
import pandas as pd
from plotly import graph_objects as go

# TODO:
# 1. Create a Pandas extension accessor for Apple Watch health data analysis.
# 2. Implement methods to filter data by type and date.
# 3. Get number of sessions and their types.
# 4. Get stats for every session type.
# 5. Plot time series for a given data type.

@pd.api.extensions.register_dataframe_accessor("applewatch")
class AppleWatchAnalyzer:
    """A Pandas extension accessor for analyzing Apple Watch health data."""
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        # Parse dates for easier filtering
        for col in ['startDate', 'endDate', 'creationDate']:
            if col in self._obj.columns:
                self._obj[col] = pd.to_datetime(self._obj[col], errors='coerce')
        # Try to convert value to numeric if possible
        if 'value' in self._obj.columns:
            self._obj['value'] = pd.to_numeric(self._obj['value'], errors='coerce')

    def filter_by_type(self, data_type):
        """Return a DataFrame filtered by the 'type' column."""
        assert isinstance(data_type, str), "data_type must be a string"
        if 'type' not in self._obj.columns:
            raise ValueError("The DataFrame does not contain a 'type' column.")
        if data_type not in self._obj['type'].unique():
            raise ValueError(f"Data type '{data_type}' not found in the dataset.")
        return self._obj[self._obj['type'] == data_type].copy()

    def filter_by_date(self, start=None, end=None):
        """Return a DataFrame filtered by startDate between start and end."""
        df = self._obj
        if start:
            df = df[df['startDate'] >= pd.to_datetime(start)]
        if end:
            df = df[df['startDate'] <= pd.to_datetime(end)]
        return df.copy()

    def plot_time_series(self, data_type, start=None, end=None, agg='mean'):
        """Plot a time series for a given data_type and optional date range."""
        df = self.filter_by_type(data_type)
        if start or end:
            df = df[
                (df['startDate'] >= pd.to_datetime(start) if start else True) &
                (df['startDate'] <= pd.to_datetime(end) if end else True)
            ]
        if df.empty:
            print("No data for this selection.")
            return
        df = df.copy()
        df['date_only'] = df['startDate'].dt.date
        grouped = df.groupby('date_only')['value']
        if agg == 'mean':
            series = grouped.mean()
        elif agg == 'sum':
            series = grouped.sum()
        else:
            series = grouped.mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines+markers',
            name=data_type
        ))
        fig.update_layout(
            title=f"{data_type} Time Series",
            xaxis_title='Date',
            yaxis_title=data_type,
            template='plotly_white',
            xaxis=dict(tickformat='%Y-%m-%d'),
            yaxis=dict(title=data_type)
        )
        return fig

    @property
    def types(self):
        """Return all unique types in the data."""
        return self._obj['type'].dropna().unique()

    @property
    def activity_types(self):
        """Return all unique activity types."""
        if 'activityType' in self._obj.columns:
            return self._obj['activityType'].dropna().unique()
        return []

    @property
    def source_names(self):
        """Return all unique sourceName values."""
        return self._obj['sourceName'].dropna().unique()
    
    @classmethod
    def read_file(cls, file_path, **kwargs):
        """Read a CSV file and return an instance of AppleWatchAnalyzer."""
        df = pd.read_csv(file_path, **kwargs)
        return df
    
    def get_sessions(self):
        """Get all unique sessions based on startDate and endDate."""
        if 'startDate' not in self._obj.columns or 'endDate' not in self._obj.columns:
            raise ValueError("The DataFrame must contain 'startDate' and 'endDate' columns.")
        sessions = self._obj[['startDate', 'endDate', 'type']].drop_duplicates().copy()
        sessions['startDate'] = pd.to_datetime(sessions['startDate'], errors='coerce')
        sessions['endDate'] = pd.to_datetime(sessions['endDate'], errors='coerce')
        return sessions.dropna().reset_index(drop=True)
    
    def get_session_from_date(self, date):
        """Get a session that contains the specified date."""
        if 'startDate' not in self._obj.columns or 'endDate' not in self._obj.columns:
            raise ValueError("The DataFrame must contain 'startDate' and 'endDate' columns.")
        date = pd.to_datetime(date)
        sessions = self.get_sessions()
        session = sessions[
            (sessions['startDate'] <= date) & (sessions['endDate'] >= date)
        ]
        if session.empty:
            raise ValueError(f"No session found for date: {date}")
        return session.iloc[0]

    def get_hearth_rate_stats_from_session(self, session):
        """Get heart rate stats from a specific session."""
        if 'HeartRate' not in self._obj['type'].unique():
            raise ValueError("No heart rate data available.")
        hb_data = self._obj[
            (self._obj['startDate'] >= session['startDate']) &
            (self._obj['endDate'] <= session['endDate']) &
            (self._obj['type'] == 'HeartRate')
        ][['startDate', 'value']].copy()
        hb_data['startDate'] = pd.to_datetime(hb_data['startDate'], errors='coerce')
        hb_data['value'] = pd.to_numeric(hb_data['value'], errors='coerce')
        hb_data = hb_data.dropna().reset_index(drop=True)
        return hb_data
    
    def plot_heart_rate_session(self, session) -> go.Figure:
        """Plot heart rate data for a specific session."""
        hb_data = self.get_hearth_rate_stats_from_session(session)
        if hb_data.empty:
            print("No heart rate data for this session.")
            return
        hb_data = hb_data.sort_values(by='startDate').reset_index(drop=True)
        hb_data['startDate'] = hb_data['startDate'] - (hb_data['startDate'].min() - hb_data['startDate'].iloc[0])
        hb_data['startDate'] = hb_data['startDate'].dt.time

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hb_data['startDate'],
            y=hb_data['value'],
            mode='lines+markers',
            name='Heart Rate'
        ))
        fig.update_layout(
            title='Heart Rate Data for Session',
            xaxis_title='Time',
            yaxis_title='Heart Rate (bpm)',
            template='plotly_white',
            yaxis=dict(title='Heart Rate (bpm)', range=[hb_data['value'].min() - 10, hb_data['value'].max() + 10])
        )
        return fig

    def plot_hearth_rate_for_session_date(self, date):
        """Plot heart rate data for a session on a specific date."""
        session = self.get_session_from_date(date)
        self.plot_heart_rate_session(session)

# %% usage example
apple_watch_data = pd.DataFrame.applewatch.read_file('gaming_health_data/recorded_data/APPLE_WATCH/apple_health_export_2025-05-28.csv')
apple_watch_data.applewatch.plot_hearth_rate_for_session_date('2025-05-08')
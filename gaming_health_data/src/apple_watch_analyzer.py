# %%
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
        # Parse dates for easier filtering and normalize timezones
        for col in ['startDate', 'endDate', 'creationDate']:
            if col in self._obj.columns:
                self._obj[col] = pd.to_datetime(self._obj[col], errors='coerce')
                # Convert timezone-aware to timezone-naive (UTC)
                if hasattr(self._obj[col].dtype, 'tz') and self._obj[col].dtype.tz is not None:
                    self._obj[col] = self._obj[col].dt.tz_convert('UTC').dt.tz_localize(None)
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
        df = self._obj.copy()
        if start:
            start_dt = pd.to_datetime(start)
            # Normalize timezone if needed
            if hasattr(start_dt, 'tz') and start_dt.tz is not None:
                start_dt = start_dt.tz_convert('UTC').tz_localize(None)
            df = df[df['startDate'] >= start_dt]
        if end:
            end_dt = pd.to_datetime(end)
            # Normalize timezone if needed  
            if hasattr(end_dt, 'tz') and end_dt.tz is not None:
                end_dt = end_dt.tz_convert('UTC').tz_localize(None)
            df = df[df['startDate'] <= end_dt]
        return df

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
    
    def get_sessions(self):
        """Get all unique sessions based on startDate and endDate."""
        if 'startDate' not in self._obj.columns or 'endDate' not in self._obj.columns:
            raise ValueError("The DataFrame must contain 'startDate' and 'endDate' columns.")
        sessions = self._obj[['startDate', 'endDate', 'type']].drop_duplicates().copy()
        sessions['startDate'] = pd.to_datetime(sessions['startDate'], errors='coerce')
        sessions['endDate'] = pd.to_datetime(sessions['endDate'], errors='coerce')
        return sessions.dropna().reset_index(drop=True)

    def get_gaming_sessions(self):
        """Get all gaming sessions."""
        if 'workoutActivityType' not in self._obj.columns:
            raise ValueError("The DataFrame must contain a 'workoutActivityType' column.")
        gaming_sessions = self._obj[self._obj['workoutActivityType'] == "HKWorkoutActivityTypeFitnessGaming"].copy()
        return gaming_sessions.reset_index(drop=True)
    
    def gaming_data_for_session(self, date: str):
        """Get heart rate data for a gaming session on the specified date."""
        session_of_interest = self._obj[self._obj['startDate'].dt.date == pd.to_datetime(date).date()]
        start_date, end_date = session_of_interest['startDate'].min(), session_of_interest['endDate'].max()

        if pd.isna(start_date) or pd.isna(end_date):
            print(f"No gaming session found on {date}")
            return pd.DataFrame()
        
        hr_data = self._obj[
            (self._obj['startDate'] >= start_date) &
            (self._obj['endDate'] <= end_date) &
            (self._obj['type'] == 'HeartRate')
        ].copy()
        return hr_data.reset_index(drop=True)
    
    def plot_heart_rate_for_session_date(self, date: str):
        """Plot heart rate data for a gaming session on the specified date."""
        hr_data = self.gaming_data_for_session(date)
        if hr_data.empty:
            print(f"No heart rate data found for gaming session on {date}")
            return
        hr_data = hr_data.sort_values(by='startDate').reset_index(drop=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hr_data['startDate'],
            y=hr_data['value'],
            mode='lines+markers',
            name='Heart Rate'
        ))
        fig.update_layout(
            title=f'Heart Rate Data for Gaming Session on {date}',
            xaxis_title='Time',
            yaxis_title='Heart Rate (bpm)',
            template='plotly_white',
            yaxis=dict(title='Heart Rate (bpm)', range=[hr_data['value'].min() - 10, hr_data['value'].max() + 10])
        )
        return fig
        
    def workout_activity_overview(self, workout_activity_type: str):
        assert workout_activity_type in self.workout_activity_type, "Invalid workout_activity_type"
        activity_df = self._obj[self._obj['workoutActivityType'] == 'workout_activity_type'].copy()
        # TBD
        return activity_df

    def all_workout_activity_overview(self):
        activity_types_df = self._obj[self._obj['workoutActivityType'].isin(self.workout_activity_type)].copy()
        activity_types_df['averageDuration'] = activity_types_df.groupby('workoutActivityType')['duration'].transform('mean')
        average_duration_dict = activity_types_df.groupby('workoutActivityType')['averageDuration'].first().to_dict()
        occurence_dict = activity_types_df['workoutActivityType'].value_counts().to_dict()
        overview_df = pd.DataFrame({
            'averageDuration': average_duration_dict,
            'occurrence': occurence_dict
        })
        return overview_df

    @property
    def types(self):
        """Return all unique types in the data."""
        return self._obj['type'].dropna().unique()

    @property
    def workout_activity_type(self):
        """Return all unique activity types."""
        if 'workoutActivityType' in self._obj.columns:
            return self._obj['workoutActivityType'].dropna().unique()
        return []

    @property
    def source_names(self):
        """Return all unique sourceName values."""
        return self._obj['sourceName'].dropna().unique()

    def get_heart_rate_for_date(self, target_date):
        """Get all heart rate data for a specific date."""
        if 'HeartRate' not in self._obj['type'].unique():
            return pd.DataFrame()
            
        # Normalize target date
        target_date = pd.to_datetime(target_date)
        
        # Get start and end of the target date
        start_of_day = target_date.normalize()
        end_of_day = start_of_day + pd.Timedelta(days=1)
        
        hr_data = self._obj[
            (self._obj['startDate'] >= start_of_day) &
            (self._obj['startDate'] < end_of_day) &
            (self._obj['type'] == 'HeartRate')
        ][['startDate', 'value']].copy()
        
        if hr_data.empty:
            return pd.DataFrame()
            
        hr_data['startDate'] = pd.to_datetime(hr_data['startDate'], errors='coerce')
        hr_data['value'] = pd.to_numeric(hr_data['value'], errors='coerce')
        hr_data = hr_data.dropna().reset_index(drop=True)
        
        # Add time_seconds column for easier synchronization
        if not hr_data.empty:
            start_time = hr_data['startDate'].min()
            hr_data['time_seconds'] = (hr_data['startDate'] - start_time).dt.total_seconds()
        
        return hr_data

    def get_gaming_session_heart_rate(self, date):
        """Get heart rate data specifically for gaming sessions on a given date."""
        try:
            gaming_sessions = self.get_gaming_sessions()
            if gaming_sessions.empty:
                print(f"No gaming sessions found")
                return pd.DataFrame()
            
            # Filter sessions by date
            target_date = pd.to_datetime(date).date()
            session_on_date = gaming_sessions[gaming_sessions['startDate'].dt.date == target_date]
            
            if session_on_date.empty:
                print(f"No gaming session found on {date}")
                return pd.DataFrame()
            
            # Get the first gaming session on that date
            session = session_on_date.iloc[0]
            start_date = session['startDate']
            end_date = session['endDate']
            
            # Get heart rate data for this gaming session
            hr_data = self._obj[
                (self._obj['startDate'] >= start_date) &
                (self._obj['startDate'] <= end_date) &
                (self._obj['type'] == 'HeartRate')
            ][['startDate', 'value']].copy()
            
            if hr_data.empty:
                return pd.DataFrame()
            
            hr_data = hr_data.sort_values('startDate').reset_index(drop=True)
            hr_data['startDate'] = pd.to_datetime(hr_data['startDate'], errors='coerce')
            hr_data['value'] = pd.to_numeric(hr_data['value'], errors='coerce')
            hr_data = hr_data.dropna().reset_index(drop=True)
            
            # Add time_seconds from session start for synchronization
            if not hr_data.empty:
                hr_data['time_seconds'] = (hr_data['startDate'] - start_date).dt.total_seconds()
                hr_data['session_start'] = start_date
                hr_data['session_end'] = end_date
                hr_data['session_duration'] = (end_date - start_date).total_seconds()
                
                # Add measurement period attributes like EKGAnalyzer
                hr_data.measure_start_date = start_date
                hr_data.measure_end_date = end_date
            
            return hr_data
            
        except Exception as e:
            print(f"Error getting gaming session heart rate: {e}")
            return pd.DataFrame()

    def get_heart_rate_statistics(self, date):
        """Get comprehensive heart rate statistics for a given date."""
        hr_data = self.get_heart_rate_for_date(date)
        
        if hr_data.empty:
            return None
            
        stats = {
            'date': date,
            'total_measurements': len(hr_data),
            'mean_hr': hr_data['value'].mean(),
            'median_hr': hr_data['value'].median(),
            'min_hr': hr_data['value'].min(),
            'max_hr': hr_data['value'].max(),
            'std_hr': hr_data['value'].std(),
            'hr_range': hr_data['value'].max() - hr_data['value'].min(),
            'duration_minutes': hr_data['time_seconds'].max() / 60 if 'time_seconds' in hr_data.columns else None
        }
        
        # Calculate heart rate zones (approximate)
        if stats['mean_hr'] > 0:
            stats['resting_zone'] = len(hr_data[hr_data['value'] < stats['mean_hr'] * 0.6])
            stats['fat_burn_zone'] = len(hr_data[(hr_data['value'] >= stats['mean_hr'] * 0.6) & 
                                                 (hr_data['value'] < stats['mean_hr'] * 0.7)])
            stats['cardio_zone'] = len(hr_data[(hr_data['value'] >= stats['mean_hr'] * 0.7) & 
                                               (hr_data['value'] < stats['mean_hr'] * 0.85)])
            stats['peak_zone'] = len(hr_data[hr_data['value'] >= stats['mean_hr'] * 0.85])
        
        return stats

    def compare_gaming_vs_daily_hr(self, date):
        """Compare heart rate during gaming session vs entire day."""
        daily_hr = self.get_heart_rate_for_date(date)
        gaming_hr = self.get_gaming_session_heart_rate(date)
        
        if daily_hr.empty:
            return None
            
        comparison = {
            'date': date,
            'daily_measurements': len(daily_hr),
            'daily_mean_hr': daily_hr['value'].mean(),
            'daily_max_hr': daily_hr['value'].max(),
            'daily_min_hr': daily_hr['value'].min()
        }
        
        if not gaming_hr.empty:
            comparison.update({
                'gaming_measurements': len(gaming_hr),
                'gaming_mean_hr': gaming_hr['value'].mean(),
                'gaming_max_hr': gaming_hr['value'].max(),
                'gaming_min_hr': gaming_hr['value'].min(),
                'gaming_vs_daily_mean_diff': gaming_hr['value'].mean() - daily_hr['value'].mean(),
                'gaming_session_duration_min': gaming_hr['session_duration'].iloc[0] / 60 if 'session_duration' in gaming_hr.columns else None
            })
        else:
            comparison.update({
                'gaming_measurements': 0,
                'gaming_mean_hr': None,
                'gaming_max_hr': None,
                'gaming_min_hr': None,
                'gaming_vs_daily_mean_diff': None,
                'gaming_session_duration_min': None
            })
            
        return comparison

    @property
    def measure_start_date(self):
        """Get the earliest measurement date in the dataset."""
        if 'startDate' not in self._obj.columns or self._obj.empty:
            return None
        return self._obj['startDate'].min()
    
    @property 
    def measure_end_date(self):
        """Get the latest measurement date in the dataset."""
        if 'endDate' not in self._obj.columns or self._obj.empty:
            return None
        return self._obj['endDate'].max()
    
    def get_gaming_session_start_date(self, date):
        """Get the start date of a specific gaming session."""
        gaming_start, _ = self.get_gaming_session_measurement_period(date)
        return gaming_start
    
    def get_gaming_session_end_date(self, date):
        """Get the end date of a specific gaming session."""
        _, gaming_end = self.get_gaming_session_measurement_period(date)
        return gaming_end
    
    def get_measurement_period_for_date(self, target_date):
        """Get the measurement period (start and end) for a specific date."""
        target_date = pd.to_datetime(target_date).normalize()
        end_of_day = target_date + pd.Timedelta(days=1)
        
        date_data = self._obj[
            (self._obj['startDate'] >= target_date) &
            (self._obj['startDate'] < end_of_day)
        ]
        
        if date_data.empty:
            return None, None
            
        return date_data['startDate'].min(), date_data['endDate'].max()
    
    def get_gaming_session_measurement_period(self, date):
        """Get the exact measurement period for a gaming session on a given date."""
        try:
            gaming_sessions = self.get_gaming_sessions()
            if gaming_sessions.empty:
                return None, None
            
            # Filter sessions by date
            target_date = pd.to_datetime(date).date()
            session_on_date = gaming_sessions[gaming_sessions['startDate'].dt.date == target_date]
            
            if session_on_date.empty:
                return None, None
            
            # Get the first gaming session on that date
            session = session_on_date.iloc[0]
            return session['startDate'], session['endDate']
            
        except Exception as e:
            print(f"Error getting gaming session measurement period: {e}")
            return None, None
    
    def get_session_from_date(self, target_date):
        """
        Get a session (gaming or general activity session) from a specific date.
        
        Parameters:
        -----------
        target_date : str
            Date in YYYY-MM-DD format
            
        Returns:
        --------
        dict
            Session information with startDate and endDate
        """
        try:
            # First try to get gaming sessions
            gaming_sessions = self.get_gaming_sessions()
            if not gaming_sessions.empty:
                target_date_obj = pd.to_datetime(target_date).date()
                session_on_date = gaming_sessions[gaming_sessions['startDate'].dt.date == target_date_obj]
                
                if not session_on_date.empty:
                    session = session_on_date.iloc[0]
                    return {
                        'startDate': session['startDate'],
                        'endDate': session['endDate'],
                        'type': 'gaming_session',
                        'workoutActivityType': session.get('workoutActivityType', 'Unknown')
                    }
            
            # If no gaming session found, get general session period for the date
            start_date, end_date = self.get_measurement_period_for_date(target_date)
            if start_date and end_date:
                return {
                    'startDate': start_date,
                    'endDate': end_date,
                    'type': 'general_session'
                }
            
            # If nothing found, return None
            return None
            
        except Exception as e:
            print(f"Error getting session from date {target_date}: {e}")
            return None
    
    def get_heart_rate_stats_from_session(self, session):
        """
        Get heart rate statistics from a session object.
        
        Parameters:
        -----------
        session : dict
            Session object containing startDate and endDate
            
        Returns:
        --------
        pd.DataFrame
            Heart rate data for the session with time_seconds column
        """
        if not session or 'startDate' not in session or 'endDate' not in session:
            return pd.DataFrame()
            
        try:
            start_date = session['startDate']
            end_date = session['endDate']
            
            # Get heart rate data for this session period
            hr_data = self._obj[
                (self._obj['startDate'] >= start_date) &
                (self._obj['startDate'] <= end_date) &
                (self._obj['type'] == 'HeartRate')
            ][['startDate', 'value']].copy()
            
            if hr_data.empty:
                return pd.DataFrame()
            
            hr_data = hr_data.sort_values('startDate').reset_index(drop=True)
            hr_data['startDate'] = pd.to_datetime(hr_data['startDate'], errors='coerce')
            hr_data['value'] = pd.to_numeric(hr_data['value'], errors='coerce')
            hr_data = hr_data.dropna().reset_index(drop=True)
            
            # Add time_seconds from session start for synchronization
            if not hr_data.empty:
                hr_data['time_seconds'] = (hr_data['startDate'] - start_date).dt.total_seconds()
                hr_data['session_start'] = start_date
                hr_data['session_end'] = end_date
                hr_data['session_duration'] = (end_date - start_date).total_seconds()
            
            return hr_data
            
        except Exception as e:
            print(f"Error getting heart rate stats from session: {e}")
            return pd.DataFrame()
    
    @classmethod
    def read_file(cls, file_path, **kwargs):
        """Read a CSV file and return an instance of AppleWatchAnalyzer."""
        df = pd.read_csv(file_path, **kwargs)
        return df

# %% usage example
# apple_watch_data = pd.DataFrame.applewatch.read_file('gaming_health_data/recorded_data/APPLE_WATCH/apple_health_export_2025-11-08.csv')
# apple_watch_data.applewatch.plot_heart_rate_for_session_date('2025-11-08')
# types = apple_watch_data.applewatch.workout_activity_type
# apple_watch_data.applewatch.plot_heart_rate_for_session_date('2025-11-08')

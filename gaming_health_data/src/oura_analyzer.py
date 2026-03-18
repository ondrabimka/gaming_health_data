import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, Union
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)


@pd.api.extensions.register_dataframe_accessor("oura")
class OuraAnalyzer:
    """
    Oura Ring data analyzer for multimodal gaming health analysis.
    
    Provides essential methods for extracting heart rate data synchronized
    with gaming sessions for cross-device comparison with EKG and Apple Watch.
    """
    
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._data_cache = {}
        
    @classmethod
    def read_oura_data(cls, data_path: Union[str, Path]) -> pd.DataFrame:
        """
        Read Oura data files and combine into comprehensive dataset.
        
        Parameters:
        -----------
        data_path : str or Path
            Path to directory containing Oura CSV files or JSON file
            
        Returns:
        --------
        pd.DataFrame
            Combined Oura data with all metrics
        """
        data_path = Path(data_path)
        
        # Check if it's a JSON file
        if data_path.is_file() and data_path.suffix == '.json':
            with open(data_path, 'r') as f:
                json_data = json.load(f)
            
            # Extract heart rate data from JSON
            if 'heartrate' in json_data:
                hr_records = json_data['heartrate']
                hr_df = pd.DataFrame(hr_records)
                hr_df['timestamp'] = pd.to_datetime(hr_df['timestamp'])
                hr_df['date'] = hr_df['timestamp'].dt.date
                
                # Store all data in attributes
                hr_df.attrs['oura_data'] = {'heartrate': hr_df}
                
                print(f"Loaded Oura JSON: {len(hr_df)} heart rate records")
                return hr_df
        
        # Otherwise, read from CSV files directory
        oura_files = {
            'heartrate': 'heartrate.csv',
            'temperature': 'temperature.csv',
            'dailystress': 'dailystress.csv',
            'dailyreadiness': 'dailyreadiness.csv',
            'dailyactivity': 'dailyactivity.csv',
        }
        
        data = {}
        
        # Read each file if it exists
        for key, filename in oura_files.items():
            file_path = data_path / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, sep=';')
                    print(f"Loaded {filename}: {len(df)} records")
                    data[key] = df
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        # Create main dataframe
        main_df = pd.DataFrame()
        
        if 'heartrate' in data:
            hr_df = data['heartrate'].copy()
            hr_df['timestamp'] = pd.to_datetime(hr_df['timestamp'])
            hr_df['date'] = hr_df['timestamp'].dt.date
            main_df = hr_df
            main_df.attrs['oura_data'] = data
            
        print(f"Combined Oura dataset: {len(main_df)} records")
        print(f"Available metrics: {list(data.keys())}")
        
        return main_df
    
    def _get_data(self, data_type: str) -> pd.DataFrame:
        """Get specific data type from cache or main dataframe."""
        if data_type in self._data_cache:
            return self._data_cache[data_type]
            
        if hasattr(self._obj, 'attrs') and 'oura_data' in self._obj.attrs:
            oura_data = self._obj.attrs['oura_data']
            if data_type in oura_data:
                self._data_cache[data_type] = oura_data[data_type]
                return oura_data[data_type]
        
        return pd.DataFrame()
    
    def get_heart_rate_for_date(self, target_date: str) -> pd.DataFrame:
        """
        Get all heart rate data for a specific date.
        
        Parameters:
        -----------
        target_date : str
            Date in YYYY-MM-DD format
            
        Returns:
        --------
        pd.DataFrame
            Heart rate data with columns: timestamp, bpm, time_seconds
        """
        hr_data = self._get_data('heartrate')
        if hr_data.empty:
            return pd.DataFrame()
        
        # Normalize target date
        target_date = pd.to_datetime(target_date).date()
        
        # Filter by date
        hr_df = hr_data.copy()
        hr_df['timestamp'] = pd.to_datetime(hr_df['timestamp'])
        hr_df['date'] = hr_df['timestamp'].dt.date
        
        date_hr = hr_df[hr_df['date'] == target_date].copy()
        
        if date_hr.empty:
            return pd.DataFrame()
        
        # Add time_seconds column for synchronization with gaming data
        start_time = date_hr['timestamp'].min()
        date_hr['time_seconds'] = (date_hr['timestamp'] - start_time).dt.total_seconds()
        
        # Keep both 'bpm' and 'value' columns for compatibility
        date_hr = date_hr.rename(columns={'bpm': 'bpm_original'})
        date_hr['bpm'] = date_hr['bpm_original']
        date_hr['value'] = date_hr['bpm_original']
        
        return date_hr[['timestamp', 'bpm', 'value', 'time_seconds']].sort_values('timestamp').reset_index(drop=True)
    
    def get_session_from_date(self, target_date: str) -> dict:
        """
        Get session information for a specific date.
        Compatible with Apple Watch analyzer interface.
        
        This returns the time boundaries of Oura measurements for the target date,
        which should align with the gaming session if Oura was recording during gaming.
        
        Parameters:
        -----------
        target_date : str
            Date in YYYY-MM-DD format
            
        Returns:
        --------
        dict
            Session information with startDate and endDate (datetime objects)
        """
        hr_data = self._get_data('heartrate')
        if hr_data.empty:
            return None
        
        # Normalize target date
        target_date_dt = pd.to_datetime(target_date).date()
        
        # Get all heart rate data for the date
        hr_df = hr_data.copy()
        hr_df['timestamp'] = pd.to_datetime(hr_df['timestamp'])
        hr_df['date'] = hr_df['timestamp'].dt.date
        
        date_hr = hr_df[hr_df['date'] == target_date_dt]
        
        if date_hr.empty:
            print(f"[WARNING] No Oura data found for {target_date}")
            return None
        
        # Filter to the active gaming session window.
        # 'workout' = active exercise/gaming session (highest priority)
        # 'awake'   = active but not a workout (fallback)
        # 'rest'    = sleep/rest data (excluded)
        if 'source' in date_hr.columns:
            workout_hr = date_hr[date_hr['source'] == 'workout'].sort_values('timestamp').reset_index(drop=True)
            if not workout_hr.empty:
                # There may be multiple workout blocks in a day (e.g., morning run + evening gaming).
                # Split into contiguous blocks separated by gaps > 30 min and return the LAST one,
                # which corresponds to the most recent (gaming) session.
                time_diffs = workout_hr['timestamp'].diff()
                gap_threshold = pd.Timedelta(minutes=30)
                block_ids = (time_diffs > gap_threshold).cumsum()
                last_block_id = block_ids.max()
                active_hr = workout_hr[block_ids == last_block_id]
                total_blocks = last_block_id + 1
                print(f"[DEBUG] Found {total_blocks} workout block(s); using last block: {len(active_hr)} measurements")
            else:
                active_hr = date_hr[date_hr['source'] != 'rest']
                if active_hr.empty:
                    active_hr = date_hr
                print(f"[DEBUG] No workout data, using non-rest data: {len(active_hr)} measurements")
        else:
            active_hr = date_hr
        
        start_time = active_hr['timestamp'].min()
        end_time = active_hr['timestamp'].max()
        
        print(f"[DEBUG] Oura session: {start_time} to {end_time}")
        print(f"[DEBUG] Duration: {(end_time - start_time).total_seconds() / 60:.1f} minutes")
        
        return {
            'startDate': start_time,
            'endDate': end_time,
            'date': str(target_date),
            'source': 'Oura Ring',
            'measurement_count': len(active_hr)
        }
    
    def get_heart_rate_stats_from_session(self, session: dict) -> pd.DataFrame:
        """
        Get heart rate statistics from a session object.
        Compatible with Apple Watch analyzer interface.
        
        Parameters:
        -----------
        session : dict
            Session object containing startDate and endDate
            
        Returns:
        --------
        pd.DataFrame
            Heart rate data for the session with time_seconds column
        """
        if not session:
            return pd.DataFrame()
        
        # Get all heart rate data
        hr_data = self._get_data('heartrate')
        if hr_data.empty:
            return pd.DataFrame()
        
        hr_df = hr_data.copy()
        hr_df['timestamp'] = pd.to_datetime(hr_df['timestamp'])
        
        # Filter by session timeframe if startDate and endDate are provided
        if 'startDate' in session and 'endDate' in session:
            start_date = pd.to_datetime(session['startDate'])
            end_date = pd.to_datetime(session['endDate'])
            
            # Filter to session timeframe
            hr_df = hr_df[
                (hr_df['timestamp'] >= start_date) &
                (hr_df['timestamp'] <= end_date)
            ].copy()
            
            if hr_df.empty:
                print(f"[WARNING] No Oura heart rate data found between {start_date} and {end_date}")
                return pd.DataFrame()
            
            # Calculate time_seconds relative to session start
            hr_df['time_seconds'] = (hr_df['timestamp'] - start_date).dt.total_seconds()
            
        elif 'date' in session:
            # Fallback to date-based filtering
            target_date = pd.to_datetime(session['date']).date()
            hr_df['date'] = hr_df['timestamp'].dt.date
            hr_df = hr_df[hr_df['date'] == target_date].copy()
            
            if hr_df.empty:
                return pd.DataFrame()
            
            # Calculate time_seconds relative to first measurement
            hr_df['time_seconds'] = (hr_df['timestamp'] - hr_df['timestamp'].min()).dt.total_seconds()
        else:
            return pd.DataFrame()
        
        # Standardize column names
        hr_df = hr_df.rename(columns={'bpm': 'bpm_original'})
        hr_df['bpm'] = hr_df['bpm_original']
        hr_df['value'] = hr_df['bpm_original']
        
        result = hr_df[['timestamp', 'bpm', 'value', 'time_seconds']].sort_values('timestamp').reset_index(drop=True)
        
        print(f"[DEBUG] Oura HR data filtered: {len(result)} measurements")
        if len(result) > 0:
            print(f"[DEBUG] Oura time range: {result['time_seconds'].min():.1f}s to {result['time_seconds'].max():.1f}s")
            print(f"[DEBUG] Oura HR range: {result['value'].min():.0f} - {result['value'].max():.0f} BPM")
        
        return result
    
    def get_measurement_period_for_date(self, target_date: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the measurement period (start and end) for a specific date.
        
        Parameters:
        -----------
        target_date : str
            Date in YYYY-MM-DD format
            
        Returns:
        --------
        tuple
            (start_time, end_time) as datetime objects
        """
        hr_data = self.get_heart_rate_for_date(target_date)
        
        if hr_data.empty:
            return None, None
        
        return hr_data['timestamp'].min(), hr_data['timestamp'].max()
    
    def get_temperature_for_date(self, target_date: str) -> pd.DataFrame:
        """
        Get skin temperature data for a specific date.
        
        Parameters:
        -----------
        target_date : str
            Date in YYYY-MM-DD format
            
        Returns:
        --------
        pd.DataFrame
            Temperature data with columns: timestamp, skin_temp, time_seconds
        """
        temp_data = self._get_data('temperature')
        if temp_data.empty:
            return pd.DataFrame()
        
        # Normalize target date
        target_date = pd.to_datetime(target_date).date()
        
        # Filter by date
        temp_df = temp_data.copy()
        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
        temp_df['date'] = temp_df['timestamp'].dt.date
        
        date_temp = temp_df[temp_df['date'] == target_date].copy()
        
        if date_temp.empty:
            return pd.DataFrame()
        
        # Add time_seconds column for synchronization
        start_time = date_temp['timestamp'].min()
        date_temp['time_seconds'] = (date_temp['timestamp'] - start_time).dt.total_seconds()
        
        return date_temp[['timestamp', 'skin_temp', 'time_seconds']].sort_values('timestamp').reset_index(drop=True)
    
    def get_readiness_for_date(self, target_date: str) -> dict:
        """
        Get readiness metrics for a specific date.
        
        Parameters:
        -----------
        target_date : str
            Date in YYYY-MM-DD format
            
        Returns:
        --------
        dict
            Readiness metrics including score and contributors
        """
        readiness_data = self._get_data('dailyreadiness')
        if readiness_data.empty:
            return None
        
        # Normalize target date
        target_date = pd.to_datetime(target_date).date()
        
        # Filter by date
        readiness_df = readiness_data.copy()
        readiness_df['day'] = pd.to_datetime(readiness_df['day']).dt.date
        
        date_readiness = readiness_df[readiness_df['day'] == target_date]
        
        if date_readiness.empty:
            return None
        
        # Extract first row
        row = date_readiness.iloc[0]
        
        return {
            'score': row.get('score'),
            'contributors': row.get('contributors'),
            'temperature_deviation': row.get('temperature_deviation'),
            'date': target_date
        }
    
    def get_activity_for_date(self, target_date: str) -> dict:
        """
        Get activity metrics for a specific date.
        
        Parameters:
        -----------
        target_date : str
            Date in YYYY-MM-DD format
            
        Returns:
        --------
        dict
            Activity metrics including steps, calories, and activity score
        """
        activity_data = self._get_data('dailyactivity')
        if activity_data.empty:
            return None
        
        # Normalize target date
        target_date = pd.to_datetime(target_date).date()
        
        # Filter by date
        activity_df = activity_data.copy()
        activity_df['day'] = pd.to_datetime(activity_df['day']).dt.date
        
        date_activity = activity_df[activity_df['day'] == target_date]
        
        if date_activity.empty:
            return None
        
        # Extract first row
        row = date_activity.iloc[0]
        
        return {
            'score': row.get('score'),
            'steps': row.get('steps'),
            'active_calories': row.get('active_calories'),
            'total_calories': row.get('total_calories'),
            'sedentary_time': row.get('sedentary_time'),
            'high_activity_time': row.get('high_activity_time'),
            'date': target_date
        }

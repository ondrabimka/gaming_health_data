# %%
import pandas as pd
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px



@pd.api.extensions.register_dataframe_accessor("Video")
class VideoAnalyzer:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._validate()

    def _validate(self):
        # Check if this is health data or action data
        if "time" in self._obj.columns and "current_health" in self._obj.columns and "max_health" in self._obj.columns:
            # This is health data - no additional validation needed
            return
        elif "action" in self._obj.columns and "timestamp" in self._obj.columns:
            # This is action data - original validation
            return
        else:
            # Neither format is recognized
            available_cols = list(self._obj.columns)
            raise AttributeError(f"DataFrame format not recognized. Available columns: {available_cols}. Expected either ['time', 'current_health', 'max_health'] for health data or ['action', 'timestamp'] for action data.")

    @staticmethod
    def from_file(file_path):
        return pd.read_csv(file_path)
    
    @staticmethod
    def from_health_csv(file_path):
        """Load health data from CSV and return cleaned DataFrame"""
        df = pd.read_csv(file_path)
        return df
    
    def clean_health_data(self, max_health_threshold=999, min_health_ratio=0.1, death_window=5.0):
        """
        Clean health data by removing invalid readings and handling deaths.
        
        Parameters:
        -----------
        max_health_threshold : int, optional (default=999)
            Maximum reasonable health value
        min_health_ratio : float, optional (default=0.1) 
            Minimum ratio of current_health/max_health to consider valid
        death_window : float, optional (default=5.0)
            Time window in seconds after death to remove subsequent readings
            
        Returns:
        --------
        pd.DataFrame
            Cleaned health data
        """
        # Ensure we have the required columns
        if not all(col in self._obj.columns for col in ['time', 'current_health', 'max_health']):
            raise ValueError("DataFrame must contain 'time', 'current_health', and 'max_health' columns")
        
        df = self._obj.copy()
        
        # Step 1: Remove rows with missing values
        df_clean = df.dropna(subset=['current_health', 'max_health']).copy()
        
        # Step 2: Filter out unreasonable health values
        valid_mask = (
            (df_clean['current_health'] >= 0) &
            (df_clean['max_health'] > 0) &
            (df_clean['max_health'] <= max_health_threshold) &
            (df_clean['current_health'] <= df_clean['max_health'])
        )
        df_clean = df_clean[valid_mask].copy()
        
        # Step 3: Remove readings where health ratio is too low (likely OCR errors)
        health_ratio = df_clean['current_health'] / df_clean['max_health']
        ratio_mask = (df_clean['current_health'] == 0) | (health_ratio >= min_health_ratio)
        df_clean = df_clean[ratio_mask].copy()
        
        # Step 4: Handle deaths (current_health = 0)
        df_clean = self._handle_deaths(df_clean, death_window)
        
        # Step 5: Remove consecutive duplicate readings
        df_clean = self._remove_duplicate_readings(df_clean)
        
        return df_clean.reset_index(drop=True)
    
    def _handle_deaths(self, df, death_window):
        """
        Handle death events by removing subsequent readings within death_window.
        """
        if df.empty:
            return df
            
        death_indices = df[df['current_health'] == 0].index.tolist()
        indices_to_remove = []
        
        for death_idx in death_indices:
            death_time = df.loc[death_idx, 'time']
            
            # Find all readings within death_window seconds after death
            window_mask = (
                (df['time'] > death_time) & 
                (df['time'] <= death_time + death_window) &
                (df['current_health'] > 0)  # Remove non-zero health readings after death
            )
            
            window_indices = df[window_mask].index.tolist()
            indices_to_remove.extend(window_indices)
        
        # Remove the identified indices
        return df.drop(indices_to_remove)
    
    def _remove_duplicate_readings(self, df):
        """
        Remove consecutive readings with identical health values.
        """
        if df.empty:
            return df
            
        # Keep first occurrence and rows where health changed
        mask = (
            (df['current_health'].shift() != df['current_health']) |
            (df['max_health'].shift() != df['max_health'])
        )
        
        # Always keep the first row
        mask.iloc[0] = True
        
        return df[mask]
    
    def analyze_health_patterns(self):
        """
        Analyze health patterns in the cleaned data.
        
        Returns:
        --------
        dict
            Dictionary containing various health statistics
        """
        df = self._obj
        
        if df.empty:
            return {}
        
        # Calculate health statistics
        stats = {
            'total_duration': df['time'].max() - df['time'].min(),
            'total_readings': len(df),
            'death_count': len(df[df['current_health'] == 0]),
            'avg_health': df[df['current_health'] > 0]['current_health'].mean(),
            'min_health': df[df['current_health'] > 0]['current_health'].min(),
            'max_health_value': df['max_health'].max(),
            'health_changes': len(df) - 1,
            'time_at_full_health': self._calculate_full_health_time(),
            'time_below_half_health': self._calculate_low_health_time()
        }
        
        return stats
    
    def _calculate_full_health_time(self):
        """Calculate total time spent at full health."""
        df = self._obj
        if df.empty:
            return 0
            
        full_health_mask = df['current_health'] == df['max_health']
        full_health_periods = df[full_health_mask]
        
        if len(full_health_periods) < 2:
            return 0
            
        # Estimate time spent at full health
        time_diffs = full_health_periods['time'].diff().dropna()
        return time_diffs.sum()
    
    def _calculate_low_health_time(self):
        """Calculate total time spent below 50% health."""
        df = self._obj
        if df.empty:
            return 0
            
        low_health_mask = (df['current_health'] > 0) & (df['current_health'] < df['max_health'] * 0.5)
        low_health_periods = df[low_health_mask]
        
        if len(low_health_periods) < 2:
            return 0
            
        # Estimate time spent at low health
        time_diffs = low_health_periods['time'].diff().dropna()
        return time_diffs.sum()
    
    def plot_health_timeline(self, title="Health Over Time"):
        """
        Plot health timeline with deaths highlighted.
        """
        df = self._obj
        
        if df.empty:
            print("No data to plot")
            return
        
        fig = go.Figure()
        
        # Plot current health
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['current_health'],
            mode='lines+markers',
            name='Current Health',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ))
        
        # Plot max health
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['max_health'],
            mode='lines',
            name='Max Health',
            line=dict(color='blue', width=1, dash='dash')
        ))
        
        # Highlight deaths
        deaths = df[df['current_health'] == 0]
        if not deaths.empty:
            fig.add_trace(go.Scatter(
                x=deaths['time'],
                y=deaths['current_health'],
                mode='markers',
                name='Deaths',
                marker=dict(color='black', size=10, symbol='x')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time (seconds)",
            yaxis_title="Health",
            hovermode='x unified'
        )
        
        fig.show()

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


# %%
# Load and clean health data
health_df = pd.read_csv("Cell output 6 [DW].csv")

# Clean the data
cleaned_health = health_df.Video.clean_health_data(
    max_health_threshold=999,
    min_health_ratio=0.1,
    death_window=5.0
)

# Analyze patterns
stats = cleaned_health.Video.analyze_health_patterns()
print("Health Statistics:")
for key, value in stats.items():
    print(f"{key}: {value}")

# Plot the timeline
cleaned_health.Video.plot_health_timeline("Player Health Over Time")

# Export cleaned data
cleaned_health.to_csv("cleaned_health_data.csv", index=False)
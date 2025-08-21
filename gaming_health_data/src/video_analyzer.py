# %%
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@pd.api.extensions.register_dataframe_accessor("Video")
class VideoAnalyzer:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._validate()

    def _validate(self):
        """
        Validate DataFrame format and determine data type.
        Supports:
        - Automatic health data: ['time', 'current_health', 'max_health']  
        - Manual action data: ['timestamp', 'action']
        """
        # Check if this is health data (automatic annotation)
        if "time" in self._obj.columns and "current_health" in self._obj.columns and "max_health" in self._obj.columns:
            self.data_type = "automatic_health"
            return
        # Check if this is action data (manual annotation)
        elif "action" in self._obj.columns and "timestamp" in self._obj.columns:
            self.data_type = "manual_actions"
            return
        else:
            # Neither format is recognized
            available_cols = list(self._obj.columns)
            raise AttributeError(f"DataFrame format not recognized. Available columns: {available_cols}. Expected either ['time', 'current_health', 'max_health'] for automatic health data or ['action', 'timestamp'] for manual action data.")

    @staticmethod
    def from_file(file_path):
        """Load data from CSV file and return DataFrame with Video accessor."""
        return pd.read_csv(file_path)
    
    @staticmethod
    def from_health_csv(file_path):
        """Load automatic health data from CSV and return cleaned DataFrame."""
        df = pd.read_csv(file_path)
        return df
    
    @staticmethod
    def from_manual_csv(file_path):
        """Load manual action data from CSV and return processed DataFrame."""
        df = pd.read_csv(file_path)
        return df
    
    def convert_timestamp_to_seconds(self):
        """
        Convert timestamp column from HH:MM:SS format to seconds.
        Only works with manual action data.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with timestamp converted to seconds
        """
        if self.data_type != "manual_actions":
            raise ValueError("This method only works with manual action data")
            
        df = self._obj.copy()
        
        # Convert timestamp to seconds
        def time_to_seconds(time_str):
            try:
                time_obj = datetime.strptime(str(time_str), '%H:%M:%S')
                return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
            except:
                return None
                
        df['time_seconds'] = df['timestamp'].apply(time_to_seconds)
        return df
    
    def get_data_info(self):
        """
        Get comprehensive information about the loaded data.
        
        Returns:
        --------
        dict
            Dictionary with data statistics and information
        """
        df = self._obj
        info = {
            'data_type': self.data_type,
            'total_rows': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
        }
        
        if self.data_type == "automatic_health":
            info.update({
                'duration_seconds': df['time'].max() - df['time'].min() if not df['time'].isna().all() else 0,
                'health_readings': len(df.dropna(subset=['current_health'])),
                'unique_health_values': df['current_health'].nunique(),
                'max_health_values': df['max_health'].unique(),
                'death_count': len(df[df['current_health'] == 0]) if 'current_health' in df.columns else 0
            })
        
        elif self.data_type == "manual_actions":
            df_time = self.convert_timestamp_to_seconds()
            info.update({
                'duration_seconds': df_time['time_seconds'].max() - df_time['time_seconds'].min() if not df_time['time_seconds'].isna().all() else 0,
                'unique_actions': df['action'].nunique(),
                'action_counts': df['action'].value_counts().to_dict(),
                'total_events': len(df)
            })
        
        return info
    
    def extract_events_from_health(self, health_threshold=0.3, time_window=2.0):
        """
        Extract gaming events from automatic health data.
        
        Parameters:
        -----------
        health_threshold : float
            Minimum health drop ratio to consider as damage event
        time_window : float
            Time window for grouping events (seconds)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with extracted events
        """
        if self.data_type != "automatic_health":
            raise ValueError("This method only works with automatic health data")
            
        df = self.clean_health_data()
        events = []
        
        if df.empty:
            return pd.DataFrame(columns=['time_seconds', 'event_type', 'health_before', 'health_after'])
        
        # Detect deaths
        death_mask = df['current_health'] == 0
        death_times = df[death_mask]['time'].values
        for death_time in death_times:
            events.append({
                'time_seconds': death_time,
                'event_type': 'death',
                'health_before': None,
                'health_after': 0
            })
        
        # Detect major health drops
        df['health_diff'] = df['current_health'].diff()
        df['time_diff'] = df['time'].diff()
        
        # Find significant health drops
        damage_mask = (
            (df['health_diff'] < -30) & 
            (df['time_diff'] < time_window) &
            (df['current_health'] > 0)
        )
        
        damage_events = df[damage_mask]
        for _, row in damage_events.iterrows():
            events.append({
                'time_seconds': row['time'],
                'event_type': 'damage',
                'health_before': row['current_health'] - row['health_diff'],
                'health_after': row['current_health']
            })
        
        # Detect healing events
        heal_mask = (
            (df['health_diff'] > 30) & 
            (df['time_diff'] < time_window)
        )
        
        heal_events = df[heal_mask]
        for _, row in heal_events.iterrows():
            events.append({
                'time_seconds': row['time'],
                'event_type': 'heal',
                'health_before': row['current_health'] - row['health_diff'],
                'health_after': row['current_health']
            })
        
        return pd.DataFrame(events).sort_values('time_seconds').reset_index(drop=True)
    
    @staticmethod
    def compare_annotations(manual_df, automatic_df, time_tolerance=3.0):
        """
        Compare manual and automatic annotations to assess quality.
        
        Parameters:
        -----------
        manual_df : pd.DataFrame
            Manual annotation data with Video accessor
        automatic_df : pd.DataFrame  
            Automatic annotation data with Video accessor
        time_tolerance : float
            Time tolerance for matching events (seconds)
            
        Returns:
        --------
        dict
            Comparison results including matches, misses, and quality metrics
        """
        # Convert manual annotations to seconds
        manual_time = manual_df.Video.convert_timestamp_to_seconds()
        
        # Extract events from automatic data
        auto_events = automatic_df.Video.extract_events_from_health()
        
        if auto_events.empty:
            return {
                'manual_events': len(manual_time),
                'automatic_events': 0,
                'matches': 0,
                'quality_score': 0.0,
                'details': "No automatic events detected"
            }
        
        # Focus on death events for comparison (most reliable)
        manual_deaths = manual_time[manual_time['action'] == 'death']['time_seconds'].values
        auto_deaths = auto_events[auto_events['event_type'] == 'death']['time_seconds'].values
        
        matches = 0
        matched_manual = set()
        matched_auto = set()
        
        # Find matches within tolerance
        for i, manual_death_time in enumerate(manual_deaths):
            for j, auto_death_time in enumerate(auto_deaths):
                if abs(manual_death_time - auto_death_time) <= time_tolerance:
                    matches += 1
                    matched_manual.add(i)
                    matched_auto.add(j)
                    break
        
        # Calculate quality metrics
        precision = matches / len(auto_deaths) if len(auto_deaths) > 0 else 0
        recall = matches / len(manual_deaths) if len(manual_deaths) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'manual_events': len(manual_time),
            'automatic_events': len(auto_events),
            'manual_deaths': len(manual_deaths),
            'automatic_deaths': len(auto_deaths),
            'matched_deaths': matches,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'quality_score': f1_score,  # Overall quality based on F1
            'unmatched_manual': len(manual_deaths) - matches,
            'unmatched_automatic': len(auto_deaths) - matches
        }
    
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
    
    def plot_health_timeline(self, title="Health Over Time", show_events=True):
        """
        Plot health timeline with events highlighted.
        Works with automatic health data.
        """
        if self.data_type != "automatic_health":
            raise ValueError("This method only works with automatic health data")
            
        df = self.clean_health_data()
        
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
            marker=dict(size=3)
        ))
        
        # Plot max health
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['max_health'],
            mode='lines',
            name='Max Health',
            line=dict(color='blue', width=1, dash='dash')
        ))
        
        if show_events:
            # Extract and highlight events
            events = self.extract_events_from_health()
            
            # Highlight deaths
            deaths = events[events['event_type'] == 'death']
            if not deaths.empty:
                fig.add_trace(go.Scatter(
                    x=deaths['time_seconds'],
                    y=[0] * len(deaths),
                    mode='markers',
                    name='Deaths',
                    marker=dict(color='black', size=10, symbol='x')
                ))
            
            # Highlight major damage
            damage = events[events['event_type'] == 'damage']
            if not damage.empty:
                fig.add_trace(go.Scatter(
                    x=damage['time_seconds'],
                    y=damage['health_after'],
                    mode='markers',
                    name='Major Damage',
                    marker=dict(color='orange', size=8, symbol='triangle-down')
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time (seconds)",
            yaxis_title="Health",
            hovermode='x unified'
        )
        
        fig.show()
    
    def plot_action_timeline(self, title="Action Timeline"):
        """
        Plot action timeline for manual annotations.
        Works with manual action data.
        """
        if self.data_type != "manual_actions":
            raise ValueError("This method only works with manual action data")
            
        df = self.convert_timestamp_to_seconds()
        
        # Create a categorical plot
        fig = go.Figure()
        
        # Get unique actions and assign colors
        unique_actions = df['action'].unique()
        
        # Create a comprehensive color palette that covers all actions
        all_colors = (
            px.colors.qualitative.Set1 + 
            px.colors.qualitative.Set2 + 
            px.colors.qualitative.Set3 + 
            px.colors.qualitative.Pastel1 + 
            px.colors.qualitative.Pastel2
        )
        
        # Ensure we have enough colors (cycle through if needed)
        colors = [all_colors[i % len(all_colors)] for i in range(len(unique_actions))]
        color_map = dict(zip(unique_actions, colors))
        
        # Plot each action as a point
        for action in unique_actions:
            action_data = df[df['action'] == action]
            fig.add_trace(go.Scatter(
                x=action_data['time_seconds'],
                y=[action] * len(action_data),
                mode='markers',
                name=action,
                marker=dict(
                    color=color_map[action],
                    size=8,
                    symbol='circle'
                ),
                text=[f"{action} at {t:.1f}s" for t in action_data['time_seconds']],
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time (seconds)",
            yaxis_title="Action",
            hovermode='closest',
            height=max(400, len(unique_actions) * 30)
        )
        
        fig.show()
    
    def plot_comparison(self, other_df, title="Annotation Comparison"):
        """
        Plot comparison between manual and automatic annotations.
        
        Parameters:
        -----------
        other_df : pd.DataFrame
            The other DataFrame to compare with (with Video accessor)
        title : str
            Plot title
        """
        from plotly.subplots import make_subplots
        
        # Determine which is which
        if self.data_type == "manual_actions" and hasattr(other_df, 'Video') and other_df.Video.data_type == "automatic_health":
            manual_df = self._obj
            auto_df = other_df
        elif self.data_type == "automatic_health" and hasattr(other_df, 'Video') and other_df.Video.data_type == "manual_actions":
            auto_df = self._obj
            manual_df = other_df
        else:
            raise ValueError("One DataFrame must be manual actions and the other automatic health data")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Automatic Health Detection', 'Manual Action Annotations'],
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )
        
        # Plot automatic health data
        auto_cleaned = auto_df.Video.clean_health_data()
        if not auto_cleaned.empty:
            # Current health
            fig.add_trace(
                go.Scatter(
                    x=auto_cleaned['time'],
                    y=auto_cleaned['current_health'],
                    mode='lines+markers',
                    name='Current Health',
                    line=dict(color='red', width=2),
                    marker=dict(size=3)
                ),
                row=1, col=1
            )
            
            # Max health  
            fig.add_trace(
                go.Scatter(
                    x=auto_cleaned['time'],
                    y=auto_cleaned['max_health'],
                    mode='lines',
                    name='Max Health',
                    line=dict(color='blue', width=1, dash='dash')
                ),
                row=1, col=1
            )
            
            # Deaths from automatic
            auto_events = auto_df.Video.extract_events_from_health()
            deaths = auto_events[auto_events['event_type'] == 'death']
            if not deaths.empty:
                fig.add_trace(
                    go.Scatter(
                        x=deaths['time_seconds'],
                        y=[0] * len(deaths),
                        mode='markers',
                        name='Auto Deaths',
                        marker=dict(color='black', size=10, symbol='x')
                    ),
                    row=1, col=1
                )
        
        # Plot manual actions
        manual_time = manual_df.Video.convert_timestamp_to_seconds()
        
        # Focus on key actions for comparison
        key_actions = ['death', 'kill', 'damage', 'heal']
        action_y_positions = {action: i for i, action in enumerate(key_actions)}
        
        for action in key_actions:
            action_data = manual_time[manual_time['action'] == action]
            if not action_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=action_data['time_seconds'],
                        y=[action] * len(action_data),
                        mode='markers',
                        name=f'Manual {action}',
                        marker=dict(size=8),
                        showlegend=True
                    ),
                    row=2, col=1
                )
        
        # Add vertical lines for deaths to show correlation
        manual_deaths = manual_time[manual_time['action'] == 'death']['time_seconds'].values
        for death_time in manual_deaths[:10]:  # Limit to first 10 to avoid clutter
            fig.add_vline(
                x=death_time,
                line=dict(color='gray', width=1, dash='dot'),
                opacity=0.5
            )
        
        fig.update_layout(
            title=title,
            height=800,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Health", row=1, col=1)
        fig.update_yaxes(title_text="Action Type", row=2, col=1)
        
        fig.show()

    def plot_timeline(self):
        """
        Legacy method - plots a timeline of actions (for backward compatibility).
        Use plot_action_timeline() for new code.
        """
        if self.data_type == "manual_actions":
            self.plot_action_timeline()
        else:
            self.plot_health_timeline()


# %%
# Example usage code for testing (commented out for module import)
"""
# Load automatic health data
auto_df = pd.read_csv("annotated_video_2_20250508132346.csv")
auto_info = auto_df.Video.get_data_info()
print("Automatic data info:", auto_info)

# Load manual action data  
manual_df = pd.read_csv("video_annotation_manual_20250508132346.csv")
manual_info = manual_df.Video.get_data_info()
print("Manual data info:", manual_info)

# Compare annotations
comparison = VideoAnalyzer.compare_annotations(manual_df, auto_df)
print("Comparison results:", comparison)

# Plot comparison
manual_df.Video.plot_comparison(auto_df, "Manual vs Automatic Annotation Comparison")
"""
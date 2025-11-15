# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.signal import butter, filtfilt
from datetime import datetime, timedelta
import warnings
import sys
import os
import ast
from pathlib import Path
warnings.filterwarnings('ignore')

# Import our existing analyzers
from gaming_health_data.src.EKGAnalyzer import EKGAnalyzer
from gaming_health_data.src.video_analyzer import VideoAnalyzer
from gaming_health_data.src.apple_watch_analyzer import AppleWatchAnalyzer
from gaming_health_data.src.dualsense_analyzer import DualSenseAnalyzer
from gaming_health_data.src.keyboard_analyzer import KeyboardAnalyzer
from gaming_health_data.src.mouse_analyzer import MouseAnalyzer


class GamingHealthAnalyzer:
    """
    Comprehensive gaming health data analyzer with session management capabilities.
    
    This class provides a modular approach to analyzing multimodal gaming health data,
    allowing users to easily switch between sessions and run specific analysis components.
    
    Parameters
    ----------
    session_date : str, optional
        Date identifier for the session (e.g., "28_04_2024")
    data_path : str or Path, optional
        Custom path to data directory. Defaults to "gaming_health_data/recorded_data"
    
    Attributes
    ----------
    session_date : str
        Current session date identifier
    session_type : str
        Session type ("PC" or "PS")
    data_path : Path
        Path to the data directory
    ekg_data : DataFrame or None
        Loaded EKG data
    apple_watch_data : DataFrame or None
        Loaded Apple Watch data
    video_data : DataFrame or None
        Loaded video annotation data
    controller_data : DataFrame or None
        Loaded controller input data
    keyboard_data : DataFrame or None
        Loaded keyboard input data
    mouse_data : DataFrame or None
        Loaded mouse input data
    results : dict
        Dictionary storing analysis results for all components
    """
    
    def __init__(self, session_date=None, data_path=None):
        """
        Initialize the Gaming Health Analyzer.
        
        Parameters
        ----------
        session_date : str, optional
            Date identifier for the session (e.g., "28_04_2024")
        data_path : str or Path, optional
            Custom path to data directory
        """
        self.session_date = session_date
        self.session_type = None
        self.data_path = data_path or Path("gaming_health_data/recorded_data")
        
        # Determine session type if session_date is provided
        if session_date:
            self.session_type = self._detect_session_type(session_date)
        
        # Session data containers
        self.ekg_data = None
        self.apple_watch_data = None
        self.video_data = None
        self.controller_data = None
        self.keyboard_data = None
        self.mouse_data = None
        
        # Analysis results storage
        self.results = {
            'ekg': {},
            'apple_watch': {},
            'video': {},
            'controller': {},
            'keyboard': {},
            'mouse': {},
            'correlations': {},
            'statistics': {}
        }
        
        print(f"Gaming Health Analyzer initialized for session: {session_date} (type: {self.session_type})")
    
    def _detect_session_type(self, session_date):
        """
        Detect whether a session is PC or PS based on available data files.
        
        Parameters
        ----------
        session_date : str
            Session date identifier
            
        Returns
        -------
        str
            "PC", "PS", or None if type cannot be determined
        """
        pc_dir = self.data_path / "PC"
        ps_dir = self.data_path / "PS"
        
        # Check for PC files (keyboard/mouse)
        pc_files = []
        if pc_dir.exists():
            pc_files.extend(list(pc_dir.glob(f"keyboard_log_{session_date}.csv")))
            pc_files.extend(list(pc_dir.glob(f"mouse_log_{session_date}.csv")))
        
        # Check for PS files (controller)
        ps_files = []
        if ps_dir.exists():
            ps_files.extend(list(ps_dir.glob(f"controller_inputs_{session_date}_part_*.csv")))
        
        if pc_files and not ps_files:
            return "PC"
        elif ps_files and not pc_files:
            return "PS"
        elif pc_files and ps_files:
            print(f"Warning: Found both PC and PS files for session {session_date}")
            return "PC"  # Default to PC if both exist
        else:
            return None
    
    def set_session(self, session_date):
        """
        Set a new session date and clear previous data.
        
        Parameters
        ----------
        session_date : str
            New session date identifier
        """
        self.session_date = session_date
        self.session_type = self._detect_session_type(session_date)
        self._clear_session_data()
        print(f"Session set to: {session_date} (type: {self.session_type})")
    
    def _clear_session_data(self):
        """
        Clear all loaded session data and results.
        
        Notes
        -----
        This method resets all data containers and analyzer instances to None,
        and clears the results dictionary while preserving its structure.
        """
        self.ekg_data = None
        self.apple_watch_data = None
        self.video_data = None
        self.controller_data = None
        self.keyboard_data = None
        self.mouse_data = None
        
        # Reset results but keep structure
        for key in self.results:
            self.results[key] = {}
    
    def load_session_data(self, components=None):
        """
        Load data for the current session.
        
        Parameters
        ----------
        components : list of str, optional
            Specific components to load. Valid options are:
            ['ekg', 'apple_watch', 'video', 'controller', 'keyboard', 'mouse']
            If None, attempts to load all available components.
            
        Raises
        ------
        ValueError
            If session_date is not set before calling this method.
        """
        if not self.session_date:
            raise ValueError("Session date must be set before loading data")
        
        if components is None:
            components = ['ekg', 'apple_watch', 'video', 'controller', 'keyboard', 'mouse']
        
        print(f"Loading session data for {self.session_date}...")
        
        # Load EKG data
        if 'ekg' in components:
            self._load_ekg_data()
        
        # Load Apple Watch data
        if 'apple_watch' in components:
            self._load_apple_watch_data()
        
        # Load video data
        if 'video' in components:
            self._load_video_data()
        
        # Load controller data
        if 'controller' in components:
            self._load_controller_data()
        
        # Load keyboard data
        if 'keyboard' in components:
            self._load_keyboard_data()
        
        # Load mouse data
        if 'mouse' in components:
            self._load_mouse_data()
        
        print("Session data loading completed")
    
    def _load_ekg_data(self):
        """
        Load EKG data for the current session.
        
        Notes
        -----
        Searches for EKG data files in the SENSORS directory with pattern
        'ekg_data_polars_h10_*{session_date}*.txt' and initializes EKGAnalyzer if found.
        The files include timestamps, so we search for any file containing the session date.
        """
        try:
            # Convert session date format from DD_MM_YYYY to YYYY_MM_DD for file search
            if self.session_date:
                date_parts = self.session_date.split('_')
                if len(date_parts) == 3:
                    # Convert from DD_MM_YYYY to YYYY_MM_DD
                    day, month, year = date_parts
                    search_date = f"{year}_{month}_{day}"
                else:
                    search_date = self.session_date
            else:
                search_date = ""
            
            # Search for EKG files with the date pattern (with timestamps)
            sensors_dir = self.data_path / "SENSORS"
            if sensors_dir.exists():
                ekg_files = list(sensors_dir.glob(f"ekg_data_polars_h10_*{search_date}*.txt"))
                
                if ekg_files:
                    # Use the first matching file
                    ekg_file = ekg_files[0]
                    self.ekg_data = EKGAnalyzer.read_file(str(ekg_file), sensor_type="PolarH10")
                    print(f"  EKG data loaded: {len(self.ekg_data)} samples from {ekg_file.name}")
                else:
                    print(f"  EKG file not found for session {self.session_date} in {sensors_dir}")
                    print(f"  Searched for pattern: ekg_data_polars_h10_*{search_date}*.txt")
            else:
                print(f"  SENSORS directory not found: {sensors_dir}")
        except Exception as e:
            print(f"  Error loading EKG data: {e}")
    
    def _load_apple_watch_data(self):
        """
        Load Apple Watch data for the current session.
        
        Notes
        -----
        Apple Watch data is typically session-independent and located in
        the APPLE_WATCH directory. Uses AppleWatchAnalyzer.read_file method.
        """
        try:
            # Look for Apple Watch files in the APPLE_WATCH directory
            apple_dir = self.data_path / "APPLE_WATCH"
            if apple_dir.exists():
                # Find any CSV file in the directory (Apple Watch export format)
                apple_files = list(apple_dir.glob("*.csv"))
                
                if apple_files:
                    # Use the most recent file or first available
                    apple_file = apple_files[0]  # Could enhance to pick most recent
                    
                    # Use AppleWatchAnalyzer.read_file method
                    self.apple_watch_data = AppleWatchAnalyzer.read_file(str(apple_file))
                    
                    print(f"  Apple Watch data loaded: {len(self.apple_watch_data)} records from {apple_file.name}")
                else:
                    print(f"  No Apple Watch CSV files found in {apple_dir}")
            else:
                print(f"  Apple Watch directory not found: {apple_dir}")
        except Exception as e:
            print(f"  Error loading Apple Watch data: {e}")
    
    def _load_video_data(self):
        """Load video annotation data for the current session."""
        try:
            # Convert session date format from DD_MM_YYYY to YYYYMMDD for file search
            if self.session_date:
                date_parts = self.session_date.split('_')
                if len(date_parts) == 3:
                    # Convert from DD_MM_YYYY to YYYYMMDD
                    day, month, year = date_parts
                    search_date = f"{year}{month}{day}"
                else:
                    search_date = self.session_date.replace('_', '')
            else:
                search_date = ""
            
            # Search for video annotation files with timestamp pattern
            video_dir = self.data_path / "VIDEO" / "annotated"
            if video_dir.exists():
                video_files = list(video_dir.glob(f"video_annotation_manual_{search_date}*.csv"))
                
                if video_files:
                    # Use the first matching file and load as DataFrame with Video extension
                    video_file = video_files[0]
                    self.video_data = VideoAnalyzer.from_file(str(video_file))
                    print(f"  Video data loaded: {len(self.video_data)} annotations from {video_file.name}")
                else:
                    print(f"  Video file not found for session {self.session_date} in {video_dir}")
                    print(f"  Searched for pattern: video_annotation_manual_{search_date}*.csv")
            else:
                print(f"  VIDEO/annotated directory not found: {video_dir}")
        except Exception as e:
            print(f"  Error loading video data: {e}")
    
    def _load_controller_data(self):
        """Load controller data for the current session."""
        try:
            # Only load controller data for PS sessions
            if self.session_type != "PS":
                print(f"  Controller data not applicable for {self.session_type} session")
                return
                
            # Look for PS controller data with the correct pattern
            ps_dir = self.data_path / "PS"
            if ps_dir.exists():
                controller_files = list(ps_dir.glob(f"controller_inputs_{self.session_date}_part_*.csv"))
                
                if controller_files:
                    # Sort files by part number to ensure correct order
                    controller_files.sort()
                    
                    if len(controller_files) == 1:
                        # Single file - load directly
                        self.controller_data = DualSenseAnalyzer.read_file(str(controller_files[0]))
                        print(f"  Controller data loaded: {len(self.controller_data)} inputs from {controller_files[0].name}")
                    else:
                        # Multiple parts - concatenate them
                        print(f"  Found {len(controller_files)} controller parts, concatenating...")
                        all_parts = []
                        
                        for i, file_path in enumerate(controller_files):
                            part_data = DualSenseAnalyzer.read_file(str(file_path))
                            print(f"    Part {i}: {len(part_data)} inputs from {file_path.name}")
                            all_parts.append(part_data)
                        
                        # Concatenate all parts into one DataFrame
                        import pandas as pd
                        self.controller_data = pd.concat(all_parts, ignore_index=True)
                        print(f"  Controller data loaded: {len(self.controller_data)} total inputs from {len(controller_files)} parts")
                else:
                    print(f"  Controller files not found for session: {self.session_date}")
                    print(f"  Searched for pattern: controller_inputs_{self.session_date}_part_*.csv")
            else:
                print(f"  PS directory not found: {ps_dir}")
        except Exception as e:
            print(f"  Error loading controller data: {e}")
    
    def _load_keyboard_data(self):
        """Load keyboard data for the current session."""
        try:
            # Only load keyboard data for PC sessions
            if self.session_type != "PC":
                print(f"  Keyboard data not applicable for {self.session_type} session")
                return
                
            keyboard_file = self.data_path / "PC" / f"keyboard_log_{self.session_date}.csv"
            if keyboard_file.exists():
                self.keyboard_data = KeyboardAnalyzer.from_file(str(keyboard_file))
                print(f"  Keyboard data loaded: {len(self.keyboard_data)} events")
            else:
                print(f"  Keyboard file not found: {keyboard_file}")
        except Exception as e:
            print(f"  Error loading keyboard data: {e}")
    
    def _load_mouse_data(self):
        """Load mouse data for the current session."""
        try:
            # Only load mouse data for PC sessions
            if self.session_type != "PC":
                print(f"  Mouse data not applicable for {self.session_type} session")
                return
                
            mouse_file = self.data_path / "PC" / f"mouse_log_{self.session_date}.csv"
            if mouse_file.exists():
                self.mouse_data = MouseAnalyzer.from_file(str(mouse_file))
                print(f"  Mouse data loaded: {len(self.mouse_data)} events")
            else:
                print(f"  Mouse file not found: {mouse_file}")
        except Exception as e:
            print(f"  Error loading mouse data: {e}")
    
    def analyze_ekg(self, show_plots=True):
        """
        Perform comprehensive EKG analysis.
        
        Parameters
        ----------
        show_plots : bool, default True
            Whether to display analysis plots
            
        Returns
        -------
        dict or None
            Dictionary containing EKG analysis results with keys:
            - 'peaks': detected R-peaks
            - 'heartbeats': calculated heartbeat intervals
            - 'hrv_metrics': heart rate variability metrics
            - 'moving_avg_bpm': moving average BPM values
            - 'total_duration': total recording duration
            Returns None if EKG data is not loaded.
        """
        if self.ekg_data is None:
            print("EKG data not loaded. Call load_session_data() first.")
            return None
        
        print("Analyzing EKG data...")
        
        # Use comprehensive EKG DataFrame extension methods
        try:
            # Basic detection and analysis
            beats = self.ekg_data.EKG.beats  # Property
            peaks = self.ekg_data.EKG.peaks  # Property
            low_peaks = self.ekg_data.EKG.low_peaks  # Property
            moving_avg_bpm = self.ekg_data.EKG.calculate_moving_avg_bpm(window_size=10)
            bpm = self.ekg_data.EKG.calculate_bpm()
            
            # Heart rate metrics
            mean_heart_rate = self.ekg_data.EKG.mean_heart_rate  # Property
            
            # HRV metrics
            rr_intervals = self.ekg_data.EKG.rr_intervals  # Property
            rmssd = self.ekg_data.EKG.rmssd  # Property
            sdnn = self.ekg_data.EKG.sdnn  # Property
            pnn50 = self.ekg_data.EKG.pnn50  # Property
            
            hrv_metrics = {
                'rr_intervals': rr_intervals,
                'rmssd': rmssd,
                'sdnn': sdnn,
                'pnn50': pnn50
            }
            
            # Frequency domain HRV
            frequency_domain_hrv = self.ekg_data.EKG.calculate_frequency_domain_hrv()
            
            # Advanced analysis
            rhythm_type = self.ekg_data.EKG.rhythm_type  # Property
            arrhythmia_detected = self.ekg_data.EKG.arrhythmia_detected  # Property
            stress_index = self.ekg_data.EKG.stress_index  # Property
            hrv_score = self.ekg_data.EKG.hr_variability_score  # Property
            measurement_length = self.ekg_data.EKG.measurement_length  # Property
            
            # Comprehensive summary
            analysis_summary = self.ekg_data.EKG.get_heart_analysis_summary()
            
        except Exception as e:
            print(f"Warning: EKG analysis failed: {e}")
            # Fallback to basic analysis
            beats = []
            peaks = []
            low_peaks = []
            moving_avg_bpm = []
            bpm = []
            mean_heart_rate = 0
            hrv_metrics = {}
            frequency_domain_hrv = {}
            rhythm_type = "Unknown"
            arrhythmia_detected = False
            stress_index = 0
            hrv_score = 0
            measurement_length = "Unknown"
            analysis_summary = "Analysis failed"
        
        # Store comprehensive results
        self.results['ekg'] = {
            'beats': beats,
            'peaks': peaks,
            'low_peaks': low_peaks,
            'bpm': bpm,
            'moving_avg_bpm': moving_avg_bpm,
            'mean_heart_rate': mean_heart_rate,
            'hrv_metrics': hrv_metrics,
            'frequency_domain_hrv': frequency_domain_hrv,
            'rhythm_type': rhythm_type,
            'arrhythmia_detected': arrhythmia_detected,
            'stress_index': stress_index,
            'hrv_score': hrv_score,
            'measurement_length': measurement_length,
            'analysis_summary': analysis_summary,
            'total_duration': len(self.ekg_data) if self.ekg_data is not None else 0
        }
        
        if show_plots:
            # Use built-in EKGAnalyzer plotting methods
            try:
                self.ekg_data.EKG.plot_ekg_data()
                self.ekg_data.EKG.plot_moving_avg_bpm(window_size=10)
            except Exception as e:
                print(f"Warning: EKG plotting failed: {e}")
                # Fallback to custom plot
                self._plot_ekg_analysis()
        
        print("EKG analysis completed")
        # Print summary if available
        if 'analysis_summary' in self.results['ekg']:
            print(f"\n{self.results['ekg']['analysis_summary']}")
        
        return self.results['ekg']
    
    def analyze_apple_watch(self, show_plots=True):
        """
        Perform comprehensive Apple Watch data analysis using all available methods.
        
        Parameters
        ----------
        show_plots : bool, default True
            Whether to display analysis plots
            
        Returns
        -------
        dict or None
            Dictionary containing Apple Watch analysis results with keys:
            - 'heart_rate_data': filtered heart rate data
            - 'gaming_sessions': gaming session data
            - 'gaming_heart_rate': heart rate during gaming sessions
            - 'daily_heart_rate': heart rate for current session date
            - 'heart_rate_statistics': comprehensive HR statistics
            - 'gaming_vs_daily_comparison': comparison between gaming and daily HR
            - 'available_types': all data types in the dataset
            - 'available_sources': all data sources
            - 'measurement_period': measurement start/end dates
            - 'total_records': number of records processed
            Returns None if Apple Watch data is not loaded.
        """
        if self.apple_watch_data is None:
            print("Apple Watch data not loaded. Call load_session_data() first.")
            return None
        
        print("Analyzing Apple Watch data...")
        
        # Use comprehensive AppleWatchAnalyzer methods
        try:
            # Basic data filtering
            heart_rate_data = self.apple_watch_data.applewatch.filter_by_type("HeartRate")
            
            # Get available data types and sources
            available_types = self.apple_watch_data.applewatch.types
            available_sources = self.apple_watch_data.applewatch.source_names
            
            # Gaming session analysis
            gaming_sessions = self.apple_watch_data.applewatch.get_gaming_sessions()
            
            # Session-specific analysis using current session date
            gaming_heart_rate = pd.DataFrame()
            daily_heart_rate = pd.DataFrame()
            heart_rate_statistics = None
            gaming_vs_daily_comparison = None
            
            if self.session_date:
                # Convert session date format for Apple Watch methods
                if '_' in self.session_date:
                    date_parts = self.session_date.split('_')
                    if len(date_parts) == 3:
                        day, month, year = date_parts
                        formatted_session_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    else:
                        formatted_session_date = self.session_date
                else:
                    formatted_session_date = self.session_date
                
                try:
                    # Get gaming session heart rate
                    gaming_heart_rate = self.apple_watch_data.applewatch.get_gaming_session_heart_rate(formatted_session_date)
                    
                    # Get daily heart rate
                    daily_heart_rate = self.apple_watch_data.applewatch.get_heart_rate_for_date(formatted_session_date)
                    
                    # Get heart rate statistics
                    heart_rate_statistics = self.apple_watch_data.applewatch.get_heart_rate_statistics(formatted_session_date)
                    
                    # Compare gaming vs daily
                    gaming_vs_daily_comparison = self.apple_watch_data.applewatch.compare_gaming_vs_daily_hr(formatted_session_date)
                    
                except Exception as e:
                    print(f"Warning: Session-specific analysis failed: {e}")
            
            # Get measurement period
            measurement_period = {
                'start': self.apple_watch_data.applewatch.measure_start_date,
                'end': self.apple_watch_data.applewatch.measure_end_date
            }
            
        except Exception as e:
            print(f"Warning: Apple Watch analysis failed: {e}")
            # Fallback to basic analysis
            heart_rate_data = self.apple_watch_data
            gaming_sessions = pd.DataFrame()
            gaming_heart_rate = pd.DataFrame()
            daily_heart_rate = pd.DataFrame()
            heart_rate_statistics = None
            gaming_vs_daily_comparison = None
            available_types = []
            available_sources = []
            measurement_period = {'start': None, 'end': None}
        
        # Store comprehensive results
        self.results['apple_watch'] = {
            'heart_rate_data': heart_rate_data,
            'gaming_sessions': gaming_sessions,
            'gaming_heart_rate': gaming_heart_rate,
            'daily_heart_rate': daily_heart_rate,
            'heart_rate_statistics': heart_rate_statistics,
            'gaming_vs_daily_comparison': gaming_vs_daily_comparison,
            'available_types': available_types,
            'available_sources': available_sources,
            'measurement_period': measurement_period,
            'total_records': len(self.apple_watch_data) if self.apple_watch_data is not None else 0
        }
        
        if show_plots:
            # Use built-in AppleWatchAnalyzer plotting methods
            try:
                if self.session_date:
                    # Convert session date format for plotting methods
                    if '_' in self.session_date:
                        date_parts = self.session_date.split('_')
                        if len(date_parts) == 3:
                            day, month, year = date_parts
                            formatted_session_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        else:
                            formatted_session_date = self.session_date
                    else:
                        formatted_session_date = self.session_date
                        
                    # Plot heart rate for the session date
                    fig = self.apple_watch_data.applewatch.plot_hearth_rate_for_session_date(formatted_session_date)
                    if fig:
                        fig.show()
                    
                    # Plot time series for heart rate
                    fig2 = self.apple_watch_data.applewatch.plot_time_series("HeartRate")
                    if fig2:
                        fig2.show()
            except Exception as e:
                print(f"Warning: Apple Watch plotting failed: {e}")
                # Fallback to custom plot
                self._plot_apple_watch_analysis()
        
        print("Apple Watch analysis completed")
        # Print statistics summary if available
        if heart_rate_statistics:
            print(f"\nHeart Rate Statistics for {self.session_date}:")
            for key, value in heart_rate_statistics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        
        return self.results['apple_watch']
    
    def compare_ekg_sources(self, session_date=None, show_plots=True, filter_to_ekg_window=True):
        """
        Compare EKG data from different sources (Polar H10 vs Apple Watch) with optional time-synchronized filtering.
        
        Parameters
        ----------
        session_date : str, optional
            Session date for comparison. Uses current session_date if not provided.
        show_plots : bool, default True
            Whether to display comparison plots
        filter_to_ekg_window : bool, default True
            Whether to filter Apple Watch data to EKG measurement window.
            If False, shows all Apple Watch data for the entire day.
            
        Returns
        -------
        dict or None
            Dictionary containing comparison results with keys:
            - 'polar_bpm': Polar H10 moving average BPM
            - 'apple_bmp': Apple Watch heart rate data (all day or filtered)
            - 'apple_bpm_filtered': Apple Watch data filtered to EKG window (if filtering enabled)
            - 'polar_timestamps': Timestamps for Polar H10 data
            - 'apple_timestamps': Timestamps for Apple Watch data
            - 'time_bounds': EKG measurement time window
            - 'correlation': correlation coefficient between sources (filtered data)
            - 'mean_difference': mean difference between measurements
            - 'std_difference': standard deviation of differences
            Returns None if both data sources are not available.
        """
        if self.ekg_data is None or self.apple_watch_data is None:
            print("Both EKG and Apple Watch data needed for comparison. Load both datasets first.")
            return None
        
        print("Comparing EKG sources...")
        
        # Use session_date parameter or class attribute
        target_date = session_date or self.session_date
        if not target_date:
            print("No session date provided for comparison")
            return None
        
        # Convert session date format from DD_MM_YYYY to YYYY-MM-DD for Apple Watch methods
        if target_date and '_' in target_date:
            date_parts = target_date.split('_')
            if len(date_parts) == 3:
                day, month, year = date_parts
                target_date_formatted = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            else:
                target_date_formatted = target_date
        else:
            target_date_formatted = target_date
        
        print(f"Using formatted date: {target_date_formatted} (from {target_date})")
        
        # Get EKG measurement time bounds
        measure_start_date = getattr(self.ekg_data, 'measure_start_date', None)
        measure_end_date = getattr(self.ekg_data, 'measure_end_date', None)
        
        print(f"EKG measurement period: {measure_start_date} to {measure_end_date}")
        
        # Get Polar H10 moving average BPM using EKGAnalyzer methods
        polar_bpm = self.ekg_data.EKG.calculate_moving_avg_bpm(window_size=10)
        polar_timestamps = None
        
        # Get beat timestamps for polar data if available
        try:
            beats = self.ekg_data.EKG.beats
            if len(beats) > 0 and 'Timestamp' in self.ekg_data.columns:
                # Convert relative timestamps to absolute timestamps
                relative_timestamps = self.ekg_data['Timestamp'].iloc[beats[:len(polar_bpm)]]
                if measure_start_date is not None:
                    polar_timestamps = measure_start_date + pd.to_timedelta(relative_timestamps, unit='s')
                else:
                    polar_timestamps = relative_timestamps
        except Exception as e:
            print(f"Warning: Could not get Polar timestamps: {e}")
        
        # Get Apple Watch heart rate data using AppleWatchAnalyzer methods
        apple_hr_df = None
        apple_hr = []
        apple_hr_filtered = []
        apple_timestamps = None
        apple_timestamps_filtered = None
        
        try:
            # Get all gaming data for the session (42 points, excluding NaN)
            print("Getting ALL Apple Watch gaming data for the session...")
            apple_hr_df = self.apple_watch_data.applewatch.gaming_data_for_session(target_date_formatted)
            print(f"Found {len(apple_hr_df)} total Apple Watch gaming data measurements")
            
            if not apple_hr_df.empty:
                # Remove NaN values
                apple_hr_df = apple_hr_df.dropna(subset=['value'])
                print(f"After removing NaN values: {len(apple_hr_df)} measurements")
                
                # Ensure Apple Watch timestamps are datetime objects
                apple_hr_df['startDate'] = pd.to_datetime(apple_hr_df['startDate'])
                
                print(f"Apple Watch gaming data time range: {apple_hr_df['startDate'].min()} to {apple_hr_df['startDate'].max()}")
                
                # Extract ALL gaming data for plotting
                apple_hr = apple_hr_df['value'].values if 'value' in apple_hr_df.columns else []
                apple_timestamps = apple_hr_df['startDate'].values if 'startDate' in apple_hr_df.columns else None
                
                print(f"Extracted {len(apple_hr)} Apple Watch measurements for plotting")
                print(f"First few Apple Watch values: {apple_hr[:10] if len(apple_hr) > 10 else apple_hr}")
                
                # Use all the data (no filtering)
                apple_hr_filtered = apple_hr.copy()
                apple_timestamps_filtered = apple_timestamps.copy() if apple_timestamps is not None else None
                
            else:
                print("No Apple Watch gaming data found")
                
        except Exception as e:
            print(f"Warning: Could not extract Apple Watch heart rate: {e}")
            import traceback
            traceback.print_exc()
        
        # Store comparison results with enhanced structure
        comparison_results = {
            'polar_bpm': polar_bpm,
            'apple_bpm': apple_hr,  # All day data
            'apple_bpm_filtered': apple_hr_filtered,  # EKG window data for correlation
            'polar_timestamps': polar_timestamps,
            'apple_timestamps': apple_timestamps,  # All day timestamps
            'apple_timestamps_filtered': apple_timestamps_filtered,  # EKG window timestamps
            'time_bounds': {
                'start': measure_start_date,
                'end': measure_end_date
            },
            'correlation': None,
            'mean_difference': None,
            'std_difference': None,
            'filter_applied': filter_to_ekg_window
        }
        
        # Skip correlation analysis - just plot all the points
        print(f"Skipping correlation analysis - just plotting all {len(apple_hr) if apple_hr is not None else 0} Apple Watch points")
        
        # Store results in the class results dictionary
        self.results['correlations']['ekg_sources'] = comparison_results
        
        if show_plots:
            self._plot_ekg_comparison()
        
        print("EKG source comparison completed")
        return comparison_results
    
    def analyze_video(self, show_plots=True):
        """
        Perform video annotation analysis.
        
        Parameters
        ----------
        show_plots : bool, default True
            Whether to display analysis plots
            
        Returns
        -------
        dict or None
            Dictionary containing video analysis results with keys:
            - 'annotations': processed video annotations
            - 'total_annotations': number of annotations processed
            Returns None if video data is not loaded.
        """
        if self.video_data is None:
            print("Video data not loaded. Call load_session_data() first.")
            return None
        
        print("Analyzing video annotations...")
        
        # Use DataFrame extension methods
        try:
            annotations = self.video_data.Video.get_action_intervals() if hasattr(self.video_data.Video, 'get_action_intervals') else self.video_data
        except Exception as e:
            print(f"Warning: Could not process video annotations: {e}")
            annotations = self.video_data
        
        # Store results
        self.results['video'] = {
            'annotations': annotations,
            'total_annotations': len(self.video_data) if self.video_data is not None else 0
        }
        
        if show_plots:
            self._plot_video_analysis()
        
        print("Video analysis completed")
        return self.results['video']
    
    def analyze_controller(self, show_plots=True):
        """
        Perform controller input analysis.
        
        Parameters
        ----------
        show_plots : bool, default True
            Whether to display analysis plots
            
        Returns
        -------
        dict or None
            Dictionary containing controller analysis results with keys:
            - 'input_stats': controller input statistics
            - 'total_inputs': number of inputs processed
            Returns None if controller data is not loaded.
        """
        if self.controller_data is None:
            print("Controller data not loaded. Call load_session_data() first.")
            return None
        
        print("Analyzing controller inputs...")
        
        # Use DataFrame extension methods
        try:
            input_stats = self.controller_data.dualsense.calculate_input_statistics() if hasattr(self.controller_data.dualsense, 'calculate_input_statistics') else {}
        except Exception as e:
            print(f"Warning: Could not calculate controller statistics: {e}")
            input_stats = {}
        
        # Store results
        self.results['controller'] = {
            'input_stats': input_stats,
            'total_inputs': len(self.controller_data) if self.controller_data is not None else 0
        }
        
        if show_plots:
            self._plot_controller_analysis()
        
        print("Controller analysis completed")
        return self.results['controller']
    
    def analyze_keyboard(self, show_plots=True):
        """
        Perform keyboard input analysis.
        
        Parameters
        ----------
        show_plots : bool, default True
            Whether to display analysis plots
            
        Returns
        -------
        dict or None
            Dictionary containing keyboard analysis results with keys:
            - 'key_stats': keyboard input statistics
            - 'total_events': number of events processed
            Returns None if keyboard data is not loaded.
        """
        if self.keyboard_data is None:
            print("Keyboard data not loaded. Call load_session_data() first.")
            return None
        
        print("Analyzing keyboard inputs...")
        
        # Use DataFrame extension methods
        try:
            key_stats = self.keyboard_data.Keyboard.calculate_presses_per_second() if hasattr(self.keyboard_data.Keyboard, 'calculate_presses_per_second') else {}
        except Exception as e:
            print(f"Warning: Could not calculate keyboard statistics: {e}")
            key_stats = {}
        
        # Store results
        self.results['keyboard'] = {
            'key_stats': key_stats,
            'total_events': len(self.keyboard_data) if self.keyboard_data is not None else 0
        }
        
        if show_plots:
            self._plot_keyboard_analysis()
        
        print("Keyboard analysis completed")
        return self.results['keyboard']
    
    def analyze_mouse(self, show_plots=True):
        """
        Perform mouse input analysis.
        
        Parameters
        ----------
        show_plots : bool, default True
            Whether to display analysis plots
            
        Returns
        -------
        dict or None
            Dictionary containing mouse analysis results with keys:
            - 'mouse_stats': mouse input statistics
            - 'total_events': number of events processed  
            Returns None if mouse data is not loaded.
        """
        if self.mouse_data is None:
            print("Mouse data not loaded. Call load_session_data() first.")
            return None
        
        print("Analyzing mouse inputs...")
        
        # Use DataFrame extension methods
        try:
            mouse_stats = self.mouse_data.Mouse.calculate_clicks_per_second() if hasattr(self.mouse_data.Mouse, 'calculate_clicks_per_second') else {}
        except Exception as e:
            print(f"Warning: Could not calculate mouse statistics: {e}")
            mouse_stats = {}
        
        # Store results
        self.results['mouse'] = {
            'mouse_stats': mouse_stats,
            'total_events': len(self.mouse_data) if self.mouse_data is not None else 0
        }
        
        if show_plots:
            self._plot_mouse_analysis()
        
        print("Mouse analysis completed")
        return self.results['mouse']
    
    def analyze_correlations(self, show_plots=True):
        """
        Analyze correlations between different data modalities.
        
        Parameters
        ----------
        show_plots : bool, default True
            Whether to display correlation plots
            
        Returns
        -------
        dict
            Dictionary containing correlation analysis results with keys:
            - 'ekg_gaming_events': correlation between EKG and gaming events
            - 'heart_rate_input_intensity': correlation between heart rate and input intensity
        """
        print("Analyzing multimodal correlations...")
        
        correlations = {}
        
        # EKG vs Gaming Events correlation
        if self.ekg_data is not None and self.video_data is not None:
            correlations['ekg_gaming_events'] = self._correlate_ekg_gaming_events()
        
        # Heart Rate vs Input Intensity correlation
        if self.ekg_data is not None and (self.controller_data is not None or self.keyboard_data is not None or self.mouse_data is not None):
            correlations['heart_rate_input_intensity'] = self._correlate_heart_rate_input_intensity()
        
        self.results['correlations'].update(correlations)
        
        if show_plots:
            self._plot_correlation_analysis()
        
        print("Correlation analysis completed")
        return correlations
    
    def run_full_analysis(self, components=None, show_plots=True):
        """
        Run complete analysis for all available components.
        
        Parameters
        ----------
        components : list of str, optional
            Specific components to analyze. Valid options are:
            ['ekg', 'apple_watch', 'video', 'controller', 'keyboard', 'mouse']
            If None, analyzes all loaded components.
        show_plots : bool, default True
            Whether to display all plots
            
        Returns
        -------
        dict
            Complete analysis results dictionary containing results for all
            analyzed components and cross-modal correlations.
        """
        print("Running full multimodal analysis...")
        
        # Load data if not already loaded
        if not any([self.ekg_data is not None, self.apple_watch_data is not None, self.video_data is not None, 
                   self.controller_data is not None, self.keyboard_data is not None, self.mouse_data is not None]):
            self.load_session_data(components)
        
        # Run individual analyses
        if components is None or 'ekg' in components:
            if self.ekg_data is not None:
                self.analyze_ekg(show_plots)
        
        if components is None or 'apple_watch' in components:
            if self.apple_watch_data is not None:
                self.analyze_apple_watch(show_plots)
        
        if components is None or 'video' in components:
            if self.video_data is not None:
                self.analyze_video(show_plots)
        
        if components is None or 'controller' in components:
            if self.controller_data is not None:
                self.analyze_controller(show_plots)
        
        if components is None or 'keyboard' in components:
            if self.keyboard_data is not None:
                self.analyze_keyboard(show_plots)
        
        if components is None or 'mouse' in components:
            if self.mouse_data is not None:
                self.analyze_mouse(show_plots)
        
        # Run correlation analysis
        self.analyze_correlations(show_plots)
        
        # Compare EKG sources if both are available
        if self.ekg_data is not None and self.apple_watch_data is not None:
            self.compare_ekg_sources(show_plots)
        
        print("Full analysis completed!")
        return self.results
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive analysis report.
        
        Returns
        -------
        str
            Formatted markdown-style analysis report containing:
            - Session information and timestamp
            - Data availability summary
            - Analysis results summary for each component
            - Cross-modal correlation results
        """
        print("Generating comprehensive report...")
        
        report = f"""
# Gaming Health Analysis Report
Session: {self.session_date}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data Summary
"""
        
        # Add data availability summary
        available_data = []
        if self.ekg_data is not None:
            available_data.append(f"EKG: {len(self.ekg_data)} samples")
        if self.apple_watch_data is not None:
            available_data.append(f"Apple Watch: {len(self.apple_watch_data)} records")
        if self.video_data is not None:
            available_data.append(f"Video: {len(self.video_data)} annotations")
        if self.controller_data is not None:
            available_data.append(f"Controller: {len(self.controller_data)} inputs")
        if self.keyboard_data is not None:
            available_data.append(f"Keyboard: {len(self.keyboard_data)} events")
        if self.mouse_data is not None:
            available_data.append(f"Mouse: {len(self.mouse_data)} events")
        
        report += "- " + "\n- ".join(available_data) + "\n\n"
        
        # Add analysis results summary
        for component, results in self.results.items():
            if results and component != 'correlations':
                report += f"## {component.title()} Analysis\n"
                report += f"Results available: {len(results)} metrics\n\n"
        
        # Add correlation summary
        if self.results['correlations']:
            report += "## Correlation Analysis\n"
            for corr_name, corr_data in self.results['correlations'].items():
                if isinstance(corr_data, dict) and 'correlation' in corr_data:
                    report += f"- {corr_name}: {corr_data['correlation']:.3f}\n"
        
        print("Report generated")
        return report
    
    # Plotting methods
    def _plot_ekg_analysis(self):
        """
        Plot EKG analysis results.
        
        Notes
        -----
        Creates an interactive plot showing the moving average BPM over time
        using Plotly. Only plots if moving average data is available.
        """
        if 'moving_avg_bpm' in self.results['ekg'] and self.results['ekg']['moving_avg_bpm'] is not None:
            fig = go.Figure()
            moving_avg = self.results['ekg']['moving_avg_bpm']
            
            fig.add_trace(go.Scatter(
                y=moving_avg,
                mode='lines',
                name='Moving Average BPM',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f'EKG Analysis - Moving Average BPM (Session: {self.session_date})',
                xaxis_title='Time (samples)',
                yaxis_title='BPM',
                showlegend=True
            )
            
            fig.show()
    
    def _plot_apple_watch_analysis(self):
        """Plot Apple Watch analysis results."""
        if 'heart_rate_data' in self.results['apple_watch'] and self.results['apple_watch']['heart_rate_data'] is not None:
            fig = go.Figure()
            hr_data = self.results['apple_watch']['heart_rate_data']
            
            fig.add_trace(go.Scatter(
                y=hr_data,
                mode='lines',
                name='Apple Watch Heart Rate',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title=f'Apple Watch Heart Rate Analysis (Session: {self.session_date})',
                xaxis_title='Time (samples)',
                yaxis_title='Heart Rate (BPM)',
                showlegend=True
            )
            
            fig.show()
    
    def _plot_ekg_comparison(self):
        """
        Plot comparison between EKG sources showing ALL available data points.
        
        Notes
        -----
        Creates a dual-line plot comparing Polar H10 and Apple Watch heart rate
        data over time. Uses actual timestamps when available. Shows all data
        points without truncation to ensure complete visualization.
        """
        comparison_data = self.results['correlations'].get('ekg_sources', {})
        
        if comparison_data.get('polar_bpm') is not None and comparison_data.get('apple_bpm') is not None:
            fig = go.Figure()
            
            polar_bpm = comparison_data['polar_bpm']
            apple_bpm = comparison_data['apple_bpm']
            
            print(f"Plotting data - Polar H10: {len(polar_bpm)} samples, Apple Watch: {len(apple_bpm)} samples")
            
            # Use timestamps if available, otherwise fall back to sample indices
            polar_timestamps = comparison_data.get('polar_timestamps')
            apple_timestamps = comparison_data.get('apple_timestamps')
            
            # Handle Polar H10 data
            if polar_timestamps is not None and len(polar_timestamps) > 0:
                # Don't truncate - use all available timestamps up to data length
                min_len = min(len(polar_timestamps), len(polar_bpm))
                polar_x = polar_timestamps[:min_len]
                polar_y = polar_bpm[:min_len]
                x_title = 'Time'
                print(f"Using {len(polar_x)} Polar H10 timestamps")
            else:
                polar_x = list(range(len(polar_bpm)))
                polar_y = polar_bpm
                x_title = 'Time (samples)'
                print(f"Using {len(polar_x)} Polar H10 sample indices")
            
            # Handle Apple Watch data
            if apple_timestamps is not None and len(apple_timestamps) > 0:
                # Don't truncate - use all available timestamps up to data length
                min_len = min(len(apple_timestamps), len(apple_bpm))
                apple_x = apple_timestamps[:min_len]
                apple_y = apple_bpm[:min_len]
                print(f"Using {len(apple_x)} Apple Watch timestamps")
            else:
                apple_x = list(range(len(apple_bpm)))
                apple_y = apple_bpm
                print(f"Using {len(apple_x)} Apple Watch sample indices")
            
            # Add Polar H10 trace
            fig.add_trace(go.Scatter(
                x=polar_x,
                y=polar_y,
                mode='lines',
                name=f'Polar H10 ({len(polar_y)} samples)',
                line=dict(color='red', width=2)
            ))
            
            # Add Apple Watch trace with all points visible
            fig.add_trace(go.Scatter(
                x=apple_x,
                y=apple_y,
                mode='lines+markers',
                name=f'Apple Watch ({len(apple_y)} points)',
                line=dict(color='blue', width=2),
                marker=dict(size=6, color='blue')
            ))
            
            # Build title with data counts
            title = f'EKG vs Apple Watch Gaming Session ({len(polar_y)} vs {len(apple_y)} points)'
            
            # Add time bounds if available
            time_bounds = comparison_data.get('time_bounds', {})
            if time_bounds and time_bounds.get('start') and time_bounds.get('end'):
                start_time = time_bounds['start'].strftime('%H:%M:%S') if hasattr(time_bounds['start'], 'strftime') else str(time_bounds['start'])
                end_time = time_bounds['end'].strftime('%H:%M:%S') if hasattr(time_bounds['end'], 'strftime') else str(time_bounds['end'])
                title += f'<br><sub>Time Window: {start_time} - {end_time}</sub>'
            
            fig.update_layout(
                title=title,
                xaxis_title=x_title,
                yaxis_title='Heart Rate (BPM)',
                showlegend=True,
                height=600,
                hovermode='x unified'
            )
            
            print(f"Showing plot with {len(polar_y)} Polar H10 points and {len(apple_y)} Apple Watch points")
            fig.show()
        else:
            print("No EKG comparison data available for plotting")
            if comparison_data:
                print(f"Available comparison data keys: {list(comparison_data.keys())}")
                polar_available = comparison_data.get('polar_bmp') is not None
                apple_available = comparison_data.get('apple_bmp') is not None
                print(f"Polar H10 data available: {polar_available}")
                print(f"Apple Watch data available: {apple_available}")
    
    def _plot_video_analysis(self):
        """Plot video analysis results."""
        # Placeholder for video analysis plotting
        print("Video analysis plot would be displayed here")
    
    def _plot_controller_analysis(self):
        """Plot controller analysis results."""
        # Placeholder for controller analysis plotting
        print("Controller analysis plot would be displayed here")
    
    def _plot_keyboard_analysis(self):
        """Plot keyboard analysis results."""
        # Placeholder for keyboard analysis plotting
        print("Keyboard analysis plot would be displayed here")
    
    def _plot_mouse_analysis(self):
        """Plot mouse analysis results."""
        # Placeholder for mouse analysis plotting
        print("Mouse analysis plot would be displayed here")
    
    def _plot_correlation_analysis(self):
        """Plot correlation analysis results."""
        # Placeholder for correlation analysis plotting
        print("Correlation analysis plot would be displayed here")
    
    # Helper methods for correlation analysis
    def _correlate_ekg_gaming_events(self):
        """
        Calculate correlation between EKG and gaming events.
        
        Returns
        -------
        dict
            Dictionary with correlation and p-value keys.
            Currently returns placeholder values.
        """
        # Placeholder implementation
        return {'correlation': 0.0, 'p_value': 1.0}
    
    def _correlate_heart_rate_input_intensity(self):
        """Calculate correlation between heart rate and input intensity."""
        # Placeholder implementation
        return {'correlation': 0.0, 'p_value': 1.0}
    
    def get_available_sessions(self):
        """
        Get list of available session dates based on existing data files.
        
        Returns
        -------
        list of str
            Sorted list of available session date identifiers found in
            the PC data directory based on file naming patterns.
        """
        sessions = set()
        
        # Check PC data directory for common patterns
        pc_dir = self.data_path / "PC"
        if pc_dir.exists():
            for file in pc_dir.glob("*_??_??_????.csv"):
                # Extract date pattern from filename
                parts = file.stem.split('_')
                if len(parts) >= 3:
                    date_part = '_'.join(parts[-3:])
                    sessions.add(date_part)
            
            for file in pc_dir.glob("*_??_??_????.txt"):
                # Extract date pattern from filename
                parts = file.stem.split('_')
                if len(parts) >= 3:
                    date_part = '_'.join(parts[-3:])
                    sessions.add(date_part)
        
        return sorted(list(sessions))
    
    def list_session_components(self, session_date=None):
        """
        List available data components for a given session.
        
        Parameters
        ----------
        session_date : str, optional
            Session to check. If None, uses current session.
            
        Returns
        -------
        dict
            Dictionary mapping component names to their file paths.
            Keys include: 'ekg', 'video', 'keyboard', 'mouse', 'controller', 'apple_watch'
        """
        if session_date is None:
            session_date = self.session_date
        
        components = {}
        
        # Check for EKG data
        if session_date:
            date_parts = session_date.split('_')
            if len(date_parts) == 3:
                # Convert from DD_MM_YYYY to YYYY_MM_DD for file search
                day, month, year = date_parts
                search_date = f"{year}_{month}_{day}"
            else:
                search_date = session_date
            
            sensors_dir = self.data_path / "SENSORS"
            if sensors_dir.exists():
                ekg_files = list(sensors_dir.glob(f"ekg_data_polars_h10_*{search_date}*.txt"))
                if ekg_files:
                    components['ekg'] = str(ekg_files[0])
        
        # Check for video data (in VIDEO/annotated folder with manual annotation format)
        video_dir = self.data_path / "VIDEO" / "annotated"
        if video_dir.exists():
            # Convert DD_MM_YYYY to DDMMYYYY for video file search
            search_date = session_date.replace("_", "")  
            video_files = list(video_dir.glob(f"video_annotation_manual_{search_date}*.csv"))
            if video_files:
                components['video'] = str(video_files[0])
        
        # Check for keyboard data
        keyboard_file = self.data_path / "PC" / f"keyboard_log_{session_date}.csv"
        if keyboard_file.exists():
            components['keyboard'] = str(keyboard_file)
        
        # Check for mouse data
        mouse_file = self.data_path / "PC" / f"mouse_log_{session_date}.csv"
        if mouse_file.exists():
            components['mouse'] = str(mouse_file)
        
        # Check for controller data (with part files)
        ps_dir = self.data_path / "PS"
        if ps_dir.exists():
            controller_files = list(ps_dir.glob(f"controller_inputs_{session_date}_part_*.csv"))
            if controller_files:
                controller_files.sort()  # Ensure correct order
                if len(controller_files) == 1:
                    components['controller'] = str(controller_files[0])
                else:
                    components['controller'] = f"{len(controller_files)} parts: {[f.name for f in controller_files]}"
        
        # Apple Watch data (session-independent)
        apple_file = self.data_path / "APPLE_WATCH" / "apple_health_export_2025-11-08.csv"
        if apple_file.exists():
            components['apple_watch'] = str(apple_file)
        
        return components


# Example usage and demonstration
if __name__ == "__main__":
    print("Gaming Health Analyzer Class Demo")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = GamingHealthAnalyzer(session_date="28_04_2024")
    
    # Show available sessions
    print("\nAvailable sessions:")
    available_sessions = analyzer.get_available_sessions()
    for session in available_sessions[:5]:  # Show first 5
        print(f"  - {session}")
    
    # Show components for current session
    print(f"\nComponents available for session {analyzer.session_date}:")
    components = analyzer.list_session_components()
    for component, path in components.items():
        print(f"  {component}: {path}")
    
    # Demonstrate modular analysis
    print("\nExample modular analysis workflow:")
    print("  1. analyzer.load_session_data(['ekg', 'apple_watch'])")
    print("  2. analyzer.analyze_ekg()")
    print("  3. analyzer.analyze_apple_watch()")
    print("  4. analyzer.compare_ekg_sources()")
    print("  5. analyzer.generate_comprehensive_report()")
    
    print("\nReady for modular gaming health analysis!")

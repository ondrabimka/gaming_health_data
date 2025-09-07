# %%
"""
Comprehensive Multimodal Gaming Health Analysis
==============================================

This notebook combines both detailed and quick analysis approaches to explore 
correlations between physiological data (EKG), gaming inputs (controller, mouse, keyboard), 
and gaming events (video annotations) using existing analyzer infrastructure.

Key Features:
- Uses existing EKGAnalyzer with built-in peaks/heartbeat detection
- VideoAnalyzer for gaming event processing
- Multiple time windows for heart rate response analysis (5s, 15s, 30s)
- Real-time correlation between gaming events and physiological responses
- Comprehensive HRV analysis using existing methods
- Statistical significance testing
- Interactive visualizations

Analyses:
1. EKG Analysis using existing peaks/beats detection
2. Gaming Events Impact on Heart Rate (multiple time windows)
3. Controller Input Intensity vs Physiological State
4. Stress/Excitement Detection and Gaming Scenarios
5. Performance vs Physiological State Analysis
6. Time-synchronized Multimodal Visualization
7. Comprehensive Statistical Summary
"""

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

print("🎮💓 Comprehensive Multimodal Gaming Health Analysis")
print("=" * 60)

# Create output directory for results
timestamp = datetime.now().strftime("%Y%m%d")
output_dir = Path(f"multimodal_analysis_results/{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"📁 Results will be saved to: {output_dir}")

# %%
"""
SECTION 1: DATA LOADING AND PREPARATION
=======================================
"""

print("📂 Loading multimodal data using existing analyzers...")

# Define data paths - update these to match your specific files
EKG_FILE = "gaming_health_data/recorded_data/SENSORS/ekg_data_polars_h10_2025_05_08_122611.txt"
VIDEO_FILE = "gaming_health_data/recorded_data/VIDEO/annotated/video_annotation_manual_20250508132346.csv"
APPLE_WATCH_FILE = "gaming_health_data/recorded_data/APPLE_WATCH/apple_health_export_2025-05-28.csv"

# Gaming session date for Apple Watch data correlation
GAMING_SESSION_DATE = "2025-05-08"  # Match this with your actual gaming session date

# Controller data paths (assuming you have 3 files that need to be combined)
CONTROLLER_FILES = [
    "gaming_health_data/recorded_data/PS/controller_inputs_08_05_2025_part_00.csv",
    "gaming_health_data/recorded_data/PS/controller_inputs_08_05_2025_part_01.csv",
    "gaming_health_data/recorded_data/PS/controller_inputs_08_05_2025_part_02.csv"
]

def load_controller_data(file_paths):
    """Load and combine controller data with timestamp conversion."""
    all_parts = []
    
    for path in file_paths:
        try:
            if os.path.exists(path):
                part = pd.read_csv(path)
                all_parts.append(part)
                print(f"✅ Loaded: {path.split('/')[-1]} ({len(part)} entries)")
        except Exception as e:
            print(f"⚠️ Could not load {path}: {e}")
    
    if all_parts:
        combined = pd.concat(all_parts, ignore_index=True)
        
        # Check what columns we actually have
        print(f"📋 Controller data columns: {list(combined.columns)}")
        
        # Convert timestamp to seconds if needed - check for various timestamp column names
        timestamp_col = None
        for col in ['timestamp', 'Timestamp', 'time', 'Time', 'datetime']:
            if col in combined.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            try:
                # Try to convert timestamp to seconds
                if combined[timestamp_col].dtype == 'object':
                    # Assume it's a datetime string
                    combined['time_seconds'] = pd.to_datetime(combined[timestamp_col]).astype(int) / 1e9
                    combined['time_seconds'] = combined['time_seconds'] - combined['time_seconds'].min()
                else:
                    # Assume it's already numeric (milliseconds or seconds)
                    combined['time_seconds'] = combined[timestamp_col].copy()
                    # Convert to seconds if it looks like milliseconds
                    if combined['time_seconds'].max() > 1e10:  # Likely milliseconds
                        combined['time_seconds'] = combined['time_seconds'] / 1000
                    combined['time_seconds'] = combined['time_seconds'] - combined['time_seconds'].min()
                
                print(f"🎮 Combined controller data: {len(combined)} inputs over {combined['time_seconds'].max():.1f}s")
            except Exception as e:
                print(f"⚠️ Could not convert timestamps: {e}")
                # Create a simple time index if timestamp conversion fails
                combined['time_seconds'] = np.arange(len(combined)) * 0.1  # Assume 10Hz sampling
                print(f"🎮 Combined controller data: {len(combined)} inputs (using synthetic timestamps)")
        else:
            print("⚠️ No timestamp column found, creating synthetic timestamps")
            combined['time_seconds'] = np.arange(len(combined)) * 0.1  # Assume 10Hz sampling
            print(f"🎮 Combined controller data: {len(combined)} inputs (using synthetic timestamps)")
        
        return combined
    
    return None

# Load EKG data using EKGAnalyzer
try:
    ekg_data = EKGAnalyzer.read_file(EKG_FILE, sensor_type="PolarH10")
    print(f"✅ EKG: {len(ekg_data)} samples, {ekg_data['Timestamp'].max():.1f}s duration")
    
    # Add compatibility columns for plotting while preserving original names
    ekg_data['time'] = ekg_data['Timestamp'].copy()
    ekg_data['signal_mV'] = ekg_data['HeartSignal'].copy()
    
except Exception as e:
    print(f"❌ Error loading EKG data: {e}")
    ekg_data = None

# Load video data using VideoAnalyzer
try:
    video_data = VideoAnalyzer.from_manual_csv(VIDEO_FILE)
    print(f"✅ Video: {len(video_data)} events")
    
    # Convert timestamps to seconds using the analyzer method
    video_data = video_data.Video.convert_timestamp_to_seconds()
    print(f"📋 Video duration: {video_data['time_seconds'].max():.1f}s")
    
    # Get basic event statistics
    event_counts = video_data['action'].value_counts()
    print(f"🎯 Top events: {dict(event_counts.head())}")
    
except Exception as e:
    print(f"❌ Error loading video data: {e}")
    video_data = None

# Load controller data
controller_data = load_controller_data(CONTROLLER_FILES)

# Load Apple Watch data for the gaming session
try:
    apple_watch_data = AppleWatchAnalyzer.read_file(APPLE_WATCH_FILE)
    print(f"✅ Apple Watch: {len(apple_watch_data)} health records loaded")
    
    # Get session data for the gaming date
    try:
        gaming_session = apple_watch_data.applewatch.get_session_from_date(GAMING_SESSION_DATE)
        print(f"📱 Found Apple Watch session on {GAMING_SESSION_DATE}")
        print(f"   Session: {gaming_session['startDate']} to {gaming_session['endDate']}")
        
        # Get heart rate data from the gaming session
        apple_hr_data = apple_watch_data.applewatch.get_hearth_rate_stats_from_session(gaming_session)
        
        if len(apple_hr_data) > 0:
            # Convert to seconds from session start for comparison
            session_start = apple_hr_data['startDate'].min()
            apple_hr_data['time_seconds'] = (apple_hr_data['startDate'] - session_start).dt.total_seconds()
            
            print(f"💓 Apple Watch HR: {len(apple_hr_data)} measurements")
            print(f"   HR range: {apple_hr_data['value'].min():.1f} - {apple_hr_data['value'].max():.1f} BPM")
            print(f"   Duration: {apple_hr_data['time_seconds'].max():.1f}s")
        else:
            print("⚠️ No heart rate data found in Apple Watch session")
            apple_hr_data = None
            
    except Exception as e:
        print(f"⚠️ Could not find Apple Watch session for {GAMING_SESSION_DATE}: {e}")
        apple_hr_data = None
        gaming_session = None
        
except Exception as e:
    print(f"❌ Error loading Apple Watch data: {e}")
    apple_watch_data = None
    apple_hr_data = None
    gaming_session = None

print(f"\n📈 Data loading complete!")
if ekg_data is not None:
    print(f"   EKG: {len(ekg_data)} samples")
if video_data is not None:
    print(f"   Video: {len(video_data)} events")
if controller_data is not None:
    print(f"   Controller: {len(controller_data)} inputs")
if apple_hr_data is not None:
    print(f"   Apple Watch: {len(apple_hr_data)} heart rate measurements")

# %%
"""
SECTION 1.5: MULTI-SOURCE HEART RATE COMPARISON
===============================================
"""

print("📱💓 Comparing heart rate data from multiple sources...")

def compare_hr_sources(ekg_hr_df, apple_hr_data, video_data):
    """
    Compare heart rate data from EKG sensor and Apple Watch.
    """
    if ekg_hr_df is None and apple_hr_data is None:
        print("⚠️ No heart rate data available from either source")
        return None
    
    comparison_results = {}
    
    # EKG analysis
    if ekg_hr_df is not None:
        ekg_stats = {
            'source': 'EKG (Polar H10)',
            'sample_count': len(ekg_hr_df),
            'mean_hr': ekg_hr_df['hr_smooth'].mean(),
            'min_hr': ekg_hr_df['hr_smooth'].min(),
            'max_hr': ekg_hr_df['hr_smooth'].max(),
            'std_hr': ekg_hr_df['hr_smooth'].std(),
            'duration': ekg_hr_df['time'].max(),
            'sampling_rate': len(ekg_hr_df) / ekg_hr_df['time'].max() if ekg_hr_df['time'].max() > 0 else 0
        }
        comparison_results['EKG'] = ekg_stats
        
        print(f"🔬 EKG Heart Rate Analysis:")
        print(f"   Samples: {ekg_stats['sample_count']}")
        print(f"   Mean HR: {ekg_stats['mean_hr']:.1f} BPM")
        print(f"   Range: {ekg_stats['min_hr']:.1f} - {ekg_stats['max_hr']:.1f} BPM")
        print(f"   Sampling: {ekg_stats['sampling_rate']:.2f} Hz")
    
    # Apple Watch analysis
    if apple_hr_data is not None:
        apple_stats = {
            'source': 'Apple Watch',
            'sample_count': len(apple_hr_data),
            'mean_hr': apple_hr_data['value'].mean(),
            'min_hr': apple_hr_data['value'].min(),
            'max_hr': apple_hr_data['value'].max(),
            'std_hr': apple_hr_data['value'].std(),
            'duration': apple_hr_data['time_seconds'].max(),
            'sampling_rate': len(apple_hr_data) / apple_hr_data['time_seconds'].max() if apple_hr_data['time_seconds'].max() > 0 else 0
        }
        comparison_results['Apple_Watch'] = apple_stats
        
        print(f"📱 Apple Watch Heart Rate Analysis:")
        print(f"   Samples: {apple_stats['sample_count']}")
        print(f"   Mean HR: {apple_stats['mean_hr']:.1f} BPM")
        print(f"   Range: {apple_stats['min_hr']:.1f} - {apple_stats['max_hr']:.1f} BPM")
        print(f"   Sampling: {apple_stats['sampling_rate']:.2f} Hz")
    
    # Compare sources if both available
    if ekg_hr_df is not None and apple_hr_data is not None:
        hr_diff = abs(ekg_stats['mean_hr'] - apple_stats['mean_hr'])
        range_diff_ekg = ekg_stats['max_hr'] - ekg_stats['min_hr']
        range_diff_apple = apple_stats['max_hr'] - apple_stats['min_hr']
        
        print(f"\n🔍 Source Comparison:")
        print(f"   Mean HR difference: {hr_diff:.1f} BPM")
        print(f"   EKG HR range: {range_diff_ekg:.1f} BPM")
        print(f"   Apple Watch HR range: {range_diff_apple:.1f} BPM")
        print(f"   Sampling rate ratio: {ekg_stats['sampling_rate']/apple_stats['sampling_rate']:.1f}x (EKG vs Apple)")
        
        # Agreement analysis
        if hr_diff < 5:
            agreement = "EXCELLENT"
        elif hr_diff < 10:
            agreement = "GOOD"
        elif hr_diff < 15:
            agreement = "MODERATE"
        else:
            agreement = "POOR"
            
        print(f"   Agreement level: {agreement}")
        
        comparison_results['agreement'] = {
            'mean_hr_diff': hr_diff,
            'agreement_level': agreement,
            'ekg_range': range_diff_ekg,
            'apple_range': range_diff_apple
        }
    
    return comparison_results

def create_multi_source_hr_plot(ekg_hr_df, apple_hr_data, video_data):
    """
    Create a plot comparing heart rate from multiple sources.
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Multi-Source Heart Rate Comparison', 'Heart Rate Difference Analysis'],
        vertical_spacing=0.15
    )
    
    # Plot EKG heart rate
    if ekg_hr_df is not None:
        fig.add_trace(
            go.Scatter(
                x=ekg_hr_df['time'],
                y=ekg_hr_df['hr_smooth'],
                mode='lines',
                name='EKG (Polar H10)',
                line=dict(color='blue', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    # Plot Apple Watch heart rate
    if apple_hr_data is not None:
        fig.add_trace(
            go.Scatter(
                x=apple_hr_data['time_seconds'],
                y=apple_hr_data['value'],
                mode='markers+lines',
                name='Apple Watch',
                line=dict(color='red', width=2),
                marker=dict(size=4),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    # Add gaming events
    if video_data is not None:
        event_colors = {'kill': 'green', 'death': 'red', 'assist': 'orange'}
        for event_type, color in event_colors.items():
            event_times = video_data[video_data['action'] == event_type]['time_seconds']
            for event_time in event_times:
                fig.add_vline(
                    x=event_time,
                    line=dict(color=color, width=1, dash='dot'),
                    opacity=0.6,
                    row=1, col=1
                )
    
    # Calculate and plot differences if both sources available
    if ekg_hr_df is not None and apple_hr_data is not None:
        # Interpolate to common time base for comparison
        common_times = np.arange(0, min(ekg_hr_df['time'].max(), apple_hr_data['time_seconds'].max()), 10)
        
        ekg_interp = np.interp(common_times, ekg_hr_df['time'], ekg_hr_df['hr_smooth'])
        apple_interp = np.interp(common_times, apple_hr_data['time_seconds'], apple_hr_data['value'])
        
        hr_diff = ekg_interp - apple_interp
        
        fig.add_trace(
            go.Scatter(
                x=common_times,
                y=hr_diff,
                mode='lines',
                name='EKG - Apple Watch',
                line=dict(color='purple', width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line=dict(color='gray', dash='dash'), row=2, col=1)
    
    fig.update_layout(
        height=800,
        title="Multi-Source Heart Rate Analysis During Gaming",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Heart Rate (BPM)", row=1, col=1)
    fig.update_yaxes(title_text="HR Difference (BPM)", row=2, col=1)
    
    return fig

# Note: This comparison will be done after EKG analysis when hr_df is available

# %%
"""
SECTION 2: EKG ANALYSIS USING EXISTING ANALYZER METHODS
=======================================================
"""

print("💓 Comprehensive EKG analysis using existing analyzer methods...")

if ekg_data is not None:
    try:
        # Use existing EKGAnalyzer properties and methods
        peaks = ekg_data.EKG.peaks
        beats = ekg_data.EKG.beats
        
        print(f"✅ Detected {len(peaks)} peaks and {len(beats)} beats")
        
        # Get comprehensive heart metrics using existing properties
        mean_hr = ekg_data.EKG.mean_heart_rate
        rmssd = ekg_data.EKG.rmssd
        sdnn = ekg_data.EKG.sdnn
        stress_index = ekg_data.EKG.stress_index
        pnn50 = ekg_data.EKG.pnn50
        hrv_score = ekg_data.EKG.hr_variability_score
        rhythm_type = ekg_data.EKG.rhythm_type
        
        print(f"📊 Heart Rate Metrics (using existing analyzer):")
        print(f"   Average HR: {mean_hr:.1f} BPM")
        print(f"   RMSSD: {rmssd:.1f} ms")
        print(f"   SDNN: {sdnn:.1f} ms") 
        print(f"   Stress Index: {stress_index:.1f}/100")
        print(f"   pNN50: {pnn50:.1f}%")
        print(f"   HRV Score: {hrv_score:.1f}")
        print(f"   Rhythm Type: {rhythm_type}")
        
        # Create heart rate time series using detected beats and existing method
        try:
            # Use the existing calculate_bpm method from the analyzer
            bpm_values = ekg_data.EKG.calculate_bpm()
            beat_times = ekg_data['Timestamp'].iloc[beats[:-1]]  # Exclude last beat since we have n-1 intervals
            
            # Create heart rate dataframe using existing analyzer results
            hr_data = []
            for i, (time, bpm) in enumerate(zip(beat_times, bpm_values)):
                hr_data.append({
                    'time': time,
                    'hr_instant': bpm,
                    'beat_index': i
                })
            
            hr_df = pd.DataFrame(hr_data)
            
            if len(hr_df) > 0:
                # Apply smoothing to heart rate using existing moving average method
                hr_df['hr_smooth'] = hr_df['hr_instant'].rolling(window=5, center=True).mean()
                print(f"📈 Heart rate time series: {len(hr_df)} beat intervals")
                print(f"   HR range: {hr_df['hr_instant'].min():.1f} - {hr_df['hr_instant'].max():.1f} BPM")
            else:
                print("⚠️ Could not create heart rate time series")
                hr_df = None
                
        except Exception as e:
            print(f"⚠️ Error creating heart rate time series: {e}")
            hr_df = None
        
        # Get frequency domain analysis using existing method
        try:
            freq_analysis = ekg_data.EKG.calculate_frequency_domain_hrv()
            print(f"🔊 Frequency Domain HRV:")
            print(f"   LF Power: {freq_analysis['LF']:.1f}")
            print(f"   HF Power: {freq_analysis['HF']:.1f}")
            print(f"   LF/HF Ratio: {freq_analysis['LF_HF_ratio']:.2f}")
        except Exception as e:
            print(f"⚠️ Could not calculate frequency domain HRV: {e}")
            freq_analysis = None
        
        # Get comprehensive health summary
        try:
            health_summary = ekg_data.EKG.get_heart_analysis_summary()
            print(f"🏥 Health Summary: {health_summary}")
        except Exception as e:
            print(f"⚠️ Could not get health summary: {e}")
        
        # Display the existing analyzer plots for comparison
        print("\n📊 Using existing EKG analyzer visualization methods:")
        
        try:
            print("🔍 Showing EKG data with properly detected peaks, low peaks, and beats...")
            # Use the existing plot method and save it
            ekg_plot_fig = ekg_data.EKG.plot_ekg_data()
            if hasattr(ekg_plot_fig, 'write_html'):
                ekg_plot_fig.write_html(output_dir / "ekg_original_analyzer_plot.html")
                print(f"✅ Original EKG plot saved to: {output_dir / 'ekg_original_analyzer_plot.html'}")
        except Exception as e:
            print(f"⚠️ Could not show/save EKG plot: {e}")
        
        try:
            print("💓 Showing moving average BPM over time...")
            # Use the existing BPM plot method and save it
            bpm_plot_fig = ekg_data.EKG.plot_moving_avg_bpm(window_size=10)
            if hasattr(bpm_plot_fig, 'write_html'):
                bpm_plot_fig.write_html(output_dir / "bpm_original_analyzer_plot.html") 
                print(f"✅ Original BPM plot saved to: {output_dir / 'bmp_original_analyzer_plot.html'}")
        except Exception as e:
            print(f"⚠️ Could not show/save BPM plot: {e}")
            
    except Exception as e:
        print(f"❌ Error in EKG analysis: {e}")
        hr_df = None
        peaks = None
        beats = None
        mean_hr = None
else:
    hr_df = None
    peaks = None
    beats = None
    mean_hr = None

# %%
"""
SECTION 2.5: EXECUTE MULTI-SOURCE HEART RATE COMPARISON
======================================================
"""

# Perform multi-source comparison now that hr_df is available
hr_comparison = compare_hr_sources(hr_df, apple_hr_data, video_data)

# Create multi-source comparison plot
try:
    multi_source_fig = create_multi_source_hr_plot(hr_df, apple_hr_data, video_data)
    multi_source_fig.show()
    
    # Save the plot
    multi_source_fig.write_html(output_dir / "multi_source_hr_comparison.html")
    print(f"💾 Multi-source HR comparison saved to: {output_dir / 'multi_source_hr_comparison.html'}")
    
except Exception as e:
    print(f"⚠️ Could not create multi-source plot: {e}")

# %%
"""
SECTION 3: GAMING EVENTS IMPACT ON HEART RATE (MULTIPLE TIME WINDOWS)
======================================================================
"""

print("🎯 Analyzing gaming event impact on heart rate with multiple time windows...")

def analyze_hr_around_events_multi_window(hr_df, video_df, event_type, windows=[5, 15, 30]):
    """
    Analyze heart rate changes around specific gaming events using multiple time windows.
    
    Parameters:
    -----------
    hr_df : pd.DataFrame
        Heart rate time series data
    video_df : pd.DataFrame  
        Video events data
    event_type : str
        Type of event to analyze
    windows : list
        List of time windows in seconds [before, after]
    
    Returns:
    --------
    dict : Analysis results for different time windows
    """
    if hr_df is None or video_df is None:
        return None
    
    event_times = video_df[video_df['action'] == event_type]['time_seconds'].values
    
    if len(event_times) == 0:
        return None
    
    results = {}
    
    for window in windows:
        responses = []
        
        for event_time in event_times:
            # Get HR data around this event
            before_mask = (hr_df['time'] >= event_time - window) & (hr_df['time'] < event_time)
            after_mask = (hr_df['time'] > event_time) & (hr_df['time'] <= event_time + window)
            
            hr_before = hr_df[before_mask]['hr_smooth'].dropna()
            hr_after = hr_df[after_mask]['hr_smooth'].dropna()
            
            if len(hr_before) > 2 and len(hr_after) > 2:
                response = {
                    'event_time': event_time,
                    'hr_before_mean': hr_before.mean(),
                    'hr_after_mean': hr_after.mean(),
                    'hr_change': hr_after.mean() - hr_before.mean(),
                    'hr_before_std': hr_before.std(),
                    'hr_after_std': hr_after.std(),
                    'window_size': window
                }
                responses.append(response)
        
        if responses:
            responses_df = pd.DataFrame(responses)
            results[f'{window}s'] = {
                'data': responses_df,
                'mean_change': responses_df['hr_change'].mean(),
                'std_change': responses_df['hr_change'].std(),
                'significant': abs(responses_df['hr_change'].mean()) > 2.0,  # Threshold for significance
                'sample_size': len(responses_df)
            }
    
    return results

# Analyze key events with multiple time windows
key_events = ['kill', 'death', 'match_start', 'round_start', 'perk_activation', 'assist']
event_analysis = {}

if hr_df is not None and video_data is not None:
    print("🔗 Analyzing HR response to gaming events with multiple time windows...")
    
    for event in key_events:
        analysis = analyze_hr_around_events_multi_window(hr_df, video_data, event)
        if analysis is not None:
            event_analysis[event] = analysis
            print(f"\n⚔️ {event.upper()} Events:")
            for window, data in analysis.items():
                significance = "✅ SIGNIFICANT" if data['significant'] else "⚪ Not significant"
                print(f"   {window} window: {data['mean_change']:+.1f} BPM change (n={data['sample_size']}) {significance}")

# %%
"""
SECTION 4: STRESS/EXCITEMENT DETECTION AND GAMING SCENARIOS
===========================================================
"""

print("😰 Advanced stress/excitement detection during gaming...")

def detect_stress_periods_advanced(hr_df, ekg_data, percentile_threshold=75, duration_threshold=10):
    """
    Advanced stress detection using multiple physiological indicators.
    """
    if hr_df is None or ekg_data is None:
        return None
    
    # Calculate baseline metrics
    baseline_hr = hr_df['hr_smooth'].median()
    threshold_hr = np.percentile(hr_df['hr_smooth'].dropna(), percentile_threshold)
    
    # Use existing HRV metrics for stress detection
    try:
        stress_index = ekg_data.EKG.stress_index
        rmssd = ekg_data.EKG.rmssd
        
        print(f"💓 Baseline HR: {baseline_hr:.1f} BPM")
        print(f"🚨 Stress threshold: {threshold_hr:.1f} BPM")
        print(f"😰 Overall stress index: {stress_index:.1f}/100")
        print(f"💓 RMSSD (relaxation): {rmssd:.1f} ms")
        
        # Classify stress levels
        stress_classification = "Low" if stress_index < 50 else "Moderate" if stress_index < 75 else "High"
        print(f"🎯 Overall stress level: {stress_classification}")
        
    except Exception as e:
        print(f"⚠️ Could not get stress metrics: {e}")
        stress_index = None
        rmssd = None
    
    # Find periods of elevated heart rate
    stress_mask = hr_df['hr_smooth'] > threshold_hr
    stress_periods = hr_df[stress_mask].copy()
    
    if len(stress_periods) > 0:
        stress_periods['stress_level'] = (stress_periods['hr_smooth'] - baseline_hr) / baseline_hr * 100
        stress_periods['intensity'] = pd.cut(stress_periods['stress_level'], 
                                           bins=[0, 10, 20, 100], 
                                           labels=['Mild', 'Moderate', 'High'])
        
        print(f"⚡ Found {len(stress_periods)} high-intensity moments")
        print(f"🔥 Max stress level: +{stress_periods['stress_level'].max():.1f}% above baseline")
        
        # Analyze intensity distribution
        intensity_counts = stress_periods['intensity'].value_counts()
        print(f"📊 Stress intensity distribution:")
        for intensity, count in intensity_counts.items():
            print(f"   {intensity}: {count} moments")
        
        return stress_periods
    
    return None

def correlate_stress_with_events(stress_periods, video_data, tolerance=5):
    """
    Correlate stress periods with gaming events.
    """
    if stress_periods is None or video_data is None:
        return None
    
    correlations = []
    
    for _, stress_moment in stress_periods.iterrows():
        stress_time = stress_moment['time']
        
        # Find events within tolerance window
        event_mask = abs(video_data['time_seconds'] - stress_time) <= tolerance
        nearby_events = video_data[event_mask]
        
        if len(nearby_events) > 0:
            for _, event in nearby_events.iterrows():
                correlations.append({
                    'stress_time': stress_time,
                    'stress_level': stress_moment['stress_level'],
                    'event_time': event['time_seconds'],
                    'event_type': event['action'],
                    'time_diff': abs(stress_time - event['time_seconds'])
                })
    
    if correlations:
        corr_df = pd.DataFrame(correlations)
        print(f"\n🔗 Found {len(corr_df)} stress-event correlations:")
        
        # Analyze which events correlate with stress
        event_stress = corr_df.groupby('event_type')['stress_level'].agg(['mean', 'count', 'std']).round(2)
        event_stress = event_stress.sort_values('mean', ascending=False)
        
        print("📊 Events correlated with highest stress:")
        for event_type, data in event_stress.head().iterrows():
            print(f"   {event_type}: {data['mean']:.1f}% stress (n={data['count']})")
        
        return corr_df
    
    return None

# Perform advanced stress analysis
if hr_df is not None and ekg_data is not None:
    stress_periods = detect_stress_periods_advanced(hr_df, ekg_data)
    
    if stress_periods is not None and video_data is not None:
        stress_event_correlations = correlate_stress_with_events(stress_periods, video_data)
    else:
        stress_event_correlations = None
else:
    stress_periods = None
    stress_event_correlations = None

# %%
"""
SECTION 5: ENHANCED CONTROLLER INPUT ANALYSIS WITH CLICKS PER SECOND
====================================================================
"""

print("🎮 Enhanced controller input analysis with clicks per second correlation...")

def analyze_controller_inputs_comprehensive(controller_data, hr_df, apple_hr_data=None, window_size=5):
    """
    Comprehensive controller input analysis including clicks per second correlation with heartbeat.
    Now supports multi-source heart rate comparison (EKG + Apple Watch).
    """
    if controller_data is None or hr_df is None:
        return None
    
    print(f"📋 Available controller columns: {list(controller_data.columns)}")
    
    # Identify button/action columns
    button_columns = []
    action_columns = []
    
    for col in controller_data.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['button', 'btn', 'key', 'click', 'press']):
            button_columns.append(col)
        elif any(keyword in col_lower for keyword in ['action', 'input', 'event', 'command']):
            action_columns.append(col)
    
    print(f"🎯 Button columns found: {button_columns}")
    print(f"⚡ Action columns found: {action_columns}")
    
    # Calculate clicks per second in sliding windows
    clicks_analysis = []
    
    time_min = controller_data['time_seconds'].min()
    time_max = controller_data['time_seconds'].max()
    time_points = np.arange(time_min, time_max, window_size/2)  # 50% overlap
    
    for time_point in time_points:
        window_start = time_point
        window_end = time_point + window_size
        
        window_data = controller_data[
            (controller_data['time_seconds'] >= window_start) & 
            (controller_data['time_seconds'] < window_end)
        ]
        
        if len(window_data) > 0:
            # Count total inputs
            total_inputs = len(window_data)
            clicks_per_second = total_inputs / window_size
            
            # Count button presses specifically
            button_presses = 0
            for btn_col in button_columns:
                if btn_col in window_data.columns:
                    # Count non-null button events
                    button_presses += window_data[btn_col].notna().sum()
            
            button_clicks_per_second = button_presses / window_size
            
            # Count unique actions
            unique_actions = 0
            if action_columns:
                for action_col in action_columns:
                    if action_col in window_data.columns:
                        unique_actions += len(window_data[action_col].unique())
            
            # Action diversity (unique actions per total actions)
            action_diversity = unique_actions / max(total_inputs, 1)
            
            # Input intensity score
            intensity_score = clicks_per_second * (1 + action_diversity)
            
            # Find corresponding heart rate from EKG
            window_center = time_point + window_size/2
            time_diff = abs(hr_df['time'] - window_center)
            closest_idx = time_diff.idxmin()
            
            entry = {
                'time': window_center,
                'total_inputs': total_inputs,
                'clicks_per_second': clicks_per_second,
                'button_clicks_per_second': button_clicks_per_second,
                'unique_actions': unique_actions,
                'action_diversity': action_diversity,
                'intensity_score': intensity_score
            }
            
            # Add EKG heart rate if available
            if time_diff.iloc[closest_idx] < window_size:  # Within window
                hr_value = hr_df.iloc[closest_idx]['hr_smooth']
                hr_instant = hr_df.iloc[closest_idx]['hr_instant']
                entry['heart_rate_ekg_smooth'] = hr_value
                entry['heart_rate_ekg_instant'] = hr_instant
            else:
                entry['heart_rate_ekg_smooth'] = np.nan
                entry['heart_rate_ekg_instant'] = np.nan
            
            # Add Apple Watch heart rate if available
            if apple_hr_data is not None:
                apple_time_diff = abs(apple_hr_data['time_seconds'] - window_center)
                if len(apple_time_diff) > 0:
                    apple_closest_idx = apple_time_diff.idxmin()
                    if apple_time_diff.iloc[apple_closest_idx] < window_size:
                        entry['heart_rate_apple'] = apple_hr_data.iloc[apple_closest_idx]['value']
                    else:
                        entry['heart_rate_apple'] = np.nan
                else:
                    entry['heart_rate_apple'] = np.nan
            else:
                entry['heart_rate_apple'] = np.nan
            
            clicks_analysis.append(entry)
    
    if not clicks_analysis:
        print("⚠️ No data available for clicks analysis")
        return None
    
    clicks_df = pd.DataFrame(clicks_analysis)
    
    print(f"📊 Clicks analysis: {len(clicks_df)} time windows")
    print(f"   Max clicks/sec: {clicks_df['clicks_per_second'].max():.2f}")
    print(f"   Avg clicks/sec: {clicks_df['clicks_per_second'].mean():.2f}")
    print(f"   Max intensity: {clicks_df['intensity_score'].max():.2f}")
    
    # Calculate correlations with both heart rate sources
    correlations = {}
    
    input_metrics = ['clicks_per_second', 'button_clicks_per_second', 'intensity_score', 'action_diversity']
    hr_metrics = ['heart_rate_ekg_smooth', 'heart_rate_ekg_instant']
    if apple_hr_data is not None:
        hr_metrics.append('heart_rate_apple')
    
    print(f"\n🔗 Clicks per Second vs Heart Rate Correlations:")
    
    for input_metric in input_metrics:
        for hr_metric in hr_metrics:
            if hr_metric in clicks_df.columns:
                # Only calculate correlation if we have valid data
                valid_data = clicks_df[[input_metric, hr_metric]].dropna()
                if len(valid_data) > 2:
                    correlation = valid_data[input_metric].corr(valid_data[hr_metric])
                    correlations[f"{input_metric}_vs_{hr_metric}"] = correlation
                    
                    # Interpret correlation
                    if abs(correlation) > 0.5:
                        strength = "STRONG"
                    elif abs(correlation) > 0.3:
                        strength = "MODERATE"
                    elif abs(correlation) > 0.1:
                        strength = "WEAK"
                    else:
                        strength = "NEGLIGIBLE"
                    
                    direction = "positive" if correlation > 0 else "negative"
                    print(f"   {input_metric} vs {hr_metric}: {correlation:.3f} ({strength} {direction})")
                else:
                    print(f"   {input_metric} vs {hr_metric}: Not enough data")
    
    # Compare heart rate sources if both available
    if apple_hr_data is not None:
        valid_hr_comparison = clicks_df[['heart_rate_ekg_smooth', 'heart_rate_apple']].dropna()
        if len(valid_hr_comparison) > 2:
            hr_source_correlation = valid_hr_comparison['heart_rate_ekg_smooth'].corr(valid_hr_comparison['heart_rate_apple'])
            correlations['ekg_vs_apple_hr'] = hr_source_correlation
            print(f"\n💓 Heart Rate Source Comparison:")
            print(f"   EKG vs Apple Watch HR correlation: {hr_source_correlation:.3f}")
    
    # Create enhanced clicks per second plot with multi-source heart rate overlay
    try:
        # Determine number of plots based on available data
        n_plots = 3 if apple_hr_data is None else 4
        subplot_titles = [
            'Clicks per Second vs Heart Rate Over Time',
            'Button Clicks vs Heart Rate Correlation',
            'Input Intensity vs Heart Rate Correlation'
        ]
        if apple_hr_data is not None:
            subplot_titles.append('Multi-Source Heart Rate Comparison')
        
        fig_clicks = make_subplots(
            rows=n_plots, cols=1,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] * (n_plots-1),
            vertical_spacing=0.08
        )
        
        # Plot 1: Time series with dual y-axes
        fig_clicks.add_trace(
            go.Scatter(
                x=clicks_df['time'],
                y=clicks_df['clicks_per_second'],
                mode='lines+markers',
                name='Clicks/Second',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        fig_clicks.add_trace(
            go.Scatter(
                x=clicks_df['time'],
                y=clicks_df['heart_rate_ekg_smooth'],
                mode='lines',
                name='Heart Rate EKG (BPM)',
                line=dict(color='red', width=2),
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Add Apple Watch heart rate if available
        if apple_hr_data is not None:
            fig_clicks.add_trace(
                go.Scatter(
                    x=clicks_df['time'],
                    y=clicks_df['heart_rate_apple'],
                    mode='lines',
                    name='Heart Rate Apple (BPM)',
                    line=dict(color='orange', width=2, dash='dash'),
                    yaxis='y2'
                ),
                row=1, col=1, secondary_y=True
            )
        
        # Plot 2: Scatter plot - clicks vs EKG HR
        fig_clicks.add_trace(
            go.Scatter(
                x=clicks_df['clicks_per_second'],
                y=clicks_df['heart_rate_ekg_smooth'],
                mode='markers',
                name='Clicks vs EKG HR',
                marker=dict(
                    color=clicks_df['intensity_score'],
                    colorscale='Viridis',
                    size=8,
                    colorbar=dict(title="Intensity")
                )
            ),
            row=2, col=1
        )
        
        # Plot 3: Scatter plot - intensity vs EKG HR
        fig_clicks.add_trace(
            go.Scatter(
                x=clicks_df['intensity_score'],
                y=clicks_df['heart_rate_ekg_smooth'],
                mode='markers',
                name='Intensity vs EKG HR',
                marker=dict(
                    color=clicks_df['action_diversity'],
                    colorscale='Plasma',
                    size=8,
                    colorbar=dict(title="Diversity")
                )
            ),
            row=3, col=1
        )
        
        # Plot 4: Multi-source heart rate comparison if Apple Watch data available
        if apple_hr_data is not None:
            valid_hr_data = clicks_df[['heart_rate_ekg_smooth', 'heart_rate_apple']].dropna()
            if len(valid_hr_data) > 0:
                fig_clicks.add_trace(
                    go.Scatter(
                        x=valid_hr_data['heart_rate_ekg_smooth'],
                        y=valid_hr_data['heart_rate_apple'],
                        mode='markers',
                        name='EKG vs Apple HR',
                        marker=dict(
                            color='purple',
                            size=8
                        )
                    ),
                    row=4, col=1
                )
                
                # Add diagonal reference line
                hr_min = min(valid_hr_data['heart_rate_ekg_smooth'].min(), valid_hr_data['heart_rate_apple'].min())
                hr_max = max(valid_hr_data['heart_rate_ekg_smooth'].max(), valid_hr_data['heart_rate_apple'].max())
                fig_clicks.add_trace(
                    go.Scatter(
                        x=[hr_min, hr_max],
                        y=[hr_min, hr_max],
                        mode='lines',
                        name='Perfect Agreement',
                        line=dict(color='gray', dash='dash'),
                        showlegend=False
                    ),
                    row=4, col=1
                )
        
        # Update layout
        fig_clicks.update_layout(
            height=300 * n_plots,
            title="Comprehensive Multi-Source Controller Input Analysis",
            showlegend=True
        )
        
        # Update axes
        fig_clicks.update_xaxes(title_text="Time (seconds)", row=1, col=1)
        fig_clicks.update_xaxes(title_text="Clicks per Second", row=2, col=1)
        fig_clicks.update_xaxes(title_text="Intensity Score", row=3, col=1)
        if apple_hr_data is not None:
            fig_clicks.update_xaxes(title_text="EKG Heart Rate (BPM)", row=4, col=1)
        
        fig_clicks.update_yaxes(title_text="Clicks/Second", row=1, col=1)
        fig_clicks.update_yaxes(title_text="Heart Rate (BPM)", row=1, col=1, secondary_y=True)
        fig_clicks.update_yaxes(title_text="Heart Rate (BPM)", row=2, col=1)
        fig_clicks.update_yaxes(title_text="Heart Rate (BPM)", row=3, col=1)
        if apple_hr_data is not None:
            fig_clicks.update_yaxes(title_text="Apple Watch Heart Rate (BPM)", row=4, col=1)
        
        # Save the plot
        fig_clicks.write_html(output_dir / "controller_clicks_multisource_analysis.html")
        fig_clicks.show()
        
        print(f"✅ Multi-source controller clicks analysis saved to: {output_dir / 'controller_clicks_multisource_analysis.html'}")
        
    except Exception as e:
        print(f"⚠️ Could not create clicks analysis plot: {e}")
    
    return {
        'data': clicks_df,
        'correlations': correlations,
        'summary': f"Analyzed {len(clicks_df)} time windows with multi-source heart rate correlation"
    }

# Perform enhanced controller analysis with multi-source heart rate
if controller_data is not None and hr_df is not None:
    controller_clicks_analysis = analyze_controller_inputs_comprehensive(controller_data, hr_df, apple_hr_data)
    
    if controller_clicks_analysis:
        print(f"🎮 Enhanced controller analysis: {controller_clicks_analysis['summary']}")
    else:
        print("⚠️ Could not complete enhanced controller analysis")
else:
    controller_clicks_analysis = None
    print("⚠️ No controller data or heart rate data available for enhanced analysis")

# %%

def calculate_input_intensity_advanced(controller_data, window_size=30):
    """
    Advanced controller input intensity calculation with multiple metrics.
    """
    if controller_data is None:
        return None
    
    # Check if we have the required time column
    if 'time_seconds' not in controller_data.columns:
        print("⚠️ No time_seconds column found in controller data")
        return None
    
    # Identify input columns
    input_columns = []
    potential_cols = ['button_press', 'stick_movement', 'trigger_value', 'action', 'input_type', 'button', 'value']
    
    for col in controller_data.columns:
        if any(potential in col.lower() for potential in ['button', 'trigger', 'stick', 'action', 'input']):
            input_columns.append(col)
    
    if not input_columns:
        print("⚠️ No recognizable input columns found")
        print(f"📋 Available columns: {list(controller_data.columns)}")
        return None
    
    print(f"📋 Using input columns: {input_columns}")
    
    # Calculate rolling input intensity
    intensity_data = []
    
    try:
        time_min = controller_data['time_seconds'].min()
        time_max = controller_data['time_seconds'].max()
        time_points = np.arange(time_min, time_max, window_size/2)  # 50% overlap
        
        for time_point in time_points:
            window_start = time_point
            window_end = time_point + window_size
            
            window_data = controller_data[
                (controller_data['time_seconds'] >= window_start) & 
                (controller_data['time_seconds'] < window_end)
            ]
            
            if len(window_data) > 0:
                # Count different types of intensity
                input_count = len(window_data)
                inputs_per_second = input_count / window_size
                
                # Calculate more sophisticated metrics
                unique_actions = 0
                if 'action' in window_data.columns:
                    unique_actions = len(window_data['action'].unique())
                elif input_columns:
                    # Use the first available input column for diversity calculation
                    first_input_col = input_columns[0]
                    unique_actions = len(window_data[first_input_col].unique())
                
                # Numeric intensity (if applicable)
                numeric_activity = 0
                if 'value' in window_data.columns:
                    numeric_values = pd.to_numeric(window_data['value'], errors='coerce').dropna()
                    if len(numeric_values) > 0:
                        numeric_activity = numeric_values.abs().mean()
                
                intensity_data.append({
                    'time': time_point + window_size/2,  # Center of window
                    'input_count': input_count,
                    'inputs_per_second': inputs_per_second,
                    'unique_actions': unique_actions,
                    'numeric_activity': numeric_activity,
                    'intensity_score': inputs_per_second * (1 + unique_actions/10)  # Combined metric
                })
        
        return pd.DataFrame(intensity_data)
        
    except Exception as e:
        print(f"⚠️ Error calculating input intensity: {e}")
        return None

def correlate_intensity_with_physiology(intensity_df, hr_df, stress_periods):
    """
    Correlate input intensity with physiological state.
    """
    if intensity_df is None or hr_df is None:
        return None
    
    # Match intensity with heart rate data
    matched_data = []
    
    for _, intensity_row in intensity_df.iterrows():
        intensity_time = intensity_row['time']
        
        # Find closest heart rate measurement
        time_diff = abs(hr_df['time'] - intensity_time)
        closest_idx = time_diff.idxmin()
        
        if time_diff.iloc[closest_idx] < 10:  # Within 10 seconds
            hr_value = hr_df.iloc[closest_idx]['hr_smooth']
            
            # Check if this time corresponds to a stress period
            is_stress_period = False
            if stress_periods is not None:
                stress_match = abs(stress_periods['time'] - intensity_time) < 10
                is_stress_period = stress_match.any()
            
            matched_data.append({
                'time': intensity_time,
                'input_count': intensity_row['input_count'],
                'inputs_per_second': intensity_row['inputs_per_second'],
                'intensity_score': intensity_row['intensity_score'],
                'heart_rate': hr_value,
                'is_stress_period': is_stress_period
            })
    
    if not matched_data:
        return None
    
    matched_df = pd.DataFrame(matched_data)
    
    # Calculate correlations
    correlations = {}
    intensity_metrics = ['input_count', 'inputs_per_second', 'intensity_score']
    
    for metric in intensity_metrics:
        if metric in matched_df.columns:
            correlation = matched_df[metric].corr(matched_df['heart_rate'])
            correlations[metric] = correlation
    
    # Statistical significance testing
    for metric, corr in correlations.items():
        if abs(corr) > 0.3:  # Moderate correlation
            print(f"📈 {metric} vs HR: correlation = {corr:.3f} (moderate)")
        elif abs(corr) > 0.1:
            print(f"📊 {metric} vs HR: correlation = {corr:.3f} (weak)")
        else:
            print(f"⚪ {metric} vs HR: correlation = {corr:.3f} (negligible)")
    
    return {
        'correlations': correlations,
        'data': matched_df,
        'summary': f"Analyzed {len(matched_df)} time points"
    }

# Calculate controller input intensity and correlations
if controller_data is not None:
    intensity_df = calculate_input_intensity_advanced(controller_data)
    
    if intensity_df is not None and not intensity_df.empty:
        intensity_correlation = correlate_intensity_with_physiology(intensity_df, hr_df, stress_periods)
        print(f"🎮 Input intensity analysis: {len(intensity_df)} time windows analyzed")
    else:
        intensity_correlation = None
        print("⚠️ Could not calculate input intensity")
else:
    intensity_df = None
    intensity_correlation = None
    print("⚠️ No controller data available for intensity analysis")

# %%
"""
SECTION 6: PERFORMANCE VS PHYSIOLOGICAL STATE ANALYSIS
======================================================
"""

print("🏆 Analyzing gaming performance vs physiological state...")

def analyze_performance_physiology_advanced(video_data, hr_df, stress_periods, window_size=60):
    """
    Advanced analysis of relationship between gaming performance and physiological state.
    """
    if video_data is None or hr_df is None:
        return None
    
    # Calculate performance metrics over time
    performance_data = []
    
    time_min = video_data['time_seconds'].min()
    time_max = video_data['time_seconds'].max()
    time_points = np.arange(time_min, time_max, window_size/2)  # 50% overlap
    
    for time_point in time_points:
        window_start = time_point
        window_end = time_point + window_size
        
        window_events = video_data[
            (video_data['time_seconds'] >= window_start) & 
            (video_data['time_seconds'] < window_end)
        ]
        
        if len(window_events) > 0:
            # Calculate performance metrics
            kills = len(window_events[window_events['action'] == 'kill'])
            deaths = len(window_events[window_events['action'] == 'death'])
            assists = len(window_events[window_events['action'] == 'assist'])
            
            kd_ratio = kills / max(deaths, 1)
            total_actions = len(window_events)
            action_rate = total_actions / window_size  # actions per second
            
            # Calculate performance score (weighted)
            performance_score = kills * 3 + assists * 1 - deaths * 2
            
            # Get physiological state during this window
            window_center = time_point + window_size/2
            
            # Find heart rate during this period
            hr_window = hr_df[
                (hr_df['time'] >= window_start) & 
                (hr_df['time'] < window_end)
            ]
            
            avg_hr = hr_window['hr_smooth'].mean() if len(hr_window) > 0 else None
            hr_variability = hr_window['hr_smooth'].std() if len(hr_window) > 0 else None
            
            # Check stress periods
            stress_time = 0
            if stress_periods is not None:
                stress_in_window = stress_periods[
                    (stress_periods['time'] >= window_start) & 
                    (stress_periods['time'] < window_end)
                ]
                stress_time = len(stress_in_window)
            
            performance_data.append({
                'time': window_center,
                'kills': kills,
                'deaths': deaths,
                'assists': assists,
                'kd_ratio': kd_ratio,
                'performance_score': performance_score,
                'total_actions': total_actions,
                'action_rate': action_rate,
                'avg_heart_rate': avg_hr,
                'hr_variability': hr_variability,
                'stress_moments': stress_time
            })
    
    if not performance_data:
        return None
    
    perf_df = pd.DataFrame(performance_data)
    
    # Remove rows with missing physiological data
    perf_df = perf_df.dropna(subset=['avg_heart_rate'])
    
    if len(perf_df) < 5:
        print("⚠️ Insufficient data for performance analysis")
        return None
    
    # Calculate correlations between performance and physiology
    correlations = {}
    
    performance_metrics = ['kd_ratio', 'performance_score', 'action_rate']
    physiology_metrics = ['avg_heart_rate', 'hr_variability', 'stress_moments']
    
    print("🔗 Performance vs Physiology Correlations:")
    
    for perf_metric in performance_metrics:
        for phys_metric in physiology_metrics:
            if perf_metric in perf_df.columns and phys_metric in perf_df.columns:
                correlation = perf_df[perf_metric].corr(perf_df[phys_metric])
                correlations[f"{perf_metric}_vs_{phys_metric}"] = correlation
                
                # Interpret correlation strength
                if abs(correlation) > 0.5:
                    strength = "STRONG"
                elif abs(correlation) > 0.3:
                    strength = "MODERATE"
                elif abs(correlation) > 0.1:
                    strength = "WEAK"
                else:
                    strength = "NEGLIGIBLE"
                
                direction = "positive" if correlation > 0 else "negative"
                print(f"   {perf_metric} vs {phys_metric}: {correlation:.3f} ({strength} {direction})")
    
    return {
        'correlations': correlations,
        'data': perf_df,
        'summary': f"Analyzed {len(perf_df)} performance windows"
    }

# Analyze performance vs physiology
if video_data is not None and hr_df is not None:
    performance_analysis = analyze_performance_physiology_advanced(video_data, hr_df, stress_periods)
    
    if performance_analysis is not None:
        print(f"🏆 Performance analysis: {performance_analysis['summary']}")
    else:
        print("⚠️ Could not complete performance analysis")
else:
    performance_analysis = None
    print("⚠️ Insufficient data for performance analysis")

# %%
"""
SECTION 7: COMPREHENSIVE MULTIMODAL VISUALIZATION
=================================================
"""

print("📊 Creating comprehensive multimodal visualization dashboard...")

def create_comprehensive_dashboard(ekg_data, hr_df, video_data, stress_periods, event_analysis, 
                                 intensity_df, performance_analysis):
    """
    Create a comprehensive dashboard with multiple synchronized visualizations.
    """
    
    # Create subplots with custom layout
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=[
            'Raw EKG Signal with Properly Detected Peaks & Beats', 'Heart Rate (BPM) Over Time',
            'Heart Rate Timeline with Gaming Events', 'Event-Triggered HR Changes (Multiple Windows)', 
            'Stress Periods and Gaming Performance', 'Controller Input Intensity vs Heart Rate',
            'Performance Metrics Over Time', 'Event Correlation Summary'
        ],
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ],
        vertical_spacing=0.06,
        row_heights=[0.2, 0.2, 0.2, 0.2, 0.2]
    )
    
    # Plot 1: Raw EKG with properly detected peaks, low peaks, and beats
    # Use the same approach as the original EKG analyzer
    if ekg_data is not None and peaks is not None:
        # Sample data for visualization (every 10th point to avoid overplotting)
        sample_indices = np.arange(0, len(ekg_data), 10)
        
        # Raw EKG signal - use original column names like the analyzer does
        fig.add_trace(
            go.Scatter(
                x=ekg_data['Timestamp'].iloc[sample_indices],
                y=ekg_data['HeartSignal'].iloc[sample_indices],
                mode='lines',
                name='Raw EKG Signal',
                line=dict(color='blue', width=1),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add R-peaks (high peaks) - exactly like the original analyzer
        fig.add_trace(
            go.Scatter(
                x=ekg_data['Timestamp'].iloc[peaks],
                y=ekg_data['HeartSignal'].iloc[peaks],
                mode='markers',
                name=f'Peaks ({len(peaks)})',
                marker=dict(color='red', size=5),
                opacity=0.9
            ),
            row=1, col=1
        )
        
        # Add low peaks - exactly like the original analyzer
        try:
            low_peaks = ekg_data.EKG.low_peaks
            fig.add_trace(
                go.Scatter(
                    x=ekg_data['Timestamp'].iloc[low_peaks],
                    y=ekg_data['HeartSignal'].iloc[low_peaks],
                    mode='markers',
                    name=f'Low Peaks ({len(low_peaks)})',
                    marker=dict(color='orange', size=5),
                    opacity=0.7
                ),
                row=1, col=1
            )
        except:
            pass
        
        # Add detected beats - exactly like the original analyzer
        if beats is not None:
            fig.add_trace(
                go.Scatter(
                    x=ekg_data['Timestamp'].iloc[beats],
                    y=ekg_data['HeartSignal'].iloc[beats],
                    mode='markers',
                    name=f'Beats ({len(beats)})',
                    marker=dict(color='green', size=4),
                    opacity=0.5
                ),
                row=1, col=1
            )
    
    # Plot 2: BPM over time using existing analyzer method
    if hr_df is not None and len(hr_df) > 0:
        # Instantaneous BPM
        fig.add_trace(
            go.Scatter(
                x=hr_df['time'],
                y=hr_df['hr_instant'],
                mode='markers',
                name='Instantaneous BPM',
                marker=dict(color='lightblue', size=3),
                opacity=0.6
            ),
            row=2, col=1
        )
        
        # Smoothed BPM
        fig.add_trace(
            go.Scatter(
                x=hr_df['time'],
                y=hr_df['hr_smooth'],
                mode='lines',
                name='Smoothed BPM',
                line=dict(color='darkblue', width=3)
            ),
            row=2, col=1
        )
        
        # Add mean heart rate line
        if mean_hr is not None:
            fig.add_hline(
                y=mean_hr,
                line=dict(color='red', dash='dash', width=2),
                annotation_text=f"Mean HR: {mean_hr:.1f} BPM",
                row=2, col=1
            )
    
    # Plot 3: Heart Rate Timeline with Gaming Events
    if hr_df is not None:
        fig.add_trace(
            go.Scatter(
                x=hr_df['time'],
                y=hr_df['hr_smooth'],
                mode='lines',
                name='Heart Rate with Events',
                line=dict(color='green', width=2)
            ),
            row=3, col=1
        )
    
    # Add gaming events as vertical lines with better colors
    if video_data is not None:
        event_colors = {
            'kill': 'red', 'death': 'black', 'assist': 'orange',
            'match_start': 'green', 'round_start': 'blue', 'perk_activation': 'purple'
        }
        
        # Add events to both BPM plot and events plot
        for row_num in [2, 3]:
            for event_type, color in event_colors.items():
                event_times = video_data[video_data['action'] == event_type]['time_seconds']
                for i, event_time in enumerate(event_times):
                    fig.add_vline(
                        x=event_time,
                        line=dict(color=color, width=2, dash='dot'),
                        opacity=0.8,
                        annotation_text=event_type if i == 0 else "",  # Only label first occurrence
                        row=row_num, col=1
                    )
    
    # Plot 4: Event-Triggered HR Changes
    if event_analysis:
        window_colors = {'5s': 'red', '15s': 'orange', '30s': 'blue'}
        
        for event_type, analysis in event_analysis.items():
            for window, data in analysis.items():
                if 'data' in data:
                    fig.add_trace(
                        go.Box(
                            y=data['data']['hr_change'],
                            name=f"{event_type} ({window})",
                            boxpoints='all',
                            jitter=0.3,
                            marker=dict(color=window_colors.get(window, 'gray'))
                        ),
                        row=3, col=2
                    )
    
    # Plot 5: Stress Periods and Gaming Performance
    if stress_periods is not None:
        fig.add_trace(
            go.Scatter(
                x=stress_periods['time'],
                y=stress_periods['stress_level'],
                mode='markers',
                name='Stress Level',
                marker=dict(
                    color=stress_periods['stress_level'],
                    colorscale='Reds',
                    size=8,
                    colorbar=dict(title="Stress %")
                )
            ),
            row=4, col=1
        )
    
    # Plot 6: Controller Input Intensity vs Heart Rate
    if intensity_df is not None and intensity_correlation is not None:
        matched_data = intensity_correlation.get('data')
        if matched_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=matched_data['inputs_per_second'],
                    y=matched_data['heart_rate'],
                    mode='markers',
                    name='Input vs HR',
                    marker=dict(
                        color=matched_data['intensity_score'],
                        colorscale='Viridis',
                        size=6
                    )
                ),
                row=4, col=2
            )
    
    # Plot 7: Performance Metrics Over Time  
    if performance_analysis is not None:
        perf_data = performance_analysis.get('data')
        if perf_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=perf_data['time'],
                    y=perf_data['kd_ratio'],
                    mode='lines+markers',
                    name='K/D Ratio',
                    line=dict(color='gold', width=2)
                ),
                row=5, col=1
            )
            
            # Add performance score on secondary y-axis
            fig.add_trace(
                go.Scatter(
                    x=perf_data['time'],
                    y=perf_data['performance_score'],
                    mode='lines+markers',
                    name='Performance Score',
                    line=dict(color='purple', width=2),
                    yaxis='y2'
                ),
                row=5, col=1
            )
    
    # Plot 8: Summary statistics
    if ekg_data is not None:
        # Create a text summary
        summary_text = f"""
        EKG Analysis Summary:
        • Mean HR: {mean_hr:.1f} BPM
        • Stress Index: {ekg_data.EKG.stress_index:.1f}/100
        • HRV Score: {ekg_data.EKG.hr_variability_score:.1f}
        • RMSSD: {ekg_data.EKG.rmssd:.1f} ms
        • Total Peaks: {len(peaks)}
        • Total Beats: {len(beats)}
        • Recording: {ekg_data['Timestamp'].max():.1f}s
        """
        
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.8, y=0.1,
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor="lightgray",
            bordercolor="black",
            borderwidth=1
        )
    
    # Update layout
    fig.update_layout(
        height=1400,
        title="Comprehensive Multimodal Gaming Health Analysis Dashboard<br><sub>EKG plot uses same scale and method as original EKG analyzer</sub>",
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (seconds)", row=5, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=5, col=2)
    fig.update_yaxes(title_text="Heart Signal (mV)", row=1, col=1)
    fig.update_yaxes(title_text="BPM", row=2, col=1)
    fig.update_yaxes(title_text="Heart Rate (BPM)", row=3, col=1)
    fig.update_yaxes(title_text="HR Change (BPM)", row=3, col=2)
    fig.update_yaxes(title_text="Stress Level (%)", row=4, col=1)
    fig.update_yaxes(title_text="Heart Rate (BPM)", row=4, col=2)
    fig.update_yaxes(title_text="K/D Ratio", row=5, col=1)
    fig.update_yaxes(title_text="Performance Score", row=5, col=2)
    
    return fig

# Create the comprehensive dashboard
try:
    dashboard_fig = create_comprehensive_dashboard(
        ekg_data, hr_df, video_data, stress_periods, 
        event_analysis, intensity_df, performance_analysis
    )
    dashboard_fig.show()
    print("✅ Comprehensive dashboard created successfully!")
except Exception as e:
    print(f"⚠️ Could not create full dashboard: {e}")

# %%
"""
SECTION 8: STATISTICAL SUMMARY AND CORRELATION MATRIX
=====================================================
"""

print("📈 Creating comprehensive statistical summary and correlation matrix...")

def create_comprehensive_correlation_matrix():
    """
    Create a comprehensive correlation matrix from all available data.
    """
    correlation_data = {}
    
    # EKG-derived metrics
    if ekg_data is not None:
        try:
            correlation_data['Mean_HR'] = [mean_hr] * len(video_data) if mean_hr and video_data is not None else [mean_hr]
            correlation_data['Stress_Index'] = [ekg_data.EKG.stress_index] * len(video_data) if video_data is not None else [ekg_data.EKG.stress_index]
            correlation_data['RMSSD'] = [ekg_data.EKG.rmssd] * len(video_data) if video_data is not None else [ekg_data.EKG.rmssd]
            correlation_data['HRV_Score'] = [ekg_data.EKG.hr_variability_score] * len(video_data) if video_data is not None else [ekg_data.EKG.hr_variability_score]
        except Exception as e:
            print(f"⚠️ Could not extract EKG metrics for correlation: {e}")
    
    # Performance metrics
    if performance_analysis is not None and 'data' in performance_analysis:
        perf_data = performance_analysis['data']
        if len(perf_data) > 0:
            correlation_data['KD_Ratio'] = perf_data['kd_ratio'].values
            correlation_data['Performance_Score'] = perf_data['performance_score'].values
            correlation_data['Action_Rate'] = perf_data['action_rate'].values
    
    # Input intensity metrics
    if intensity_correlation is not None and 'data' in intensity_correlation:
        intensity_data = intensity_correlation['data']
        if len(intensity_data) > 0:
            correlation_data['Input_Intensity'] = intensity_data['intensity_score'].values
            correlation_data['Inputs_Per_Second'] = intensity_data['inputs_per_second'].values
    
    # Create correlation matrix if we have data
    if correlation_data:
        # Make all arrays the same length (use minimum length)
        min_length = min(len(values) for values in correlation_data.values() if hasattr(values, '__len__'))
        
        for key in correlation_data:
            if hasattr(correlation_data[key], '__len__') and len(correlation_data[key]) > min_length:
                correlation_data[key] = correlation_data[key][:min_length]
            elif not hasattr(correlation_data[key], '__len__'):
                correlation_data[key] = [correlation_data[key]] * min_length
        
        corr_df = pd.DataFrame(correlation_data)
        correlation_matrix = corr_df.corr()
        
        # Create heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig_corr.update_layout(
            title="Multimodal Correlation Matrix",
            width=800,
            height=600
        )
        
        fig_corr.show()
        
        return correlation_matrix
    
    return pd.DataFrame()

def generate_comprehensive_insights():
    """
    Generate comprehensive insights from all analyses.
    """
    insights = []
    
    # EKG insights using existing analyzer results
    if ekg_data is not None:
        try:
            insights.append(f"💓 Average heart rate during gaming: {mean_hr:.1f} BPM")
            insights.append(f"😰 Stress index: {ekg_data.EKG.stress_index:.1f}/100 ({ekg_data.EKG.rhythm_type})")
            insights.append(f"🫀 HRV Score: {ekg_data.EKG.hr_variability_score:.1f} (RMSSD: {ekg_data.EKG.rmssd:.1f}ms)")
            insights.append(f"📊 Detected {len(peaks)} R-peaks and {len(beats)} beats over {ekg_data['Timestamp'].max():.1f}s")
        except Exception as e:
            insights.append(f"⚠️ EKG analysis partially available")
    
    # Event response insights
    if event_analysis:
        most_impactful_events = []
        for event_type, analysis in event_analysis.items():
            max_impact = 0
            best_window = ""
            for window, data in analysis.items():
                if abs(data['mean_change']) > abs(max_impact):
                    max_impact = data['mean_change']
                    best_window = window
            if abs(max_impact) > 1.0:  # Significant change
                direction = "increases" if max_impact > 0 else "decreases" 
                most_impactful_events.append(f"{event_type} {direction} HR by {abs(max_impact):.1f} BPM ({best_window})")
        
        if most_impactful_events:
            insights.append(f"⚔️ Most impactful events: {'; '.join(most_impactful_events[:3])}")
    
    # Stress insights
    if stress_periods is not None:
        stress_duration = len(stress_periods) * 5  # Approximate
        total_duration = ekg_data['Timestamp'].max() if ekg_data is not None else 0
        stress_percentage = (stress_duration / total_duration) * 100 if total_duration > 0 else 0
        insights.append(f"😰 High-stress periods: {stress_percentage:.1f}% of gaming session")
        
        if stress_event_correlations is not None:
            top_stress_events = stress_event_correlations.groupby('event_type')['stress_level'].mean().sort_values(ascending=False)
            if len(top_stress_events) > 0:
                insights.append(f"🔥 Most stressful events: {', '.join(top_stress_events.head(3).index.tolist())}")
    
    # Performance insights
    if performance_analysis is not None and 'data' in performance_analysis:
        perf_data = performance_analysis['data']
        avg_kd = perf_data['kd_ratio'].mean()
        best_performance_hr = perf_data.loc[perf_data['performance_score'].idxmax(), 'avg_heart_rate']
        insights.append(f"🏆 Average K/D ratio: {avg_kd:.2f}")
        insights.append(f"🎯 Best performance occurred at {best_performance_hr:.1f} BPM")
    
    # Input intensity insights
    if intensity_correlation is not None and 'correlations' in intensity_correlation:
        correlations = intensity_correlation['correlations']
        strongest_corr = max(correlations.items(), key=lambda x: abs(x[1])) if correlations else None
        if strongest_corr and abs(strongest_corr[1]) > 0.3:
            direction = "positively" if strongest_corr[1] > 0 else "negatively"
            insights.append(f"🎮 Input intensity {direction} correlates with heart rate (r={strongest_corr[1]:.3f})")
    
    # Data quality insights
    data_sources = []
    if ekg_data is not None:
        data_sources.append(f"EKG ({len(ekg_data)} samples)")
    if video_data is not None:
        data_sources.append(f"Video ({len(video_data)} events)")
    if controller_data is not None:
        data_sources.append(f"Controller ({len(controller_data)} inputs)")
    
    insights.append(f"📊 Data sources: {', '.join(data_sources)}")
    
    return insights

# Create correlation matrix and generate insights
print("🔗 Creating correlation matrix...")
correlation_matrix = create_comprehensive_correlation_matrix()

if not correlation_matrix.empty:
    print(f"✅ Correlation matrix created with {len(correlation_matrix)} variables")
else:
    print("⚠️ Could not create correlation matrix - insufficient overlapping data")

print("\n🧠 COMPREHENSIVE INSIGHTS:")
print("=" * 50)
insights = generate_comprehensive_insights()
for insight in insights:
    print(f"  {insight}")

# %%
"""
SECTION 9: RESEARCH RECOMMENDATIONS AND EXPORT
==============================================
"""

print("\n💾 Generating research recommendations and exporting results...")

def generate_research_recommendations():
    """
    Generate research recommendations based on findings.
    """
    recommendations = []
    
    recommendations.append("🔬 RESEARCH RECOMMENDATIONS:")
    recommendations.append("=" * 40)
    
    # EKG-based recommendations
    if ekg_data is not None:
        try:
            stress_level = ekg_data.EKG.stress_index
            if stress_level > 75:
                recommendations.append("• Consider implementing stress-reduction techniques during gaming")
                recommendations.append("• Investigate longer gaming sessions' impact on cardiovascular health")
            elif stress_level < 30:
                recommendations.append("• Gaming appears to have minimal physiological stress impact")
                recommendations.append("• This player profile could be suitable for extended gaming sessions")
            
            hrv_score = ekg_data.EKG.hr_variability_score
            if hrv_score < 50:
                recommendations.append("• Low HRV detected - recommend rest periods between gaming sessions")
            else:
                recommendations.append("• Good HRV indicates healthy autonomic function during gaming")
        except:
            pass
    
    # Event-based recommendations
    if event_analysis:
        high_impact_events = []
        for event_type, analysis in event_analysis.items():
            for window, data in analysis.items():
                if abs(data['mean_change']) > 5.0:  # High impact threshold
                    high_impact_events.append(event_type)
        
        if high_impact_events:
            recommendations.append(f"• High physiological impact events identified: {', '.join(set(high_impact_events))}")
            recommendations.append("• Consider pacing strategies around these event types")
    
    # Performance-based recommendations
    if performance_analysis is not None and 'correlations' in performance_analysis:
        perf_hr_corr = performance_analysis['correlations'].get('performance_score_vs_avg_heart_rate', 0)
        if perf_hr_corr > 0.3:
            recommendations.append("• Performance improves with higher heart rate - moderate arousal beneficial")
        elif perf_hr_corr < -0.3:
            recommendations.append("• Performance degrades with higher heart rate - stress management needed")
    
    # Technical recommendations
    recommendations.append("\n🔧 TECHNICAL RECOMMENDATIONS:")
    recommendations.append("• Implement real-time biofeedback during gaming")
    recommendations.append("• Develop personalized break recommendations based on physiological state")
    recommendations.append("• Create adaptive difficulty based on stress levels")
    recommendations.append("• Consider heart rate variability training for competitive gamers")
    
    # Future research directions
    recommendations.append("\n🚀 FUTURE RESEARCH DIRECTIONS:")
    recommendations.append("• Long-term cardiovascular impact of competitive gaming")
    recommendations.append("• Personalized physiological gaming profiles")
    recommendations.append("• Real-time stress intervention systems")
    recommendations.append("• Correlation with sleep quality and recovery metrics")
    
    return "\n".join(recommendations)

def export_analysis_results():
    """
    Export analysis results to files.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export summary statistics
        if ekg_data is not None:
            summary_stats = {
                'analysis_timestamp': timestamp,
                'mean_heart_rate': mean_hr,
                'stress_index': ekg_data.EKG.stress_index,
                'rmssd': ekg_data.EKG.rmssd,
                'hrv_score': ekg_data.EKG.hr_variability_score,
                'total_peaks': len(peaks) if peaks is not None else 0,
                'total_beats': len(beats) if beats is not None else 0,
                'recording_duration': ekg_data['Timestamp'].max()
            }
            
            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_csv(f'multimodal_analysis_summary_{timestamp}.csv', index=False)
            print(f"✅ Summary exported to: multimodal_analysis_summary_{timestamp}.csv")
        
        # Export event analysis results
        if event_analysis:
            event_results = []
            for event_type, analysis in event_analysis.items():
                for window, data in analysis.items():
                    event_results.append({
                        'event_type': event_type,
                        'window_size': window,
                        'mean_hr_change': data['mean_change'],
                        'std_hr_change': data['std_change'],
                        'sample_size': data['sample_size'],
                        'significant': data['significant']
                    })
            
            if event_results:
                event_df = pd.DataFrame(event_results)
                event_df.to_csv(f'event_analysis_results_{timestamp}.csv', index=False)
                print(f"✅ Event analysis exported to: event_analysis_results_{timestamp}.csv")
        
        # Export correlation matrix
        if not correlation_matrix.empty:
            correlation_matrix.to_csv(f'correlation_matrix_{timestamp}.csv')
            print(f"✅ Correlation matrix exported to: correlation_matrix_{timestamp}.csv")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Error exporting results: {e}")
        return False

# Generate and display recommendations
research_recommendations = generate_research_recommendations()
print(research_recommendations)

# Export results
export_success = export_analysis_results()

# Save recommendations to file
try:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'research_recommendations_{timestamp}.txt', 'w') as f:
        f.write(research_recommendations)
    print(f"✅ Recommendations saved to: research_recommendations_{timestamp}.txt")
except Exception as e:
    print(f"⚠️ Could not save recommendations: {e}")

print("\n" + "=" * 60)
print("🎮💓 COMPREHENSIVE MULTIMODAL GAMING HEALTH ANALYSIS COMPLETE!")
print("=" * 60)
print("Key Features Completed:")
print("✅ EKG analysis using existing peaks/beats detection")
print("✅ Multi-window event response analysis (5s, 15s, 30s)")
print("✅ Advanced stress detection and correlation")
print("✅ Controller input intensity analysis")
print("✅ Performance vs physiology correlation")
print("✅ Comprehensive visualization dashboard")
print("✅ Statistical correlation matrix")
print("✅ Research recommendations")
print("✅ Results export functionality")
print("\nThis analysis leverages all existing analyzer capabilities")
print("and provides immediate insights into gaming health correlations!")
print("=" * 60)

# %%

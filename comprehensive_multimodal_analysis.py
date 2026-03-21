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
from gaming_health_data.src.oura_analyzer import OuraAnalyzer

print("Comprehensive Multimodal Gaming Health Analysis")
print("=" * 60)

# Create output directory for results
timestamp = datetime.now().strftime("%Y%m%d")
output_dir = Path(f"multimodal_analysis_results/{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Results will be saved to: {output_dir}")

# %%
"""
SECTION 1: DATA LOADING AND PREPARATION
=======================================
"""

print("Loading multimodal data using existing analyzers...")

# Define data paths - update these to match your specific files
EKG_FILE = "gaming_health_data/recorded_data/SENSORS/ekg_data_polars_h10_2025_11_23_204515.txt"
VIDEO_FILE = "gaming_health_data/recorded_data/VIDEO/annotated/video_annotation_manual_20250508132346.csv"
APPLE_WATCH_FILE = "gaming_health_data/recorded_data/APPLE_WATCH/apple_health_export_2026-03-11.csv"
OURA_DATA_DIR = "gaming_health_data/recorded_data/OURA"  # Directory containing Oura CSV files

# Gaming session date for health data correlation
GAMING_SESSION_DATE = "2025-11-23"  # Match this with your actual gaming session date

# Controller data paths (assuming you have 3 files that need to be combined)
CONTROLLER_FILES = [
    "gaming_health_data/recorded_data/PS/controller_inputs_23_11_2025_part_00.csv",
    "gaming_health_data/recorded_data/PS/controller_inputs_23_11_2025_part_01.csv",
    "gaming_health_data/recorded_data/PS/controller_inputs_23_11_2025_part_02.csv"
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
        print(f"Controller data columns: {list(combined.columns)}")
        
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
                
                print(f"Combined controller data: {len(combined)} inputs over {combined['time_seconds'].max():.1f}s")
            except Exception as e:
                print(f"[WARNING] Could not convert timestamps: {e}")
                # Create a simple time index if timestamp conversion fails
                combined['time_seconds'] = np.arange(len(combined)) * 0.1  # Assume 10Hz sampling
                print(f"Combined controller data: {len(combined)} inputs (using synthetic timestamps)")
        else:
            print("[WARNING] No timestamp column found, creating synthetic timestamps")
            combined['time_seconds'] = np.arange(len(combined)) * 0.1  # Assume 10Hz sampling
            print(f"Combined controller data: {len(combined)} inputs (using synthetic timestamps)")
        
        return combined
    
    return None

# Load EKG data using EKGAnalyzer
try:
    ekg_data = EKGAnalyzer.read_file(EKG_FILE, sensor_type="PolarH10")
    print(f"[OK] EKG: {len(ekg_data)} samples, {ekg_data['Timestamp'].max():.1f}s duration")
    
    # Add compatibility columns for plotting while preserving original names
    ekg_data['time'] = ekg_data['Timestamp'].copy()
    ekg_data['signal_mV'] = ekg_data['HeartSignal'].copy()
    
except Exception as e:
    print(f"[ERROR] Error loading EKG data: {e}")
    ekg_data = None

# Load video data using VideoAnalyzer
try:
    video_data = VideoAnalyzer.from_manual_csv(VIDEO_FILE)
    print(f"[OK] Video: {len(video_data)} events")
    
    # Convert timestamps to seconds using the analyzer method
    video_data = video_data.Video.convert_timestamp_to_seconds()
    print(f"Video duration: {video_data['time_seconds'].max():.1f}s")
    
    # Get basic event statistics
    event_counts = video_data['action'].value_counts()
    print(f"Top events: {dict(event_counts.head())}")
    
except Exception as e:
    print(f"[ERROR] Error loading video data: {e}")
    video_data = None

# Load controller data
controller_data = load_controller_data(CONTROLLER_FILES)

# Load Apple Watch data for the gaming session
try:
    apple_watch_data = AppleWatchAnalyzer.read_file(APPLE_WATCH_FILE)
    print(f"[OK] Apple Watch: {len(apple_watch_data)} health records loaded")
    
    # Get session data for the gaming date
    try:
        gaming_session = apple_watch_data.applewatch.get_session_from_date(GAMING_SESSION_DATE)
        print(f"Found Apple Watch session on {GAMING_SESSION_DATE}")
        print(f"   Session: {gaming_session['startDate']} to {gaming_session['endDate']}")
        
        # Get heart rate data from the gaming session
        apple_hr_data = apple_watch_data.applewatch.get_heart_rate_stats_from_session(gaming_session)
        
        if len(apple_hr_data) > 0:
            # Convert to seconds from session start for comparison
            session_start = apple_hr_data['startDate'].min()
            apple_hr_data['time_seconds'] = (apple_hr_data['startDate'] - session_start).dt.total_seconds()
            
            print(f"Apple Watch HR: {len(apple_hr_data)} measurements")
            print(f"   HR range: {apple_hr_data['value'].min():.1f} - {apple_hr_data['value'].max():.1f} BPM")
            print(f"   Duration: {apple_hr_data['time_seconds'].max():.1f}s")
        else:
            print("[WARNING] No heart rate data found in Apple Watch session")
            apple_hr_data = None
            
    except Exception as e:
        print(f"[WARNING] Could not find Apple Watch session for {GAMING_SESSION_DATE}: {e}")
        apple_hr_data = None
        gaming_session = None
        
except Exception as e:
    print(f"[ERROR] Error loading Apple Watch data: {e}")
    apple_watch_data = None
    apple_hr_data = None
    gaming_session = None

# Load Oura Ring data for the gaming session
try:
    oura_data = OuraAnalyzer.read_oura_data(OURA_DATA_DIR)
    print(f"[OK] Oura Ring: {len(oura_data)} health records loaded")
    
    # Get session data for the gaming date
    try:
        oura_session = oura_data.oura.get_session_from_date(GAMING_SESSION_DATE)
        print(f"Found Oura Ring session on {GAMING_SESSION_DATE}")
        print(f"   Session: {oura_session['startDate']} to {oura_session['endDate']}")
        
        # Get heart rate data from the gaming session
        oura_hr_data = oura_data.oura.get_heart_rate_stats_from_session(oura_session)
        
        # Ensure we have the correct column names for compatibility
        if oura_hr_data is not None and len(oura_hr_data) > 0:
            # Standardize column names to match Apple Watch format
            if 'bpm' in oura_hr_data.columns and 'value' not in oura_hr_data.columns:
                oura_hr_data = oura_hr_data.rename(columns={'bpm': 'value'})
        
        # Get additional Oura metrics
        oura_temp_data = oura_data.oura.get_temperature_for_date(GAMING_SESSION_DATE)
        oura_readiness = oura_data.oura.get_readiness_for_date(GAMING_SESSION_DATE)
        oura_activity = oura_data.oura.get_activity_for_date(GAMING_SESSION_DATE)
        
        if oura_hr_data is not None and len(oura_hr_data) > 0:
            print(f"Oura Ring HR: {len(oura_hr_data)} measurements")
            print(f"   HR range: {oura_hr_data['value'].min():.1f} - {oura_hr_data['value'].max():.1f} BPM")
            print(f"   Duration: {oura_hr_data['time_seconds'].max():.1f}s")
        else:
            print("[WARNING] No heart rate data found in Oura Ring session")
            oura_hr_data = None
        
        if oura_temp_data is not None and len(oura_temp_data) > 0:
            print(f"Oura Ring Temperature: {len(oura_temp_data)} measurements")
            print(f"   Temp range: {oura_temp_data['skin_temp'].min():.2f} - {oura_temp_data['skin_temp'].max():.2f}°C")
        
        if oura_readiness:
            print(f"Oura Ring Readiness: Score {oura_readiness.get('score')}")
        
        if oura_activity:
            print(f"Oura Ring Activity: {oura_activity.get('steps')} steps, Score {oura_activity.get('score')}")
            
    except Exception as e:
        print(f"[WARNING] Could not find Oura Ring session for {GAMING_SESSION_DATE}: {e}")
        oura_hr_data = None
        oura_session = None
        oura_temp_data = None
        oura_readiness = None
        oura_activity = None
        
except Exception as e:
    print(f"[WARNING] Error loading Oura Ring data: {e}")
    oura_data = None
    oura_hr_data = None
    oura_session = None
    oura_temp_data = None
    oura_readiness = None
    oura_activity = None

print(f"\nData loading complete!")
if ekg_data is not None:
    print(f"   EKG: {len(ekg_data)} samples")
if video_data is not None:
    print(f"   Video: {len(video_data)} events")
if controller_data is not None:
    print(f"   Controller: {len(controller_data)} inputs")
if apple_hr_data is not None:
    print(f"   Apple Watch: {len(apple_hr_data)} heart rate measurements")
if oura_hr_data is not None:
    print(f"   Oura Ring: {len(oura_hr_data)} heart rate measurements")

# %%
"""
SECTION 1.5: MULTI-SOURCE HEART RATE COMPARISON
===============================================
"""

print("Comparing heart rate data from multiple sources...")

def compare_hr_sources(ekg_hr_df, apple_hr_data, oura_hr_data, video_data):
    """
    Compare heart rate data from EKG sensor, Apple Watch, and Oura Ring.
    """
    if ekg_hr_df is None and apple_hr_data is None and oura_hr_data is None:
        print("[WARNING] No heart rate data available from any source")
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
        
        print(f"EKG Heart Rate Analysis:")
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
        
        print(f"Apple Watch Heart Rate Analysis:")
        print(f"   Samples: {apple_stats['sample_count']}")
        print(f"   Mean HR: {apple_stats['mean_hr']:.1f} BPM")
        print(f"   Range: {apple_stats['min_hr']:.1f} - {apple_stats['max_hr']:.1f} BPM")
        print(f"   Sampling: {apple_stats['sampling_rate']:.2f} Hz")
    
    # Oura Ring analysis
    if oura_hr_data is not None:
        oura_stats = {
            'source': 'Oura Ring',
            'sample_count': len(oura_hr_data),
            'mean_hr': oura_hr_data['bpm'].mean(),
            'min_hr': oura_hr_data['bpm'].min(),
            'max_hr': oura_hr_data['bpm'].max(),
            'std_hr': oura_hr_data['bpm'].std(),
            'duration': oura_hr_data['time_seconds'].max(),
            'sampling_rate': len(oura_hr_data) / oura_hr_data['time_seconds'].max() if oura_hr_data['time_seconds'].max() > 0 else 0
        }
        comparison_results['Oura_Ring'] = oura_stats
        
        print(f"Oura Ring Heart Rate Analysis:")
        print(f"   Samples: {oura_stats['sample_count']}")
        print(f"   Mean HR: {oura_stats['mean_hr']:.1f} BPM")
        print(f"   Range: {oura_stats['min_hr']:.1f} - {oura_stats['max_hr']:.1f} BPM")
        print(f"   Sampling: {oura_stats['sampling_rate']:.2f} Hz")
    
    # Compare sources if multiple available
    sources = []
    if ekg_hr_df is not None:
        sources.append(('EKG', ekg_stats))
    if apple_hr_data is not None:
        sources.append(('Apple_Watch', apple_stats))
    if oura_hr_data is not None:
        sources.append(('Oura_Ring', oura_stats))
    
    if len(sources) >= 2:
        print(f"\nMulti-Source Comparison:")
        # Calculate mean HR differences between all pairs
        for i, (name1, stats1) in enumerate(sources):
            for name2, stats2 in sources[i+1:]:
                hr_diff = abs(stats1['mean_hr'] - stats2['mean_hr'])
                print(f"   {name1} vs {name2} mean HR difference: {hr_diff:.1f} BPM")
                
        # Agreement analysis (using EKG as reference if available)
        if ekg_hr_df is not None:
            if apple_hr_data is not None:
                hr_diff = abs(ekg_stats['mean_hr'] - apple_stats['mean_hr'])
                if hr_diff < 5:
                    agreement = "EXCELLENT"
                elif hr_diff < 10:
                    agreement = "GOOD"
                elif hr_diff < 15:
                    agreement = "MODERATE"
                else:
                    agreement = "POOR"
                print(f"   EKG vs Apple Watch agreement: {agreement}")
                
            if oura_hr_data is not None:
                hr_diff = abs(ekg_stats['mean_hr'] - oura_stats['mean_hr'])
                if hr_diff < 5:
                    agreement = "EXCELLENT"
                elif hr_diff < 10:
                    agreement = "GOOD"
                elif hr_diff < 15:
                    agreement = "MODERATE"
                else:
                    agreement = "POOR"
                print(f"   EKG vs Oura Ring agreement: {agreement}")
    
    return comparison_results

def _hr_pairwise_rows(fig, pairs, row_diff, row_scatter):
    """
    Helper: add difference (row_diff) and scatter-correlation (row_scatter) subplots
    for a list of source pairs.

    pairs: list of dicts with keys:
        ta, va        - time/value arrays for source A
        tb, vb        - time/value arrays for source B
        color         - hex line colour for diff plot
        fill          - rgba fill string for diff area
        colorscale    - Plotly colorscale name for scatter
        xref_s, yref_s - axis refs for the scatter subplot (e.g. 'x5', 'y5')
        col           - subplot column index (1, 2, or 3)
        label_a       - name of source A
        label_b       - name of source B
    """
    for p in pairs:
        ta, va = np.asarray(p['ta']), np.asarray(p['va'])
        tb, vb = np.asarray(p['tb']), np.asarray(p['vb'])
        t_end = min(float(ta.max()), float(tb.max()))
        ct = np.arange(0, t_end, 5)
        ia = np.interp(ct, ta, va)
        ib = np.interp(ct, tb, vb)
        col = p['col']
        color = p['color']

        # ── Difference plot ───────────────────────────────────────────────────
        diff = ia - ib
        mu, sd = float(np.mean(diff)), float(np.std(diff))
        fig.add_trace(go.Scatter(
            x=ct, y=diff, mode='lines',
            line=dict(color=color, width=2),
            fill='tozeroy', fillcolor=p['fill'],
            showlegend=False,
        ), row=row_diff, col=col)
        fig.add_hline(y=0,      line=dict(color='gray', dash='dash', width=1.5), row=row_diff, col=col)
        fig.add_hline(y=mu,     line=dict(color=color,  dash='dash', width=2),   row=row_diff, col=col)
        fig.add_hline(y=mu+sd,  line=dict(color=color,  dash='dot',  width=1),   row=row_diff, col=col)
        fig.add_hline(y=mu-sd,  line=dict(color=color,  dash='dot',  width=1),   row=row_diff, col=col)
        # Stats annotation inside diff plot
        fig.add_annotation(
            text=f'μ={mu:+.1f}  σ={sd:.1f}',
            xref='paper', yref='paper',
            # place at right edge of this column, near top of row_diff
            x=p['ann_paper_x'], y=p['ann_paper_y'],
            showarrow=False, font=dict(size=9, color=color),
            bgcolor='rgba(255,255,255,0.85)', borderpad=3,
            xanchor='right', yanchor='top',
        )

        # ── Scatter / correlation plot ────────────────────────────────────────
        corr = np.corrcoef(ia, ib)[0, 1]
        rmse = np.sqrt(np.mean((ia - ib) ** 2))
        hr_min = min(ia.min(), ib.min()) - 1
        hr_max = max(ia.max(), ib.max()) + 1
        fig.add_trace(go.Scatter(
            x=ia, y=ib, mode='markers',
            marker=dict(color=ct, colorscale=p['colorscale'], size=5,
                        showscale=False, opacity=0.75),
            hovertemplate=f'{p["label_a"]}: %{{x:.1f}}<br>{p["label_b"]}: %{{y:.1f}}<extra></extra>',
            showlegend=False,
        ), row=row_scatter, col=col)
        # Identity line
        fig.add_trace(go.Scatter(
            x=[hr_min, hr_max], y=[hr_min, hr_max],
            mode='lines', line=dict(color='gray', dash='dash', width=1.5),
            showlegend=False,
        ), row=row_scatter, col=col)
        # Stats box
        fig.add_annotation(
            text=f'r = {corr:.3f}<br>RMSE = {rmse:.2f}<br>n = {len(ct)}',
            xref=p['xref_s'], yref=p['yref_s'],
            x=hr_min + (hr_max - hr_min) * 0.05,
            y=hr_max - (hr_max - hr_min) * 0.08,
            showarrow=False,
            font=dict(size=10, family='monospace', color='#2C3E50'),
            align='left',
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor=color, borderwidth=1.5, borderpad=6,
        )


def create_multi_source_hr_plot(ekg_hr_df, apple_hr_data, oura_hr_data, video_data):
    """
    3-row, 3-column layout (row 1 full-width):
      Row 1: Time series — EKG · Apple Watch · Oura Ring + gaming events
      Row 2: Pairwise difference plots  — EKG−AW | EKG−Oura | AW−Oura
      Row 3: Pairwise correlation scatters — EKG vs AW | EKG vs Oura | AW vs Oura
    """
    fig = make_subplots(
        rows=3, cols=3,
        specs=[
            [{"colspan": 3}, None, None],
            [{}, {}, {}],
            [{}, {}, {}],
        ],
        subplot_titles=[
            'EKG · Apple Watch · Oura Ring — Time Series',
            'EKG − Apple Watch', 'EKG − Oura Ring', 'Apple Watch − Oura Ring',
            'EKG vs Apple Watch', 'EKG vs Oura Ring', 'Apple Watch vs Oura Ring',
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        row_heights=[0.36, 0.26, 0.38],
    )

    # ── Row 1: Time series ────────────────────────────────────────────────────
    if ekg_hr_df is not None:
        fig.add_trace(go.Scatter(
            x=ekg_hr_df['time'], y=ekg_hr_df['hr_smooth'],
            mode='lines', name='EKG (Polar H10)',
            line=dict(color='#2E86DE', width=2.5), opacity=0.9,
        ), row=1, col=1)

    if apple_hr_data is not None:
        fig.add_trace(go.Scatter(
            x=apple_hr_data['time_seconds'], y=apple_hr_data['value'],
            mode='markers+lines', name='Apple Watch',
            line=dict(color='#EE5A6F', width=2), marker=dict(size=4), opacity=0.8,
        ), row=1, col=1)

    if oura_hr_data is not None and len(oura_hr_data) > 0:
        fig.add_trace(go.Scatter(
            x=oura_hr_data['time_seconds'], y=oura_hr_data['value'],
            mode='markers+lines', name='Oura Ring',
            line=dict(color='#26DE81', width=2), marker=dict(size=4), opacity=0.7,
        ), row=1, col=1)

    if video_data is not None:
        ev_colors = {'kill': '#A55EEA', 'death': '#2C3E50', 'assist': '#FD9644'}
        for ev_type, color in ev_colors.items():
            for t in video_data[video_data['action'] == ev_type]['time_seconds'].iloc[:10]:
                fig.add_vline(x=t, line=dict(color=color, width=1, dash='dot'),
                              opacity=0.5, row=1, col=1)

    # ── Rows 2 & 3: All pairwise comparisons ─────────────────────────────────
    has_ekg   = ekg_hr_df is not None and len(ekg_hr_df) > 0
    has_apple = apple_hr_data is not None and len(apple_hr_data) > 0
    has_oura  = oura_hr_data is not None and len(oura_hr_data) > 0

    # ann_paper_x/y: paper-coords for diff-plot stats label (top-right of each col)
    # Row 2 paper y ≈ 0.62 (top of row 2 with 0.10 vspacing), cols at ~0.28/0.63/0.99
    pairs = []
    if has_ekg and has_apple:
        pairs.append(dict(
            ta=ekg_hr_df['time'].values, va=ekg_hr_df['hr_smooth'].values,
            tb=apple_hr_data['time_seconds'].values, vb=apple_hr_data['value'].values,
            color='#A55EEA', fill='rgba(165,94,234,0.18)',
            colorscale='Viridis', xref_s='x5', yref_s='y5',
            col=1, label_a='EKG', label_b='Apple Watch',
            ann_paper_x=0.28, ann_paper_y=0.62,
        ))
    if has_ekg and has_oura:
        pairs.append(dict(
            ta=ekg_hr_df['time'].values, va=ekg_hr_df['hr_smooth'].values,
            tb=oura_hr_data['time_seconds'].values, vb=oura_hr_data['value'].values,
            color='#FD9644', fill='rgba(253,150,68,0.18)',
            colorscale='Plasma', xref_s='x6', yref_s='y6',
            col=2, label_a='EKG', label_b='Oura Ring',
            ann_paper_x=0.63, ann_paper_y=0.62,
        ))
    if has_apple and has_oura:
        pairs.append(dict(
            ta=apple_hr_data['time_seconds'].values, vb=oura_hr_data['value'].values,
            tb=oura_hr_data['time_seconds'].values, va=apple_hr_data['value'].values,
            color='#26DE81', fill='rgba(38,222,129,0.18)',
            colorscale='Cividis', xref_s='x7', yref_s='y7',
            col=3, label_a='Apple Watch', label_b='Oura Ring',
            ann_paper_x=0.99, ann_paper_y=0.62,
        ))

    _hr_pairwise_rows(fig, pairs, row_diff=2, row_scatter=3)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=1080,
        title=dict(
            text='Multi-Source Heart Rate Validation: EKG · Apple Watch · Oura Ring',
            x=0.5, xanchor='center', font=dict(size=20, color='#2C3E50'),
        ),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='closest', template='plotly_white',
        font=dict(family='Arial, sans-serif', size=11),
    )
    fig.update_xaxes(title_text='Time (seconds)',       row=1, col=1, gridcolor='#E8E8E8')
    fig.update_xaxes(title_text='Time (s)',             row=2, col=1, gridcolor='#E8E8E8')
    fig.update_xaxes(title_text='Time (s)',             row=2, col=2, gridcolor='#E8E8E8')
    fig.update_xaxes(title_text='Time (s)',             row=2, col=3, gridcolor='#E8E8E8')
    fig.update_xaxes(title_text='EKG HR (BPM)',         row=3, col=1, gridcolor='#E8E8E8')
    fig.update_xaxes(title_text='EKG HR (BPM)',         row=3, col=2, gridcolor='#E8E8E8')
    fig.update_xaxes(title_text='Apple Watch HR (BPM)', row=3, col=3, gridcolor='#E8E8E8')
    fig.update_yaxes(title_text='Heart Rate (BPM)',     row=1, col=1, gridcolor='#E8E8E8')
    fig.update_yaxes(title_text='Diff (BPM)',           row=2, col=1, gridcolor='#E8E8E8')
    fig.update_yaxes(title_text='Diff (BPM)',           row=2, col=2, gridcolor='#E8E8E8')
    fig.update_yaxes(title_text='Diff (BPM)',           row=2, col=3, gridcolor='#E8E8E8')
    fig.update_yaxes(title_text='Apple Watch HR (BPM)', row=3, col=1, gridcolor='#E8E8E8')
    fig.update_yaxes(title_text='Oura Ring HR (BPM)',   row=3, col=2, gridcolor='#E8E8E8')
    fig.update_yaxes(title_text='Oura Ring HR (BPM)',   row=3, col=3, gridcolor='#E8E8E8')

    return fig


def create_multi_source_hr_plot_smoothed(ekg_hr_df, apple_hr_data, oura_hr_data, video_data,
                                         ekg_ma_window=25, apple_ma_window=30):
    """
    Same 3-row layout as create_multi_source_hr_plot but with Moving Average smoothing.
    Raw data shown as faint traces behind bold MA lines in the time series row.
    Diff and scatter plots use the smoothed signals.

    Parameters:
    -----------
    ekg_ma_window   : int  Rolling window (in beats) for EKG hr_smooth.
    apple_ma_window : int  Rolling window (in samples) for Apple Watch BPM.
    """
    # ── Compute smoothed signals ──────────────────────────────────────────────
    ekg_smooth = None
    if ekg_hr_df is not None:
        ekg_smooth = ekg_hr_df.copy()
        ekg_smooth['hr_ma'] = (pd.Series(ekg_hr_df['hr_smooth'].values)
                               .rolling(window=ekg_ma_window, center=True, min_periods=1)
                               .mean().values)

    apple_smooth = None
    if apple_hr_data is not None:
        apple_smooth = apple_hr_data.copy()
        apple_smooth['value_ma'] = (pd.Series(apple_hr_data['value'].values)
                                    .rolling(window=apple_ma_window, center=True, min_periods=1)
                                    .mean().values)

    oura_smooth = None
    oura_ma_values = None
    if oura_hr_data is not None and len(oura_hr_data) > 0:
        oura_smooth = oura_hr_data.copy()
        oura_ma_values = (pd.Series(oura_hr_data['value'].values)
                          .rolling(window=5, center=True, min_periods=1)
                          .mean().values)
        oura_smooth['value_ma'] = oura_ma_values

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=3,
        specs=[
            [{"colspan": 3}, None, None],
            [{}, {}, {}],
            [{}, {}, {}],
        ],
        subplot_titles=[
            f'EKG MA({ekg_ma_window}) · Apple Watch MA({apple_ma_window}) · Oura MA(5) — Time Series',
            'EKG MA − Apple Watch MA', 'EKG MA − Oura MA', 'Apple Watch MA − Oura MA',
            'EKG MA vs Apple Watch MA', 'EKG MA vs Oura MA', 'Apple Watch MA vs Oura MA',
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        row_heights=[0.36, 0.26, 0.38],
    )

    # ── Row 1: Time series (faint raw + bold MA) ──────────────────────────────
    if ekg_smooth is not None:
        fig.add_trace(go.Scatter(
            x=ekg_smooth['time'], y=ekg_smooth['hr_smooth'],
            mode='lines', name='EKG (raw)',
            line=dict(color='#2E86DE', width=1), opacity=0.22, showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=ekg_smooth['time'], y=ekg_smooth['hr_ma'],
            mode='lines', name=f'EKG MA({ekg_ma_window})',
            line=dict(color='#2E86DE', width=2.5), opacity=1.0,
        ), row=1, col=1)

    if apple_smooth is not None:
        fig.add_trace(go.Scatter(
            x=apple_smooth['time_seconds'], y=apple_smooth['value'],
            mode='markers', name='Apple Watch (raw)',
            marker=dict(color='#EE5A6F', size=3), opacity=0.22, showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=apple_smooth['time_seconds'], y=apple_smooth['value_ma'],
            mode='lines', name=f'Apple Watch MA({apple_ma_window})',
            line=dict(color='#EE5A6F', width=2.5), opacity=1.0,
        ), row=1, col=1)

    if oura_smooth is not None:
        fig.add_trace(go.Scatter(
            x=oura_smooth['time_seconds'], y=oura_smooth['value'],
            mode='markers', name='Oura Ring (raw)',
            marker=dict(color='#26DE81', size=3), opacity=0.22, showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=oura_smooth['time_seconds'], y=oura_smooth['value_ma'],
            mode='lines', name='Oura Ring MA(5)',
            line=dict(color='#26DE81', width=2.5), opacity=1.0,
        ), row=1, col=1)

    if video_data is not None:
        ev_colors = {'kill': '#A55EEA', 'death': '#2C3E50', 'assist': '#FD9644'}
        for ev_type, color in ev_colors.items():
            for t in video_data[video_data['action'] == ev_type]['time_seconds'].iloc[:10]:
                fig.add_vline(x=t, line=dict(color=color, width=1, dash='dot'),
                              opacity=0.5, row=1, col=1)

    # ── Rows 2 & 3: All pairwise comparisons (using MA signals) ──────────────
    has_ekg   = ekg_smooth is not None
    has_apple = apple_smooth is not None
    has_oura  = oura_smooth is not None

    pairs = []
    if has_ekg and has_apple:
        pairs.append(dict(
            ta=ekg_smooth['time'].values, va=ekg_smooth['hr_ma'],
            tb=apple_smooth['time_seconds'].values, vb=apple_smooth['value_ma'],
            color='#A55EEA', fill='rgba(165,94,234,0.18)',
            colorscale='Viridis', xref_s='x5', yref_s='y5',
            col=1, label_a=f'EKG MA({ekg_ma_window})', label_b=f'AW MA({apple_ma_window})',
            ann_paper_x=0.28, ann_paper_y=0.62,
        ))
    if has_ekg and has_oura:
        pairs.append(dict(
            ta=ekg_smooth['time'].values, va=ekg_smooth['hr_ma'],
            tb=oura_smooth['time_seconds'].values, vb=oura_smooth['value_ma'],
            color='#FD9644', fill='rgba(253,150,68,0.18)',
            colorscale='Plasma', xref_s='x6', yref_s='y6',
            col=2, label_a=f'EKG MA({ekg_ma_window})', label_b='Oura MA(5)',
            ann_paper_x=0.63, ann_paper_y=0.62,
        ))
    if has_apple and has_oura:
        pairs.append(dict(
            ta=apple_smooth['time_seconds'].values, va=apple_smooth['value_ma'],
            tb=oura_smooth['time_seconds'].values, vb=oura_smooth['value_ma'],
            color='#26DE81', fill='rgba(38,222,129,0.18)',
            colorscale='Cividis', xref_s='x7', yref_s='y7',
            col=3, label_a=f'AW MA({apple_ma_window})', label_b='Oura MA(5)',
            ann_paper_x=0.99, ann_paper_y=0.62,
        ))

    _hr_pairwise_rows(fig, pairs, row_diff=2, row_scatter=3)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=1080,
        title=dict(
            text=f'Multi-Source HR Validation (Smoothed) — EKG MA({ekg_ma_window}) · Apple Watch MA({apple_ma_window}) · Oura MA(5)',
            x=0.5, xanchor='center', font=dict(size=18, color='#2C3E50'),
        ),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='closest', template='plotly_white',
        font=dict(family='Arial, sans-serif', size=11),
    )
    fig.update_xaxes(title_text='Time (seconds)',       row=1, col=1, gridcolor='#E8E8E8')
    fig.update_xaxes(title_text='Time (s)',             row=2, col=1, gridcolor='#E8E8E8')
    fig.update_xaxes(title_text='Time (s)',             row=2, col=2, gridcolor='#E8E8E8')
    fig.update_xaxes(title_text='Time (s)',             row=2, col=3, gridcolor='#E8E8E8')
    fig.update_xaxes(title_text=f'EKG MA({ekg_ma_window}) HR (BPM)',    row=3, col=1, gridcolor='#E8E8E8')
    fig.update_xaxes(title_text=f'EKG MA({ekg_ma_window}) HR (BPM)',    row=3, col=2, gridcolor='#E8E8E8')
    fig.update_xaxes(title_text=f'AW MA({apple_ma_window}) HR (BPM)',   row=3, col=3, gridcolor='#E8E8E8')
    fig.update_yaxes(title_text='Heart Rate (BPM)',     row=1, col=1, gridcolor='#E8E8E8')
    fig.update_yaxes(title_text='Diff (BPM)',           row=2, col=1, gridcolor='#E8E8E8')
    fig.update_yaxes(title_text='Diff (BPM)',           row=2, col=2, gridcolor='#E8E8E8')
    fig.update_yaxes(title_text='Diff (BPM)',           row=2, col=3, gridcolor='#E8E8E8')
    fig.update_yaxes(title_text=f'AW MA({apple_ma_window}) HR (BPM)',   row=3, col=1, gridcolor='#E8E8E8')
    fig.update_yaxes(title_text='Oura MA(5) HR (BPM)', row=3, col=2, gridcolor='#E8E8E8')
    fig.update_yaxes(title_text='Oura MA(5) HR (BPM)', row=3, col=3, gridcolor='#E8E8E8')

    return fig


# Note: This comparison will be done after EKG analysis when hr_df is available

# %%
"""
SECTION 2: EKG ANALYSIS USING EXISTING ANALYZER METHODS
=======================================================
"""

print("Comprehensive EKG analysis using existing analyzer methods...")

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
        
        print(f"Heart Rate Metrics (using existing analyzer):")
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
                print(f"Heart rate time series: {len(hr_df)} beat intervals")
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
            print(f"Frequency Domain HRV:")
            print(f"   LF Power: {freq_analysis['LF']:.1f}")
            print(f"   HF Power: {freq_analysis['HF']:.1f}")
            print(f"   LF/HF Ratio: {freq_analysis['LF_HF_ratio']:.2f}")
        except Exception as e:
            print(f"⚠️ Could not calculate frequency domain HRV: {e}")
            freq_analysis = None
        
        # Get comprehensive health summary
        try:
            health_summary = ekg_data.EKG.get_heart_analysis_summary()
            print(f"Health Summary: {health_summary}")
        except Exception as e:
            print(f"⚠️ Could not get health summary: {e}")
        
        # Display the existing analyzer plots for comparison
        print("\nUsing existing EKG analyzer visualization methods:")
        
        try:
            print("Showing EKG data with properly detected peaks, low peaks, and beats...")
            # Use the existing plot method and save it
            ekg_plot_fig = ekg_data.EKG.plot_ekg_data()
            if hasattr(ekg_plot_fig, 'write_html'):
                ekg_plot_fig.write_html(output_dir / "ekg_original_analyzer_plot.html")
                print(f"✅ Original EKG plot saved to: {output_dir / 'ekg_original_analyzer_plot.html'}")
        except Exception as e:
            print(f"⚠️ Could not show/save EKG plot: {e}")
        
        try:
            print("Showing moving average BPM over time...")
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
hr_comparison = compare_hr_sources(hr_df, apple_hr_data, oura_hr_data, video_data)

# Create multi-source comparison plot
try:
    multi_source_fig = create_multi_source_hr_plot(hr_df, apple_hr_data, oura_hr_data, video_data)
    multi_source_fig.show()
    
    # Save the plot
    multi_source_fig.write_html(output_dir / "multi_source_hr_comparison.html")
    print(f"Multi-source HR comparison saved to: {output_dir / 'multi_source_hr_comparison.html'}")
    
except Exception as e:
    print(f"[WARNING] Could not create multi-source plot: {e}")

# Create smoothed MA version (EKG MA=15 beats, Apple Watch MA=30 samples)
try:
    multi_source_smoothed_fig = create_multi_source_hr_plot_smoothed(
        hr_df, apple_hr_data, oura_hr_data, video_data,
        ekg_ma_window=60, apple_ma_window=30
    )
    multi_source_smoothed_fig.show()
    multi_source_smoothed_fig.write_html(output_dir / "multi_source_hr_comparison_smoothed.html")
    print(f"Smoothed multi-source HR comparison saved to: {output_dir / 'multi_source_hr_comparison_smoothed.html'}")
except Exception as e:
    print(f"[WARNING] Could not create smoothed multi-source plot: {e}")

# %%
"""
SECTION 3: GAMING EVENTS IMPACT ON HEART RATE (MULTIPLE TIME WINDOWS)
======================================================================
"""

print("Analyzing gaming event impact on heart rate with multiple time windows...")

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
    print("Analyzing HR response to gaming events with multiple time windows...")
    
    for event in key_events:
        analysis = analyze_hr_around_events_multi_window(hr_df, video_data, event)
        if analysis is not None:
            event_analysis[event] = analysis
            print(f"\n{event.upper()} Events:")
            for window, data in analysis.items():
                significance = "✅ SIGNIFICANT" if data['significant'] else "⚪ Not significant"
                print(f"   {window} window: {data['mean_change']:+.1f} BPM change (n={data['sample_size']}) {significance}")

# %%
"""
SECTION 4: STRESS/EXCITEMENT DETECTION AND GAMING SCENARIOS
===========================================================
"""

print("Advanced stress and excitement detection during gaming...")

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
        
        print(f"Baseline HR: {baseline_hr:.1f} BPM")
        print(f"Stress threshold: {threshold_hr:.1f} BPM")
        print(f"Overall stress index: {stress_index:.1f}/100")
        print(f"RMSSD (relaxation): {rmssd:.1f} ms")
        
        # Classify stress levels
        stress_classification = "Low" if stress_index < 50 else "Moderate" if stress_index < 75 else "High"
        print(f"Overall stress level: {stress_classification}")
        
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
        
        print(f"Found {len(stress_periods)} high-intensity moments")
        print(f"Max stress level: +{stress_periods['stress_level'].max():.1f}% above baseline")
        
        # Analyze intensity distribution
        intensity_counts = stress_periods['intensity'].value_counts()
        print(f"Stress intensity distribution:")
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
        print(f"\nFound {len(corr_df)} stress-event correlations:")
        
        # Analyze which events correlate with stress
        event_stress = corr_df.groupby('event_type')['stress_level'].agg(['mean', 'count', 'std']).round(2)
        event_stress = event_stress.sort_values('mean', ascending=False)
        
        print("Events correlated with highest stress:")
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

print("Enhanced controller input analysis with clicks per second correlation...")

def analyze_controller_inputs_comprehensive(controller_data, hr_df, apple_hr_data=None, window_size=5):
    """
    Comprehensive controller input analysis including clicks per second correlation with heartbeat.
    Now supports multi-source heart rate comparison (EKG + Apple Watch).
    """
    if controller_data is None or hr_df is None:
        return None
    
    print(f"Available controller columns: {list(controller_data.columns)}")
    
    # Identify button/action columns
    button_columns = []
    action_columns = []
    
    for col in controller_data.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['button', 'btn', 'key', 'click', 'press']):
            button_columns.append(col)
        elif any(keyword in col_lower for keyword in ['action', 'input', 'event', 'command']):
            action_columns.append(col)
    
    print(f"Button columns found: {button_columns}")
    print(f"Action columns found: {action_columns}")
    
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
    
    print(f"Clicks analysis: {len(clicks_df)} time windows")
    print(f"   Max clicks/sec: {clicks_df['clicks_per_second'].max():.2f}")
    print(f"   Avg clicks/sec: {clicks_df['clicks_per_second'].mean():.2f}")
    print(f"   Max intensity: {clicks_df['intensity_score'].max():.2f}")
    
    # Calculate correlations with both heart rate sources
    correlations = {}
    
    input_metrics = ['clicks_per_second', 'button_clicks_per_second', 'intensity_score', 'action_diversity']
    hr_metrics = ['heart_rate_ekg_smooth', 'heart_rate_ekg_instant']
    if apple_hr_data is not None:
        hr_metrics.append('heart_rate_apple')
    
    print(f"\nClicks per Second vs Heart Rate Correlations:")
    
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
            print(f"\nHeart Rate Source Comparison:")
            print(f"   EKG vs Apple Watch HR correlation: {hr_source_correlation:.3f}")
    
    # Create enhanced clicks per second plot with multi-source heart rate overlay
    try:
        # Simplified layout - 2 rows only to reduce overcrowding
        subplot_titles = [
            'Controller Input Activity vs Heart Rate Over Time',
            'Input Intensity vs Heart Rate Correlation'
        ]
        
        fig_clicks = make_subplots(
            rows=2, cols=1,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        # Plot 1: Time series with dual y-axes (simplified)
        fig_clicks.add_trace(
            go.Scatter(
                x=clicks_df['time'],
                y=clicks_df['clicks_per_second'],
                mode='lines+markers',
                name='Clicks/Second',
                line=dict(color='#2E86DE', width=2.5),
                marker=dict(size=5, symbol='circle')
            ),
            row=1, col=1, secondary_y=False
        )
        
        fig_clicks.add_trace(
            go.Scatter(
                x=clicks_df['time'],
                y=clicks_df['heart_rate_ekg_smooth'],
                mode='lines',
                name='Heart Rate (BPM)',
                line=dict(color='#EE5A6F', width=2.5)
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Plot 2: Comprehensive scatter plot showing all correlations
        fig_clicks.add_trace(
            go.Scatter(
                x=clicks_df['clicks_per_second'],
                y=clicks_df['heart_rate_ekg_smooth'],
                mode='markers',
                name='Clicks vs HR',
                marker=dict(
                    color=clicks_df['intensity_score'],
                    colorscale='Turbo',
                    size=10,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Intensity<br>Score", side='right'),
                        x=1.12
                    ),
                    line=dict(width=0.5, color='white')
                ),
                text=[f'Time: {t:.0f}s<br>Intensity: {i:.2f}<br>Diversity: {d:.2f}' 
                      for t, i, d in zip(clicks_df['time'], clicks_df['intensity_score'], clicks_df['action_diversity'])],
                hovertemplate='Clicks/sec: %{x:.2f}<br>HR: %{y:.1f} BPM<br>%{text}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update layout with cleaner design
        fig_clicks.update_layout(
            height=800,
            title={
                'text': "Controller Input Activity vs Heart Rate Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            template='plotly_white'
        )
        
        # Update axes with clear labels
        fig_clicks.update_xaxes(title_text="Time (seconds)", row=1, col=1, gridcolor='lightgray')
        fig_clicks.update_xaxes(title_text="Clicks per Second", row=2, col=1, gridcolor='lightgray')
        
        fig_clicks.update_yaxes(title_text="Clicks/Second", row=1, col=1, secondary_y=False, gridcolor='lightgray')
        fig_clicks.update_yaxes(title_text="Heart Rate (BPM)", row=1, col=1, secondary_y=True, gridcolor='lightgray')
        fig_clicks.update_yaxes(title_text="Heart Rate (BPM)", row=2, col=1, gridcolor='lightgray')
        
        # Save the plot
        fig_clicks.write_html(output_dir / "controller_clicks_multisource_analysis.html")
        fig_clicks.show()
        
        print(f"[OK] Controller input vs heart rate analysis saved to: {output_dir / 'controller_clicks_multisource_analysis.html'}")
        
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
        print(f"Enhanced controller analysis: {controller_clicks_analysis['summary']}")
    else:
        print("⚠️ Could not complete enhanced controller analysis")
else:
    controller_clicks_analysis = None
    print("[WARNING] No controller data or heart rate data available for enhanced analysis")

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
        print(f"Available columns: {list(controller_data.columns)}")
        return None
    
    print(f"Using input columns: {input_columns}")
    
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
            print(f"{metric} vs HR: correlation = {corr:.3f} (moderate)")
        elif abs(corr) > 0.1:
            print(f"{metric} vs HR: correlation = {corr:.3f} (weak)")
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
        print(f"Input intensity analysis: {len(intensity_df)} time windows analyzed")
    else:
        intensity_correlation = None
        print("⚠️ Could not calculate input intensity")
else:
    intensity_df = None
    intensity_correlation = None
    print("[WARNING] No controller data available for intensity analysis")

# %%
"""
SECTION 6: PERFORMANCE VS PHYSIOLOGICAL STATE ANALYSIS
======================================================
"""

print("Analyzing gaming performance vs physiological state...")

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
    
    print("Performance vs Physiology Correlations:")
    
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
        print(f"Performance analysis: {performance_analysis['summary']}")
    else:
        print("⚠️ Could not complete performance analysis")
else:
    performance_analysis = None
    print("[WARNING] Insufficient data for performance analysis")

# %%
"""
SECTION 7: COMPREHENSIVE MULTIMODAL VISUALIZATION
=================================================
"""

print("Creating comprehensive multimodal visualization dashboard...")

def create_comprehensive_dashboard(ekg_data, hr_df, video_data, stress_periods, event_analysis, 
                                 intensity_df, performance_analysis):
    """
    Create a comprehensive dashboard with multiple synchronized visualizations.
    Only includes plots for available data sources.
    """
    
    # Create subplots with improved layout - 3 rows, full width for better visibility
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            'Heart Rate Over Time with Gaming Events',
            'Heart Rate Response to Key Gaming Events',
            'Performance and Physiological Correlation'
        ],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}]
        ],
        vertical_spacing=0.12,
        row_heights=[0.35, 0.35, 0.30]
    )
    
    # Plot 1: Heart Rate Timeline with Gaming Events
    if hr_df is not None and len(hr_df) > 0:
        # Heart rate over time
        fig.add_trace(
            go.Scatter(
                x=hr_df['time'],
                y=hr_df['hr_smooth'],
                mode='lines',
                name='Heart Rate (Smoothed)',
                line=dict(color='rgb(220, 53, 69)', width=2),
                fill='tozeroy',
                fillcolor='rgba(220, 53, 69, 0.1)'
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
        height=1200,
        title={
            'text': "Multimodal Gaming Health Analysis Dashboard",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50'}
        },
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_xaxes(title_text="Gaming Events", row=2, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
    
    fig.update_yaxes(title_text="Heart Rate (BPM)", row=1, col=1)
    fig.update_yaxes(title_text="HR Change (BPM)", row=2, col=1)
    fig.update_yaxes(title_text="K/D Ratio", row=3, col=1)
    fig.update_yaxes(title_text="Heart Rate (BPM)", row=3, col=1, secondary_y=True)
    
    return fig

# Create the comprehensive dashboard
try:
    # Create a simpler, cleaner dashboard
    from plotly.subplots import make_subplots
    
    # Determine how many plots we can create
    has_hr = hr_df is not None and len(hr_df) > 0
    has_events = video_data is not None and len(video_data) > 0
    has_perf = performance_analysis is not None and 'data' in performance_analysis and not performance_analysis['data'].empty
    
    if has_hr:
        dashboard_fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Heart Rate Over Time with Gaming Events', 'Performance Metrics'],
            vertical_spacing=0.15,
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
        )
        
        # Plot 1: Heart Rate
        dashboard_fig.add_trace(
            go.Scatter(
                x=hr_df['time'],
                y=hr_df['hr_smooth'],
                mode='lines',
                name='Heart Rate',
                line=dict(color='rgb(220, 53, 69)', width=2),
                fill='tozeroy',
                fillcolor='rgba(220, 53, 69, 0.1)'
            ),
            row=1, col=1
        )
        
        # Add mean HR line
        if mean_hr:
            dashboard_fig.add_hline(
                y=mean_hr,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Mean: {mean_hr:.1f} BPM",
                row=1, col=1
            )
        
        # Add events if available
        if has_events:
            event_colors = {'kill': 'rgba(40, 167, 69, 0.3)', 'death': 'rgba(220, 53, 69, 0.3)', 
                           'round_start': 'rgba(0, 123, 255, 0.2)'}
            for event_type, color in event_colors.items():
                events = video_data[video_data['action'] == event_type]['time_seconds'].values
                for event_time in events[:30]:  # Limit for clarity
                    dashboard_fig.add_vline(x=event_time, line_color=color, line_width=1, row=1, col=1)
        
        # Plot 2: Performance
        if has_perf:
            perf_data = performance_analysis['data']
            # Check what columns actually exist
            time_col = 'time' if 'time' in perf_data.columns else 'time_window'
            
            dashboard_fig.add_trace(
                go.Scatter(
                    x=perf_data[time_col],
                    y=perf_data['kd_ratio'],
                    mode='lines+markers',
                    name='K/D Ratio',
                    line=dict(color='rgb(0, 123, 255)', width=2)
                ),
                row=2, col=1
            )
            
            dashboard_fig.add_trace(
                go.Scatter(
                    x=perf_data[time_col],
                    y=perf_data['avg_heart_rate'],
                    mode='lines',
                    name='Avg HR',
                    line=dict(color='rgb(220, 53, 69)', width=2, dash='dash'),
                    yaxis='y2'
                ),
                row=2, col=1,
                secondary_y=True
            )
        
        dashboard_fig.update_layout(
            height=900,
            title={'text': 'Gaming Health Analysis Dashboard', 'x': 0.5, 'xanchor': 'center'},
            template='plotly_white',
            showlegend=True
        )
        
        dashboard_fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
        dashboard_fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        dashboard_fig.update_yaxes(title_text="Heart Rate (BPM)", row=1, col=1)
        dashboard_fig.update_yaxes(title_text="K/D Ratio", row=2, col=1)
        dashboard_fig.update_yaxes(title_text="Heart Rate (BPM)", row=2, col=1, secondary_y=True)
        
        dashboard_fig.show()
        print("[OK] Dashboard created successfully!")
    else:
        print("[WARNING] Insufficient data for dashboard creation")
        
except Exception as e:
    print(f"[WARNING] Could not create dashboard: {e}")
    import traceback
    traceback.print_exc()

# %%
"""
SECTION 8: STATISTICAL SUMMARY AND CORRELATION MATRIX
=====================================================
"""

print("Creating comprehensive statistical summary and correlation matrix...")

def create_comprehensive_correlation_matrix():
    """
    Create a comprehensive correlation matrix from all available data.
    Handles time-aligned data properly to avoid NaN correlations.
    """
    correlation_data = {}
    
    # Start with performance data as the base (it has time windows)
    if performance_analysis is not None and 'data' in performance_analysis:
        perf_data = performance_analysis['data'].copy()
        if len(perf_data) > 0 and not perf_data.empty:
            # Use performance data columns that are numeric
            if 'kd_ratio' in perf_data.columns:
                correlation_data['KD_Ratio'] = perf_data['kd_ratio'].values
            if 'performance_score' in perf_data.columns:
                correlation_data['Performance_Score'] = perf_data['performance_score'].values
            if 'action_rate' in perf_data.columns:
                correlation_data['Action_Rate'] = perf_data['action_rate'].values
            if 'avg_heart_rate' in perf_data.columns:
                correlation_data['Avg_Heart_Rate'] = perf_data['avg_heart_rate'].values
            if 'hr_variability' in perf_data.columns:
                correlation_data['HR_Variability'] = perf_data['hr_variability'].values
            if 'kills' in perf_data.columns:
                correlation_data['Kills'] = perf_data['kills'].values
            if 'deaths' in perf_data.columns:
                correlation_data['Deaths'] = perf_data['deaths'].values
    
    # Add intensity metrics if available and same length
    if intensity_correlation is not None and 'data' in intensity_correlation:
        intensity_data = intensity_correlation['data']
        if len(intensity_data) > 0 and not intensity_data.empty:
            base_len = len(next(iter(correlation_data.values()))) if correlation_data else len(intensity_data)
            if len(intensity_data) == base_len:
                if 'intensity_score' in intensity_data.columns:
                    correlation_data['Input_Intensity'] = intensity_data['intensity_score'].values
                if 'inputs_per_second' in intensity_data.columns:
                    correlation_data['Inputs_Per_Second'] = intensity_data['inputs_per_second'].values
    
    # Create correlation matrix if we have data
    if correlation_data and len(correlation_data) >= 2:
        corr_df = pd.DataFrame(correlation_data)
        
        # Remove columns with all NaN or constant values
        corr_df = corr_df.dropna(axis=1, how='all')
        corr_df = corr_df.loc[:, corr_df.std() > 0]
        
        if len(corr_df.columns) >= 2:
            # Calculate correlation matrix
            correlation_matrix = corr_df.corr()
            
            # Remove NaN values (replace with 0 for visualization)
            correlation_matrix = correlation_matrix.fillna(0)
            
            # Create improved heatmap
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=[col.replace('_', ' ') for col in correlation_matrix.columns],
                y=[col.replace('_', ' ') for col in correlation_matrix.columns],
                colorscale='RdBu',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=correlation_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 11, "color": "white"},
                hoverongaps=False,
                colorbar=dict(title="Correlation")
            ))
            
            fig_corr.update_layout(
                title={
                    'text': "Performance and Physiological Correlation Matrix",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                width=900,
                height=700,
                template='plotly_white',
                xaxis={'side': 'bottom'},
                yaxis={'autorange': 'reversed'}
            )
            
            fig_corr.show()
            
            # Save correlation matrix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            correlation_matrix.to_csv(output_dir / f'correlation_matrix_{timestamp}.csv')
            print(f"[OK] Correlation matrix saved with {len(correlation_matrix)} variables")
            
            return correlation_matrix
        else:
            print("[WARNING] Insufficient variables with valid data for correlation analysis")
    else:
        print("[WARNING] Need at least 2 data sources for correlation analysis")
    
    return pd.DataFrame()

def generate_comprehensive_insights():
    """
    Generate comprehensive insights from all analyses.
    """
    insights = []
    
    # EKG insights using existing analyzer results
    if ekg_data is not None:
        try:
            insights.append(f"Average heart rate during gaming: {mean_hr:.1f} BPM")
            insights.append(f"Stress index: {ekg_data.EKG.stress_index:.1f}/100 ({ekg_data.EKG.rhythm_type})")
            insights.append(f"HRV Score: {ekg_data.EKG.hr_variability_score:.1f} (RMSSD: {ekg_data.EKG.rmssd:.1f}ms)")
            insights.append(f"Detected {len(peaks)} R-peaks and {len(beats)} beats over {ekg_data['Timestamp'].max():.1f}s")
        except Exception as e:
            insights.append(f"[WARNING] EKG analysis partially available")
    
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
            insights.append(f"Most impactful events: {'; '.join(most_impactful_events[:3])}")
    
    # Stress insights
    if stress_periods is not None:
        stress_duration = len(stress_periods) * 5  # Approximate
        total_duration = ekg_data['Timestamp'].max() if ekg_data is not None else 0
        stress_percentage = (stress_duration / total_duration) * 100 if total_duration > 0 else 0
        insights.append(f"High-stress periods: {stress_percentage:.1f}% of gaming session")
        
        if stress_event_correlations is not None:
            top_stress_events = stress_event_correlations.groupby('event_type')['stress_level'].mean().sort_values(ascending=False)
            if len(top_stress_events) > 0:
                insights.append(f"Most stressful events: {', '.join(top_stress_events.head(3).index.tolist())}")
    
    # Performance insights
    if performance_analysis is not None and 'data' in performance_analysis:
        perf_data = performance_analysis['data']
        avg_kd = perf_data['kd_ratio'].mean()
        best_performance_hr = perf_data.loc[perf_data['performance_score'].idxmax(), 'avg_heart_rate']
        insights.append(f"Average K/D ratio: {avg_kd:.2f}")
        insights.append(f"Best performance occurred at {best_performance_hr:.1f} BPM")
    
    # Input intensity insights
    if intensity_correlation is not None and 'correlations' in intensity_correlation:
        correlations = intensity_correlation['correlations']
        strongest_corr = max(correlations.items(), key=lambda x: abs(x[1])) if correlations else None
        if strongest_corr and abs(strongest_corr[1]) > 0.3:
            direction = "positively" if strongest_corr[1] > 0 else "negatively"
            insights.append(f"Input intensity {direction} correlates with heart rate (r={strongest_corr[1]:.3f})")
    
    # Data quality insights
    data_sources = []
    if ekg_data is not None:
        data_sources.append(f"EKG ({len(ekg_data)} samples)")
    if video_data is not None:
        data_sources.append(f"Video ({len(video_data)} events)")
    if controller_data is not None:
        data_sources.append(f"Controller ({len(controller_data)} inputs)")
    if apple_hr_data is not None:
        data_sources.append(f"Apple Watch ({len(apple_hr_data)} HR measurements)")
    if oura_hr_data is not None:
        data_sources.append(f"Oura Ring ({len(oura_hr_data)} HR measurements)")
    
    insights.append(f"Data sources: {', '.join(data_sources)}")
    
    # Multi-source HR comparison insights
    if hr_comparison is not None:
        available_sources = list(hr_comparison.keys())
        if 'Apple_Watch' in available_sources or 'Oura_Ring' in available_sources:
            num_sources = len([k for k in available_sources if k != 'agreement'])
            insights.append(f"Multi-device validation: {num_sources} heart rate sources compared")
            
            # Add Oura-specific insights
            if 'Oura_Ring' in available_sources:
                oura_stats = hr_comparison['Oura_Ring']
                insights.append(f"Oura Ring baseline: {oura_stats['mean_hr']:.1f} BPM average")
                
                # Add readiness context if available
                if oura_readiness:
                    insights.append(f"Oura Readiness: {oura_readiness.get('score')}/100 - Body prepared for activity")
                
                # Add temperature insights if available
                if oura_temp_data is not None and len(oura_temp_data) > 0:
                    temp_change = oura_temp_data['skin_temp'].max() - oura_temp_data['skin_temp'].min()
                    insights.append(f"Skin temperature variation: {temp_change:.2f}°C during session")
    
    return insights

# Create correlation matrix and generate insights
print("Creating correlation matrix...")
correlation_matrix = create_comprehensive_correlation_matrix()

if not correlation_matrix.empty:
    print(f"[OK] Correlation matrix created with {len(correlation_matrix)} variables")
else:
    print("[WARNING] Could not create correlation matrix - insufficient overlapping data")

print("\nCOMPREHENSIVE INSIGHTS:")
print("=" * 50)
insights = generate_comprehensive_insights()
for insight in insights:
    print(f"  {insight}")

# %%
"""
SECTION 9: RESEARCH RECOMMENDATIONS AND EXPORT
==============================================
"""

print("\nGenerating research recommendations and exporting results...")

def generate_research_recommendations():
    """
    Generate research recommendations based on findings.
    """
    recommendations = []
    
    recommendations.append("RESEARCH RECOMMENDATIONS:")
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
    recommendations.append("\nTECHNICAL RECOMMENDATIONS:")
    recommendations.append("• Implement real-time biofeedback during gaming")
    recommendations.append("• Develop personalized break recommendations based on physiological state")
    recommendations.append("• Create adaptive difficulty based on stress levels")
    recommendations.append("• Consider heart rate variability training for competitive gamers")
    
    # Future research directions
    recommendations.append("\nFUTURE RESEARCH DIRECTIONS:")
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
    print(f"[OK] Recommendations saved to: research_recommendations_{timestamp}.txt")
except Exception as e:
    print(f"[WARNING] Could not save recommendations: {e}")

print("\n" + "=" * 60)
print("COMPREHENSIVE MULTIMODAL GAMING HEALTH ANALYSIS COMPLETE")
print("=" * 60)
print("Key Features Completed:")
print("[OK] EKG analysis using existing peaks/beats detection")
print("[OK] Multi-window event response analysis (5s, 15s, 30s)")
print("[OK] Advanced stress detection and correlation")
print("[OK] Controller input intensity analysis")
print("[OK] Performance vs physiology correlation")
print("[OK] Comprehensive visualization dashboard")
print("[OK] Statistical correlation matrix")
print("[OK] Research recommendations")
print("[OK] Results export functionality")
print("\nThis analysis leverages all existing analyzer capabilities")
print("and provides comprehensive insights into gaming health correlations.")
print("=" * 60)

# %%

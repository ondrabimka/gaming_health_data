# %%
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import plotly.graph_objects as go

# %%
ekg_data = pd.read_csv("gaming_health_data/recorded_data/ekg_data.txt", names=['Timestamp','HeartSignal'])

### EKG data ###
# %% Add dots to peaks
ekg_data['Timestamp'] = ekg_data['Timestamp'] - ekg_data['Timestamp'].iloc[0]
peaks, _ = find_peaks(ekg_data['HeartSignal'], distance=10, height=1.9, prominence=0.7)
low_peaks, _ = find_peaks(-ekg_data['HeartSignal'], distance=10, height=-1.3, prominence=0.5)
ekg_data['Peaks'] = ekg_data['HeartSignal']
ekg_data['Peaks'].iloc[peaks] = ekg_data['HeartSignal'].iloc[peaks]
ekg_data['Peaks'].iloc[low_peaks] = ekg_data['HeartSignal'].iloc[low_peaks]
ekg_data['Smoothed'] = savgol_filter(ekg_data['HeartSignal'], 80, 3)

fig = go.Figure()
fig.add_scatter(x=ekg_data['Timestamp'], y=ekg_data['HeartSignal'], mode='lines', name='Raw')
fig.add_scatter(x=ekg_data['Timestamp'].iloc[peaks], y=ekg_data['Peaks'].iloc[peaks], mode='markers', name='Peaks')
fig.add_scatter(x=ekg_data['Timestamp'].iloc[low_peaks], y=ekg_data['Peaks'].iloc[low_peaks], mode='markers', name='Low Peaks')
fig.add_scatter(x=ekg_data['Timestamp'], y=ekg_data['Smoothed'], mode='lines', name='Smoothed')
fig.show()

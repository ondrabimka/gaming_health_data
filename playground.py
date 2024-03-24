# %%
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import plotly.express as px

# %%
ekg_data = pd.read_csv("gaming_healt_data/recorded_data/ekg_data.txt", names=['Timestamp','HeartSignal'])
keyboard_log = pd.read_csv("gaming_healt_data/recorded_data/keyboard_log.csv")
mouse_log = pd.read_csv("gaming_healt_data/recorded_data/mouse_log.csv")

### EKG data ###
# %% Add dots to peaks
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

### Keyboard log ###
# %% take first row as 0 and calculate time difference
keyboard_log['Time (ms)'] = keyboard_log['Time (ms)'] - keyboard_log['Time (ms)'].iloc[0]

# %% count keypresses in 1 second intervals
keypresses = []
for i in range(0, keyboard_log['Time (ms)'].iloc[-1], 1000):
    keypresses.append(len(keyboard_log[(keyboard_log['Time (ms)'] > i) & (keyboard_log['Time (ms)'] < i+1000)]))

keypresses = pd.DataFrame(keypresses, columns=['Keypresses'])

# %% plot keypresses 
fig = px.line(keypresses, y='Keypresses')
fig.show()

### Mouse log ###
# %%
mouse_log['Time (ms)'] = mouse_log['Time (ms)'] - mouse_log['Time (ms)'].iloc[0]
mouse_log['X'] = mouse_log['Details'].str.extract(r'X: (-?\d+)')
mouse_log['Y'] = mouse_log['Details'].str.extract(r'Y: (-?\d+)')
mouse_log['X'] = mouse_log['X'].astype(float)
mouse_log['Y'] = mouse_log['Y'].astype(float)

# %% Plot heatmap of mouse movement
fig = px.scatter(mouse_log, x='X', y='Y', color='Time (ms)')
fig.show()

# %% plot it in 3d
# fig = px.scatter_3d(mouse_log, x='X', y='Y', z='Time (ms)', color='Time (ms)')
# fig.show()
# %% Animate mouse movement
# fig = px.scatter(mouse_log[:100], x='X', y='Y', animation_frame='Time (ms)')
# fig.show()


# %%

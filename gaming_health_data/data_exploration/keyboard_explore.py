import pandas as pd
import plotly.express as px

keyboard_log = pd.read_csv("gaming_health_data/recorded_data/keyboard_log.csv")

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
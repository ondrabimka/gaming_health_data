import pandas as pd
import plotly.express as px
from gaming_health_data.src.utils import DATA_DIR

mouse_log = pd.read_csv(DATA_DIR / "mouse_log.csv")

### Mouse log ###
mouse_log['Time (ms)'] = mouse_log['Time (ms)'] - mouse_log['Time (ms)'].iloc[0]
mouse_log['X'] = mouse_log['Details'].str.extract(r'X: (-?\d+)')
mouse_log['Y'] = mouse_log['Details'].str.extract(r'Y: (-?\d+)')
mouse_log['X'] = mouse_log['X'].astype(float)
mouse_log['Y'] = mouse_log['Y'].astype(float)

# %% Plot heatmap of mouse movement
fig = px.scatter(mouse_log, x='X', y='Y', color='Time (ms)')
fig.show()
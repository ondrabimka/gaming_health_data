# %%
import pandas as pd
from gaming_health_data.src.EKGAnalyzer import EKGAnalyzer
# %%
ekg_analyzer = EKGAnalyzer.from_file("C:/Users/Admin/Desktop/embedded_code/gaming_health_data/gaming_health_data/recorded_data/ekg_data_28_04_2024.txt", to_seconds=False)
# %%
ekg_analyzer.EKG.signal_center
# %%
ekg_analyzer.EKG.plot_ekg_data()
# %%
ekg_analyzer.EKG.plot_moving_avg_bpm()
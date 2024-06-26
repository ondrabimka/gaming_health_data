# %%
import pandas as pd
from gaming_health_data.src.EKGAnalyzer import EKGAnalyzer
from gaming_health_data.src.utils import DATA_DIR
# %%
ekg_analyzer = EKGAnalyzer.from_file(DATA_DIR / "ekg_data_28_04_2024.txt", to_seconds=False)
# %%
ekg_analyzer.EKG.signal_center
# %%
ekg_analyzer.EKG.plot_ekg_data()
# %%
ekg_analyzer.EKG.plot_moving_avg_bpm()
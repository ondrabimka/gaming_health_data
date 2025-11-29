## Welcome to Gaming Health Data repository!
Thank you for visiting the Gaming Health Data repository! This repository is dedicated to collecting and analyzing health data related to gaming. Whether you are a gamer, a researcher, or simply interested in the intersection of gaming and health, you've come to the right place.

This repository contains comprehensive tools for real-time physiological monitoring during gaming sessions, including EKG heart rate analysis, video game event detection, and advanced correlation analysis between gaming events and physiological responses.

### Key Features

- **Advanced EKG Analysis**: Comprehensive heart rate variability (HRV) analysis with stress detection
- **Video Game Event Detection**: Automated detection of deaths, damage, healing from gameplay video
- **Physiological Correlation**: Advanced analysis correlating gaming events with heart responses
- **Interactive Visualizations**: Multi-panel plots showing health, heart metrics, and gaming events
- **Clinical-Grade Metrics**: RMSSD, SDNN, pNN50, stress index, and more
- **Real-Time Monitoring**: Support for Polar H10 and AD8232 EKG sensors

### Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
- [Key Components](#key-components)
- [Analysis Capabilities](#analysis-capabilities)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

### About

This repository provides a complete framework for gaming health research, including:

- **EKG Data Collection & Analysis**: Real-time heart monitoring during gaming sessions
- **Video Game Event Detection**: Automated analysis of gameplay videos to detect health events
- **Physiological Response Analysis**: Correlation between gaming stress and heart rate changes  
- **Advanced Visualization Tools**: Interactive plots for comprehensive data exploration
- **Research-Grade Analytics**: Statistical analysis of gaming-health relationships


### Getting Started

To get started with gaming health analysis, follow these steps:

1. **Clone the repository** to your local machine:
    ```bash
    git clone https://github.com/ondrabimka/gaming_health_data.git
    cd gaming_health_data
    ```

2. **Install dependencies** (Python 3.8+ required):
    ```bash
    pip install pandas numpy plotly scipy scikit-learn opencv-python pytesseract
    ```

3. **Quick Start with Demo**:
    - Open `gaming_health_correlation_demo.ipynb` for a complete walkthrough
    - Or explore individual components in the `gaming_health_data/src/` directory

4. **Basic Usage**:
    ```python
    # EKG Analysis
    from gaming_health_data.src.EKGAnalyzer import EKGAnalyzer
    ekg_data = EKGAnalyzer.read_file("data.txt", sensor_type="PolarH10")
    summary = ekg_data.EKG.get_heart_analysis_summary()
    
    # Gaming Health Correlation
    from gaming_health_data.src.gaming_health_correlator import GamingHealthCorrelator
    correlator = GamingHealthCorrelator()
    correlator.load_data(ekg_file="ekg.txt", video_file="health.csv")
    correlator.plot_comprehensive_analysis()
    ```

### Key Components

#### **EKGAnalyzer** - Advanced Heart Rate Analysis
- **Multi-sensor support**: Polar H10, AD8232 sensors
- **HRV Metrics**: RMSSD, SDNN, pNN50, LF/HF ratio
- **Adaptive Peak Detection**: Smart signal processing based on data characteristics  
- **Stress Analysis**: Real-time stress index calculation (0-100 scale)
- **Clinical Accuracy**: Physiologically validated thresholds and calculations

#### **VideoAnnotator** - Gaming Event Detection  
- **Automated OCR**: Extract health values from gameplay video
- **Event Detection**: Deaths, major damage, healing events, low health periods
- **Data Cleaning**: Robust filtering of OCR errors and invalid readings
- **Timeline Analysis**: Health patterns over entire gaming sessions

#### **GamingHealthCorrelator** - Physiological Response Analysis
- **Event Correlation**: Heart rate changes around gaming events  
- **Stress Pattern Analysis**: Rolling analysis of stress throughout sessions
- **Statistical Analysis**: Quantify relationships between events and physiology
- **Interactive Visualization**: Multi-panel plots with event markers

### Analysis Capabilities

#### **Heart Rate Variability (HRV) Analysis**
```
Sample Output:
Mean HR: 85.6 bpm | Avg RR: 701 ms | Duration: 53.5 min | Beats: 4579
HRV Metrics: RMSSD: 61.3 ms | SDNN: 96.1 ms | pNN50: 7.2% | HRV Score: 75/100  
Frequency: LF/HF ratio: 2.98 | Rhythm: Sinus rhythm | Arrhythmia: No | Stress: 38/100
```

#### **Gaming Event Detection**
- **Deaths**: Automatic detection when player health = 0
- **Major Damage**: Rapid health drops (>30 health in <2 seconds)  
- **Low Health Periods**: Extended time below 25% health
- **Healing Events**: Rapid health increases (healing/health packs)

#### **Correlation Analysis**  
- **Pre/Post Event Analysis**: Heart rate changes 10 seconds before/after events
- **Event Severity Impact**: Different physiological responses by event severity
- **Recovery Patterns**: How quickly heart rate returns to baseline
- **Session-Wide Patterns**: Overall stress trends during gaming

### Project Structure

```
gaming_health_data/
├── gaming_health_data/
│   ├── src/                           # Core analysis modules
│   │   ├── EKGAnalyzer.py            # Advanced EKG/HRV analysis
│   │   ├── video_annotator_pytesseract.py  # Video health detection
│   │   ├── video_analyzer.py         # Health data processing
│   │   ├── apple_watch_analyzer.py   # Apple Watch health data analysis
│   │   ├── mouse_analyzer.py         # Mouse movement analysis
│   │   ├── keyboard_analyzer.py      # Keyboard input analysis  
│   │   └── dualsense_analyzer.py     # Controller input analysis
│   ├── data_exploration/             # Analysis notebooks
│   │   ├── exploration_notebook.ipynb    # Main analysis notebook
│   │   ├── ekg_explore.py           # EKG data exploration
│   │   ├── mouse_explore.py         # Mouse data exploration
│   │   └── keyboard_explore.py      # Keyboard data exploration
│   ├── data_loggers/                 # Data collection tools
│   │   ├── PC/                       # PC-based loggers
│   │   ├── PS/                       # PlayStation loggers
│   │   └── SENSORS/                  # Sensor data loggers
│   └── recorded_data/                # Collected datasets
│       ├── PC/                       # PC gaming data
│       ├── PS/                       # PlayStation data
│       ├── SENSORS/                  # EKG/heart sensor data
│       ├── APPLE_WATCH/             # Apple Health exports
│       └── VIDEO/                    # Gameplay video recordings
└── README.md                        # This file
```

#### Key Directories:
- **[src/](gaming_health_data/src/)**: Core analysis modules and classes
- **[data_exploration/](gaming_health_data/data_exploration/README.md)**: Interactive notebooks for data analysis
- **[data_loggers/](gaming_health_data/data_loggers/README.md)**: Tools for real-time data collection
- **[recorded_data/](gaming_health_data/recorded_data/README.md)**: Sample datasets and recordings

#### Main Analysis Files:
- **[exploration_notebook.ipynb](gaming_health_data/data_exploration/exploration_notebook.ipynb)**: Interactive analysis

#### **Research Questions You Can Answer**
- **"Does getting shot increase heart rate?"** - Measure HR changes around damage events
- **"How stressful are death events?"** - Quantify stress response to deaths
- **"Do low health periods create sustained stress?"** - Analyze prolonged activation  
- **"Which events are most physiologically demanding?"** - Compare event impact
- **"How quickly do I recover from stress?"** - Study heart rate recovery patterns 

### Contributing

Contributions are welcome! This project is actively being developed for gaming health research.

#### **Ways to Contribute:**
- **Bug Reports**: Found an issue? Open an issue with details
- **Feature Requests**: Ideas for new analysis capabilities  
- **Data Contributions**: Share anonymized gaming health datasets
- **Research**: Validation studies, new correlation methods
- **Documentation**: Improve guides, add tutorials
- **Code**: Bug fixes, performance improvements, new features

#### **Current Research Areas:**
- Real-time stress detection during gaming
- Multi-modal physiological monitoring (EKG + other sensors)
- Machine learning for event prediction
- Cross-game stress pattern analysis
- Long-term health impact studies

### Research & Citations

If you use this repository in your research, please cite:

```bibtex
@misc{gaming_health_data_2025,
  title={Gaming Health Data: Physiological Response Analysis During Gaming Sessions},
  author={Ondrej Bimka},
  year={2025},
  url={https://github.com/ondrabimka/gaming_health_data}
}
```

#### **Related Research Areas:**
- Gaming psychology and physiological arousal
- Heart rate variability in competitive gaming
- Stress detection through EKG analysis
- Real-time biofeedback systems
- Gaming performance and physiological state

### License

This repository is licensed under the [MIT License](LICENSE). Feel free to use the code and resources available here for your own projects and research.

---

### **Acknowledgments**

- Polar H10 sensor community for excellent EKG data quality
- OpenCV and Tesseract teams for computer vision tools
- Plotly team for interactive visualization capabilities  
- Scientific Python ecosystem (NumPy, SciPy, Pandas)

**Happy Gaming & Stay Healthy!**

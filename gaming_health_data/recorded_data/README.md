# Gaming Health Data Collection

This directory contains multi-modal physiological and behavioral data collected during gaming sessions for health research. The data is synchronized across multiple sensors and input devices to provide comprehensive insights into gaming-induced physiological responses.

## Data Categories

The recorded data is organized into 5 main categories:

### 1. **SENSORS** - Physiological Measurements
- **Polar H10 EKG**: High-precision heart rate and ECG data
- **Raspberry Pi sensors**: Remote physiological sensors (see setup below)

### 2. **PC** - Computer Input Tracking  
- **Keyboard logs**: Keystroke timing and patterns
- **Mouse logs**: Movement, clicks, and interaction patterns

### 3. **PS** - PlayStation Controller Data
- **DualSense controller**: Button presses, analog stick movements, triggers

### 4. **VIDEO** - Session Recordings
- **Annotated gameplay**: Video recordings with game-specific annotations
- **Synchronization markers**: Time-aligned with sensor data

### 5. **APPLE_WATCH** - Consumer Health Data
- **Heart rate**: Continuous HR monitoring during gaming sessions
- **Activity data**: Movement and exercise detection

## Hardware Requirements

### Physiological Sensors
- **Polar H10 Heart Rate Sensor**
  - Bluetooth LE connectivity
  - ECG-accurate heart rate measurement
  - Setup: Chest strap placement for optimal signal quality

- **EKG Sensor (Raspberry Pi)**
  - Custom sensor setup via Raspberry Pi
  - Real-time ECG signal acquisition
  - Setup guide: https://github.com/ondrabimka/raspberry_pico_sensors/blob/main/raspberry_pico_sensors/ekg_heartbeat/README.md

### Input Devices
- **PC Setup**: Python 3.8+, standard keyboard/mouse
- **PlayStation**: DualSense controller with wireless connectivity
- **Apple Watch**: Series 4+ recommended for health data accuracy

## Apple Watch Data Export

To collect Apple Watch health data for analysis, follow these steps:

### Step 1: Export Health Data from iPhone
1. Open the **Health app** on your iPhone
2. Tap your **profile picture** (top right)
3. Scroll down and tap **"Export All Health Data"**
4. Choose **"Export"** and wait for processing (may take several minutes)

### Step 2: Convert XML to CSV
The exported data comes as XML files that need conversion to CSV for analysis.

**Using Simple-Apple-Health-XML-to-CSV tool:**

```bash
# Clone the converter tool
git clone https://github.com/jameno/Simple-Apple-Health-XML-to-CSV.git
cd Simple-Apple-Health-XML-to-CSV

# Extract your health export ZIP file first
unzip export.zip

# Convert to CSV (replace with your export.xml path)
python apple_health_export_xml_to_csv.py export.xml
```

### Step 3: File Placement
Place the converted `apple_health_export.csv` file in:
```
gaming_health_data/recorded_data/APPLE_WATCH/apple_health_export_YYYY-MM-DD.csv
```

**Important Notes:**
- Export data regularly for continuous monitoring
- Ensure gaming sessions are marked as "Fitness Gaming" workouts for proper detection
- Privacy: Health data never leaves your devices during our analysis

## File Structure

```
recorded_data/
├── PC/                     # Computer input data
│   ├── keyboard_log_*.csv  # Keystroke patterns and timing
│   ├── mouse_log_*.csv     # Mouse movement and clicks  
│   └── ekg_data_*.txt      # Legacy EKG files
├── PS/                     # PlayStation controller data
│   └── controller_inputs_*_part_*.csv  # DualSense input logs
├── SENSORS/                # Physiological sensor data
│   └── ekg_data_polars_h10_*.txt      # Polar H10 ECG recordings
├── VIDEO/                  # Gameplay recordings
│   └── [session recordings with annotations]
└── APPLE_WATCH/           # Consumer health data  
    └── apple_health_export_*.csv      # Exported health metrics
```

## Related Resources

- **Sensor Hardware Setup**: https://github.com/ondrabimka/raspberry_pico_sensors/blob/main/raspberry_pico_sensors/ekg_heartbeat/README.md
- **Apple Health XML Converter**: https://github.com/jameno/Simple-Apple-Health-XML-to-CSV
- **Data Analysis Tools**: See `../src/` directory for pandas extensions and analyzers

## Data Analysis

The collected data can be analyzed using our custom pandas extensions:

```python
# Load and analyze Apple Watch data
import pandas as pd
apple_watch_data = pd.read_csv('APPLE_WATCH/apple_health_export.csv')
apple_watch_data.applewatch.plot_hearth_rate_for_session_date('2025-05-08')

# Load and analyze EKG data  
from EKGAnalyzer import EKGAnalyzer
ekg_data = EKGAnalyzer.read_file_polar_h10('SENSORS/ekg_data_polars_h10_*.txt')

# Create combined visualizations
fig = apple_watch_data.applewatch.plot_combined_heart_rate_with_ekg('2025-05-08', ekg_data)
```

For detailed analysis examples, see the main project documentation.



# Data Loggers
This directory contains the data loggers for the gaming health project. The data loggers are used to collect health data from various sources, including EKG, keyboard...

Loggers are currently divided into 3 categories:
1. Sensors: Remote sensors connected via raspberry pi and polars H10 sensor.
2. PC: Stuff connected to PC (mouse, keyboard).
3. PS: Playstation controller (Dualsense).

## Requirements
#### Sensors 
- EKG sensor (setup: https://github.com/ondrabimka/raspberry_pico_sensors/blob/main/raspberry_pico_sensors/ekg_heartbeat/README.md)
- Polars H10 sensor (https://www.polar.com/us-en/sensors/h10-heart-rate-sensor).

#### PC
- Python 3.8 or higher

## File Structure
The data loggers are organized in the following structure:
- `data_loggers/`
  - `sensors/`
    - `ekg_logger.py`: EKG sensor data logger
    - `polars_h10_logger.py`: Polars H10 sensor data logger
  - `pc/`
    - `keyboard_logger.py`: Keyboard data logger
    - `mouse_logger.py`: Mouse data logger
  - `ps/`
    - `dualsense_logger.py`: Playstation controller data logger

## Data Files
Data files are stored in the [Recorded data directory](/raspberry_pico_sensors/motion_vibration/README.md) directory. Each data logger has its own subdirectory within the `gaming_health_data/recorded_data` directory.

## Next Steps
- Implement data collection from additional sources
- Enhance data logging capabilities
- Improve data processing and storage
- Develop real-time monitoring tools

## Contributing
Contributions are welcome! If you have any ideas, suggestions, or improvements, please open an issue or submit a pull request.
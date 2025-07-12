# Recorded Data
This directory contains the recorded data for the gaming health project. The recorded data is collected from various sources, including EKG, keyboard, and Playstation controller, Polars H10 sensor, and remote sensors connected via raspberry pi.

Data is currently divided into 4 categories:
1. SENSORS: Remote sensors connected via raspberry pi and Polars H10 sensor.
2. PC: Stuff connected to PC (mouse, keyboard).
3. PS: Playstation controller (Dualsense).
4. VIDEO: Annotated video recordings of the gaming sessions.

## Requirements
#### Sensors
- EKG sensor (setup: https://github.com/ondrabimka/raspberry_pico_sensors/blob/main/raspberry_pico_sensors/ekg_heartbeat/README.md)
- Polar H10 sensor.

#### PC
- Python 3.8 or higher
- Keyboard
- Mouse

#### PS
- Playstation controller (Dualsense)

## File Structure

The recorded data is organized in the following structure:
PC - contains data from the PC, including keyboard and mouse data.
PS - contains data from the Playstation controller (Dualsense).
SENSORS - contains ekg data captured using polar H10 sensor.
VIDEO - contains video annotated recordings of the gaming sessions. (Various annotation based on the game played)
APPLE_WATCH - contains data from the Apple Watch (if available). The data was extracted as described here: https://github.com/jameno/Simple-Apple-Health-XML-to-CSV



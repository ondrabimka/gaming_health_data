import time
from gaming_healt_data.data_loggers.PC.keyboard_logger import KeyboardLogger
from gaming_healt_data.data_loggers.PC.mouse_logger import MouseLogger

# File names for saving logs
mouse_log_file = 'gaming_healt_data//recorded_data//mouse_log.csv'
keyboard_log_file = 'gaming_healt_data//recorded_data//keyboard_log.csv'

# Create instances of MouseLogger and KeyboardLogger
mouse_logger = MouseLogger(mouse_log_file)
keyboard_logger = KeyboardLogger(keyboard_log_file)

# Start listeners for mouse and keyboard
mouse_logger.start()
keyboard_logger.start()

try:
    # Run for a specified duration (e.g., 60 seconds)
    time.sleep(1200)
except KeyboardInterrupt:
    # If KeyboardInterrupt (Ctrl+C) is received, stop the listeners
    pass

# Close log files
mouse_logger.close()
keyboard_logger.close()

import keyboard
import csv
import time

# Initialize a list to store key events
key_events = []

def on_key_event(e):
    # Record the time of the key event in milliseconds
    timestamp = int(time.time() * 1000)

    # Append key event details to the list
    key_events.append({'Key': e.name, 'Event Type': e.event_type, 'Timestamp': timestamp})

keyboard.hook(on_key_event)

try:
    # Wait for the 'Esc' key to be pressed
    keyboard.wait('esc')
finally:
    # Unhook the key event to stop the script
    keyboard.unhook_all()

    # Save key events to a CSV file
    csv_file_path = 'C://Users//Admin//Desktop//embedded_code//key_events.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        fieldnames = ['Key', 'Event Type', 'Timestamp']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write key events
        writer.writerows(key_events)

    print(f'Data saved to {csv_file_path}')


# %%
# import pandas as pd
# nwm = pd.read_csv("key_events.csv")
# Date_Time = pd.to_datetime(nwm.Timestamp, unit='ms')

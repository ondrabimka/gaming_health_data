import pygame
import csv
import time

# Initialize pygame for joystick handling
pygame.init()
pygame.joystick.init()

# Ensure a controller is connected
if pygame.joystick.get_count() == 0:
    print("No controller detected!")
    exit()

# Get the first detected controller
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Controller detected: {joystick.get_name()}")

# Open CSV file for writing
filename = "controller_inputs.csv"
with open(filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Button/Axis", "Value"])  # CSV Header

    print(f"Logging controller inputs to {filename}... Press Ctrl+C to stop.")

    try:
        while True:
            pygame.event.pump()  # Process events
            timestamp = time.time()

            # Log button presses
            for i in range(joystick.get_numbuttons()):
                if joystick.get_button(i):
                    writer.writerow([timestamp, f"Button {i}", 1])
                    print(f"{timestamp}, Button {i}, Pressed")
            # Log joystick movements (ignore small drift < 0.1) and ignore axis 4 and 5 (l2 and r2) if they are not pressed (-1.0)
            for i in range(joystick.get_numaxes()):
                axis_value = joystick.get_axis(i)
                if i in [4, 5] and axis_value == -1.0:
                    continue
                if abs(axis_value) > 0.1:
                    writer.writerow([timestamp, f"Axis {i}", round(axis_value, 3)])
                    print(f"{timestamp}, Axis {i}, {round(axis_value, 3)}")
            time.sleep(0.001)  # Prevent excessive CPU usage

    except KeyboardInterrupt:
        print("\nLogging stopped. CSV file saved.")

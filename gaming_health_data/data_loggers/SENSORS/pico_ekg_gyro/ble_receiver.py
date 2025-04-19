import asyncio
import struct
import sys
import time
from bleak import BleakClient, BleakScanner
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import numpy as np
import csv
import argparse

# Environmental Sensing service and characteristics UUIDs
ENV_SENSE_UUID = "0000181a-0000-1000-8000-00805f9b34fb"
TEMP_CHAR_UUID = "00002a6e-0000-1000-8000-00805f9b34fb"
EKG_CHAR_UUID = "00002a58-0000-1000-8000-00805f9b34fb"
ACCEL_CHAR_UUID = "00002ba1-0000-1000-8000-00805f9b34fb"
GYRO_CHAR_UUID = "00002ba2-0000-1000-8000-00805f9b34fb"


# Data storage
temperature_data = []
temperature_timestamps = []
ekg_data = []
ekg_timestamps = []
accel_data = {'x': [], 'y': [], 'z': []}
accel_timestamps = []
gyro_data = {'x': [], 'y': [], 'z': []}
gyro_timestamps = []

# For visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
line1, = ax1.plot([], [], 'r-')
line2, = ax2.plot([], [], 'b-')
line_accel_x, = ax3.plot([], [], 'r-', label='X')
line_accel_y, = ax3.plot([], [], 'g-', label='Y')
line_accel_z, = ax3.plot([], [], 'b-', label='Z')
line_gyro_x, = ax4.plot([], [], 'r-', label='X')
line_gyro_y, = ax4.plot([], [], 'g-', label='Y')
line_gyro_z, = ax4.plot([], [], 'b-', label='Z')

# Maximum points to display
MAX_POINTS = 100

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BLE receiver for Pico EKG and temperature data')
    parser.add_argument('--no-save', action='store_true', 
                      help='Disable saving data to CSV files (saving enabled by default)')
    parser.add_argument('--plot', action='store_true',
                      help='Enable real-time plotting (disabled by default)')
    return parser.parse_args()

def setup_plot():
    """Set up the matplotlib plot"""
    ax1.set_title('Temperature Data')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.grid(True)
    
    ax2.set_title('EKG Data')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('EKG Value')
    ax2.grid(True)
    
    ax3.set_title('Accelerometer Data')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (g)')
    ax3.grid(True)
    ax3.legend()
    
    ax4.set_title('Gyroscope Data')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angular Velocity (deg/s)')
    ax4.grid(True)
    ax4.legend()
    
    fig.tight_layout()


def update_plot(frame):
    """Update the plot with new data"""
    lines_to_return = []
    
    # Temperature plot
    if temperature_data:
        start_time = temperature_timestamps[0]
        times = [(t - start_time) for t in temperature_timestamps]
        line1.set_data(times[-MAX_POINTS:], temperature_data[-MAX_POINTS:])
        ax1.set_xlim(min(times[-MAX_POINTS:]), max(times[-MAX_POINTS:]))
        ax1.set_ylim(min(temperature_data[-MAX_POINTS:]) - 1, max(temperature_data[-MAX_POINTS:]) + 1)
    lines_to_return.append(line1)
    
    # EKG plot
    if ekg_data:
        start_time = ekg_timestamps[0]
        times = [(t - start_time) for t in ekg_timestamps]
        line2.set_data(times[-MAX_POINTS:], ekg_data[-MAX_POINTS:])
        ax2.set_xlim(min(times[-MAX_POINTS:]), max(times[-MAX_POINTS:]))
        ax2.set_ylim(min(ekg_data[-MAX_POINTS:]) - 0.1, max(ekg_data[-MAX_POINTS:]) + 0.1)
    lines_to_return.append(line2)
    
    # Accelerometer plot
    if accel_timestamps:
        start_time = accel_timestamps[0]
        times = [(t - start_time) for t in accel_timestamps]
        for line, data in [(line_accel_x, accel_data['x']), 
                          (line_accel_y, accel_data['y']), 
                          (line_accel_z, accel_data['z'])]:
            line.set_data(times[-MAX_POINTS:], data[-MAX_POINTS:])
        ax3.set_xlim(min(times[-MAX_POINTS:]), max(times[-MAX_POINTS:]))
        ax3.set_ylim(-2, 2)  # Typical range for acceleration in g
    lines_to_return.extend([line_accel_x, line_accel_y, line_accel_z])
    
    # Gyroscope plot
    if gyro_timestamps:
        start_time = gyro_timestamps[0]
        times = [(t - start_time) for t in gyro_timestamps]
        for line, data in [(line_gyro_x, gyro_data['x']), 
                          (line_gyro_y, gyro_data['y']), 
                          (line_gyro_z, gyro_data['z'])]:
            line.set_data(times[-MAX_POINTS:], data[-MAX_POINTS:])
        ax4.set_xlim(min(times[-MAX_POINTS:]), max(times[-MAX_POINTS:]))
        ax4.set_ylim(-250, 250)  # Typical range for gyro in deg/s
    lines_to_return.extend([line_gyro_x, line_gyro_y, line_gyro_z])
    
    return tuple(lines_to_return)


def temperature_notification_handler(sender, data):
    """Handle incoming temperature data"""
    # Temperature data is sent as float (4 bytes per value)
    for i in range(0, len(data), 4):
        if i + 4 <= len(data):
            value = struct.unpack('<f', data[i:i+4])[0]
            temperature_data.append(value)
            temperature_timestamps.append(time.time())
            print(f"Temperature: {value:.2f}°C")


def ekg_notification_handler(sender, data):
    """Handle incoming EKG data"""
    # EKG data is sent as 16-bit unsigned integers (2 bytes per value)
    for i in range(0, len(data), 2):
        if i + 2 <= len(data):
            value = struct.unpack('<H', data[i:i+2])[0]
            ekg_data.append(value)
            ekg_timestamps.append(time.time())
            print(f"EKG value: {value}")

def accel_notification_handler(sender, data):
    """Handle incoming accelerometer data"""
    # Each value is a float (4 bytes)
    if len(data) >= 12:  # 3 values * 4 bytes each
        x, y, z = struct.unpack('<fff', data[:12])
        accel_data['x'].append(x)
        accel_data['y'].append(y)
        accel_data['z'].append(z)
        accel_timestamps.append(time.time())
        print(f"Acceleration: X={x:.2f}, Y={y:.2f}, Z={z:.2f} g")

def gyro_notification_handler(sender, data):
    """Handle incoming gyroscope data"""
    # Each value is a float (4 bytes)
    if len(data) >= 12:  # 3 values * 4 bytes each
        x, y, z = struct.unpack('<fff', data[:12])
        gyro_data['x'].append(x)
        gyro_data['y'].append(y)
        gyro_data['z'].append(z)
        gyro_timestamps.append(time.time())
        print(f"Gyroscope: X={x:.2f}, Y={y:.2f}, Z={z:.2f} deg/s")

async def run(save_data=True, plot_data=False):
    """Main function to find and connect to the Pico"""
    print("Searching for Pico_Debug device...")
    
    device = await BleakScanner.find_device_by_name("Pico_Debug")
    
    if not device:
        print("Could not find Pico_Debug device.")
        return
    
    print(f"Found device: {device.name} ({device.address})")
    
    async with BleakClient(device) as client:
        print(f"Connected to {device.name}")
        
        # Subscribe to temperature notifications
        await client.start_notify(TEMP_CHAR_UUID, temperature_notification_handler)
        print("Subscribed to temperature notifications")
        
        # Subscribe to EKG notifications (if enabled on Pico)
        try:
            await client.start_notify(EKG_CHAR_UUID, ekg_notification_handler)
            print("Subscribed to EKG notifications")
        except Exception as e:
            print(f"Note: Could not subscribe to EKG notifications: {str(e)}")
            print("This is normal if the EKG sensor is not enabled on the Pico.")
        
        # Subscribe to accelerometer notifications
        try:
            await client.start_notify(ACCEL_CHAR_UUID, accel_notification_handler)
            print("Subscribed to accelerometer notifications")
        except Exception as e:
            print(f"Note: Could not subscribe to accelerometer notifications: {str(e)}")

        # Subscribe to gyroscope notifications
        try:
            await client.start_notify(GYRO_CHAR_UUID, gyro_notification_handler)
            print("Subscribed to gyroscope notifications")
        except Exception as e:
            print(f"Note: Could not subscribe to gyroscope notifications: {str(e)}")
        
        # Setup and start the plot if enabled
        ani = None
        if plot_data:
            setup_plot()
            ani = FuncAnimation(fig, update_plot, interval=100, blit=True)
            plt.show(block=False)
        
        # Keep the connection alive until user interrupts
        print("Receiving data. Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
                if plot_data:
                    plt.pause(0.1)  # Allow plot to update
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            # Stop notifications
            await client.stop_notify(TEMP_CHAR_UUID)
            try:
                await client.stop_notify(EKG_CHAR_UUID)
            except:
                pass
            
            if plot_data:
                plt.close()

            # Save data to CSV files if enabled
            if save_data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save temperature data
                if temperature_data:
                    temp_filename = f"temperature_data_{timestamp}.csv"
                    with open(temp_filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Timestamp', 'Temperature (°C)'])
                        for ts, temp in zip(temperature_timestamps, temperature_data):
                            writer.writerow([ts, temp])
                    print(f"Temperature data saved to {temp_filename}")
                
                # Save EKG data
                if ekg_data:
                    ekg_filename = f"ekg_data_{timestamp}.csv"
                    with open(ekg_filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Timestamp', 'EKG Value'])
                        for ts, ekg in zip(ekg_timestamps, ekg_data):
                            writer.writerow([ts, ekg])
                    print(f"EKG data saved to {ekg_filename}")
                
                # Save accelerometer data
                if save_data and accel_data['x']:
                    accel_filename = f"accelerometer_data_{timestamp}.csv"
                    with open(accel_filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Timestamp', 'X (g)', 'Y (g)', 'Z (g)'])
                        for ts, x, y, z in zip(accel_timestamps, 
                                              accel_data['x'], 
                                              accel_data['y'], 
                                              accel_data['z']):
                            writer.writerow([ts, x, y, z])
                    print(f"Accelerometer data saved to {accel_filename}")

                # Save gyroscope data
                if save_data and gyro_data['x']:
                    gyro_filename = f"gyroscope_data_{timestamp}.csv"
                    with open(gyro_filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Timestamp', 'X (deg/s)', 'Y (deg/s)', 'Z (deg/s)'])
                        for ts, x, y, z in zip(gyro_timestamps, 
                                              gyro_data['x'], 
                                              gyro_data['y'], 
                                              gyro_data['z']):
                            writer.writerow([ts, x, y, z])
                    print(f"Gyroscope data saved to {gyro_filename}")

if __name__ == "__main__":
    args = parse_arguments()
    save_data = not args.no_save  # True by default unless --no-save is specified
    plot_data = args.plot        # False by default unless --plot is specified
    asyncio.run(run(save_data=save_data, plot_data=plot_data))
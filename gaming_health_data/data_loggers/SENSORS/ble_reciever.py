import asyncio
import struct
import sys
import time
from bleak import BleakClient, BleakScanner
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Environmental Sensing service and characteristics UUIDs
ENV_SENSE_UUID = "0000181a-0000-1000-8000-00805f9b34fb"
TEMP_CHAR_UUID = "00002a6e-0000-1000-8000-00805f9b34fb"
EKG_CHAR_UUID = "00002a58-0000-1000-8000-00805f9b34fb"

# Data storage
temperature_data = []
temperature_timestamps = []
ekg_data = []
ekg_timestamps = []

# For visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
line1, = ax1.plot([], [], 'r-')
line2, = ax2.plot([], [], 'b-')

# Maximum points to display
MAX_POINTS = 100


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
    
    fig.tight_layout()


def update_plot(frame):
    """Update the plot with new data"""
    # Temperature plot
    if temperature_data:
        start_time = temperature_timestamps[0]
        times = [(t - start_time) for t in temperature_timestamps]
        
        # Update line data
        line1.set_data(times[-MAX_POINTS:], temperature_data[-MAX_POINTS:])
        
        # Adjust x and y limits
        ax1.set_xlim(min(times[-MAX_POINTS:]), max(times[-MAX_POINTS:]))
        ax1.set_ylim(min(temperature_data[-MAX_POINTS:]) - 1, max(temperature_data[-MAX_POINTS:]) + 1)
    
    # EKG plot
    if ekg_data:
        start_time = ekg_timestamps[0] if ekg_timestamps else 0
        times = [(t - start_time) for t in ekg_timestamps]
        
        # Update line data
        line2.set_data(times[-MAX_POINTS:], ekg_data[-MAX_POINTS:])
        
        # Adjust x and y limits
        ax2.set_xlim(min(times[-MAX_POINTS:]), max(times[-MAX_POINTS:]))
        ax2.set_ylim(min(ekg_data[-MAX_POINTS:]) - 1000, max(ekg_data[-MAX_POINTS:]) + 1000)
    
    return line1, line2


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


async def run():
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
        
        # Setup and start the plot
        setup_plot()
        ani = FuncAnimation(fig, update_plot, interval=100, blit=True)
        plt.show(block=False)
        
        # Keep the connection alive until user interrupts
        print("Receiving data. Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
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
            plt.close()


if __name__ == "__main__":
    asyncio.run(run())
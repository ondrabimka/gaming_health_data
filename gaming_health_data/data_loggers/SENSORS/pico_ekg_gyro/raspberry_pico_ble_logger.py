import struct
import machine
import time
import bluetooth
from ble_advertising import advertising_payload
from micropython import const
import utime
from machine import I2C, Pin
from mpu6500 import MPU6500, SF_G, SF_DEG_S


# Define BLE constants
_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_IRQ_GATTS_WRITE = const(3)

# Environmental Sensing service and characteristics
_ENV_SENSE_UUID = bluetooth.UUID(0x181A)
_TEMP_CHAR_UUID = bluetooth.UUID(0x2A6E)
_EKG_CHAR_UUID = bluetooth.UUID(0x2A58)  # Using pulse oximeter as it's somewhat related

# MPU6500 service and characteristics
_ACCEL_CHAR_UUID = bluetooth.UUID(0x2BA1)  # Using Acceleration characteristic UUID
_GYRO_CHAR_UUID = bluetooth.UUID(0x2BA2)   # Using Angular Velocity characteristic UUID

# Flag definitions
_FLAG_READ = const(0x0002)
_FLAG_NOTIFY = const(0x0010)


class BLEConnection:
    """
    A reusable Bluetooth Low Energy connection manager class using bluetooth library.
    Can be used to send any type of sensor data.
    """
    def __init__(self, name="Pico_Sensor"):
        self.name = name
        self.ble = bluetooth.BLE()
        self.ble.active(True)
        self.ble.irq(self.ble_irq)
        
        self.connected = False
        self.connection_handle = None
        self.data_handles = {}
        
        # Setup basic services
        self._register_services()
        
        # Start advertising
        self.advertise()
    
    def ble_irq(self, event, data):
        if event == _IRQ_CENTRAL_CONNECT:
            conn_handle, addr_type, addr = data
            self.connected = True
            self.connection_handle = conn_handle
            print("Connected to central device")
        elif event == _IRQ_CENTRAL_DISCONNECT:
            conn_handle, _, _ = data
            self.connected = False
            self.connection_handle = None
            print("Disconnected from central device")
            self.advertise()
        elif event == _IRQ_GATTS_WRITE:
            # Handle write events if needed
            pass
    
    def _register_services(self):
        # Setup service for sensor data
        
        # Add temperature characteristic
        temp_char = (
            _TEMP_CHAR_UUID,
            _FLAG_READ | _FLAG_NOTIFY,
        )
        
        # Add EKG characteristic
        ekg_char = (
            _EKG_CHAR_UUID,
            _FLAG_READ | _FLAG_NOTIFY,
        )

        accel_char = (
            _ACCEL_CHAR_UUID,
            _FLAG_READ | _FLAG_NOTIFY,
        )
        
        gyro_char = (
            _GYRO_CHAR_UUID,
            _FLAG_READ | _FLAG_NOTIFY,
        )
        
        # Create the environmental sensing service
        env_service = (
            _ENV_SENSE_UUID,
            (temp_char, ekg_char, accel_char, gyro_char),
        )
        
        # Register the service
        services = (env_service,)
        ((self.data_handles["temp"], 
        self.data_handles["ekg"],
        self.data_handles["accel"],
        self.data_handles["gyro"]),) = self.ble.gatts_register_services(services)
    
    def advertise(self):
        # Create advertising payload
        adv_data = advertising_payload(
            name=self.name,
            services=[_ENV_SENSE_UUID]
        )
        
        # Start advertising
        self.ble.gap_advertise(100, adv_data)
        print(f"Advertising as {self.name}")
    
    def send_data(self, data_type, data):
        """
        Send data over BLE
        data_type: String key for the type of data (e.g., "temp", "ekg")
        data: Bytearray of data to send
        """
        if not self.connected or data_type not in self.data_handles:
            return False
        
        handle = self.data_handles[data_type]
        self.ble.gatts_notify(self.connection_handle, handle, data)
        return True
    
    def is_connected(self):
        """Check if a client is connected"""
        return self.connected
    
    def stop(self):
        """Deactivate BLE"""
        self.ble.active(False)


class TemperatureSensor:
    """
    Class to handle the built-in temperature sensor on the RP2040
    """
    def __init__(self):
        # Initialize the ADC for the temperature sensor
        self.sensor = machine.ADC(4)
        
    def read_temperature(self):
        # Read the raw ADC value
        raw_value = self.sensor.read_u16()
        
        # Convert the raw value to temperature in Celsius
        # Formula from: https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf
        voltage = (raw_value * 3.3) / 65535
        temperature_c = 27 - (voltage - 0.706) / 0.001721
        
        return temperature_c


class EKGSensor:
    """
    Class to handle an EKG sensor connected to an ADC pin
    """
    def __init__(self, pin=26):
        # Initialize the ADC for the EKG sensor
        self.sensor = machine.ADC(pin)
        
    def read_value(self):
        # Read analog value from ADC
        analog_value = self.sensor.read_u16()

        # Convert the analog value to voltage (assuming 3.3V reference)
        voltage = (analog_value / 65535) * 3.3
        return voltage # , timestamp_us  # Return both voltage and timestamp


class MPU6500Sensor:
    """
    Class to handle the MPU6500 accelerometer and gyroscope sensor
    """
    def __init__(self, i2c_id: int = 0, scl_pin: int = 5, sda_pin: int = 4):

        # Initialize I2C
        self.i2c = I2C(i2c_id, scl=Pin(scl_pin), sda=Pin(sda_pin))
        self.sensor = MPU6500(self.i2c, accel_sf=SF_G, gyro_sf=SF_DEG_S)
        
        # Calibrate the sensor
        self.offset = self.sensor.calibrate(count=10000, delay=0)
        print(f"MPU6500 calibration offset: {self.offset}")
        print("mpu6500 id: " + hex(self.sensor.whoami))
        
    def read_acceleration(self):
        return self.sensor.acceleration
        
    def read_gyro(self):
        return self.sensor.gyro
        
    def read_temperature(self):
        return self.sensor.temperature

class SensorManager:
    """
    Class to manage multiple sensors and send their data over BLE
    """
    def __init__(self, ble_name="Pico_Sensors"):
        # Initialize BLE connection
        self.ble = BLEConnection(ble_name)
        
        # Sensor storage
        self.sensors = {}
        self.buffers = {}
        self.buffer_sizes = {}
        self.last_send_times = {}
        
        # By default, create the temperature sensor for testing
        self.add_sensor("temp", TemperatureSensor(), buffer_size=5, send_interval=2000)
    
    def add_sensor(self, sensor_id, sensor_object, buffer_size=20, send_interval=1000):
        """
        Add a sensor to be managed
        sensor_id: String identifier for the sensor
        sensor_object: Object with a read method
        buffer_size: Size of buffer before sending
        send_interval: Maximum time (ms) between sends
        """
        self.sensors[sensor_id] = sensor_object
        self.buffers[sensor_id] = []
        self.buffer_sizes[sensor_id] = buffer_size
        self.last_send_times[sensor_id] = time.ticks_ms()
    
    def read_and_buffer(self, sensor_id):
        """Read from a sensor and add to its buffer"""
        if sensor_id not in self.sensors:
            return
        
        # Read based on sensor type
        if sensor_id == "temp":
            value = self.sensors[sensor_id].read_temperature()
        elif sensor_id == "ekg":
            value = self.sensors[sensor_id].read_value()
        elif sensor_id == "accel":
            value = self.sensors[sensor_id].read_acceleration()
        elif sensor_id == "gyro":
            value = self.sensors[sensor_id].read_gyro()
        else:
            try:
                value = self.sensors[sensor_id].read_value()
            except AttributeError:
                try:
                    value = self.sensors[sensor_id].read()
                except:
                    print(f"Error reading from sensor {sensor_id}")
                    return
        
        # Add to buffer
        self.buffers[sensor_id].append(value)
        
        # Check if we should send
        current_time = time.ticks_ms()
        if (len(self.buffers[sensor_id]) >= self.buffer_sizes[sensor_id] or 
            time.ticks_diff(current_time, self.last_send_times[sensor_id]) > self.buffer_sizes[sensor_id]):
            self.send_sensor_data(sensor_id)
            self.last_send_times[sensor_id] = current_time
    
    def send_sensor_data(self, sensor_id):
        """Send buffered data for a specific sensor"""
        if sensor_id not in self.buffers or not self.buffers[sensor_id]:
            return
        
        data = bytearray()
        
        # Format data based on sensor type
        if sensor_id in ["accel", "gyro"]:
            # Pack all acceleration/gyroscope readings (3 floats each)
            for value in self.buffers[sensor_id]:
                data.extend(struct.pack('<fff', value[0], value[1], value[2]))
        elif sensor_id == "temp":
            # Pack temperature readings as floats
            for value in self.buffers[sensor_id]:
                data.extend(struct.pack('<f', value))
        else:
            # For EKG, pack as floats
            for value in self.buffers[sensor_id]:
                data.extend(struct.pack('<f', float(value)))
        
        # Send the data
        sent = self.ble.send_data(sensor_id, data)
        if sent:
            print(f"Sent {len(self.buffers[sensor_id])} {sensor_id} readings")
        
        # Clear buffer
        self.buffers[sensor_id] = []
    
    def run(self, sample_interval_ms=100):
        """Main loop to read from all sensors at the specified interval"""
        print("Sensor Manager running...")
        
        try:
            while True:
                # Read from each sensor
                for sensor_id in self.sensors:
                    self.read_and_buffer(sensor_id)
                
                # Small delay to prevent hogging the CPU
                time.sleep_ms(sample_interval_ms)
                
        except KeyboardInterrupt:
            # Clean shutdown
            self.ble.stop()
            print("Sensor Manager stopped")


# Example usage
if __name__ == "__main__":
    # Create a sensor manager with default temperature sensor
    manager = SensorManager(ble_name="Pico_Debug")

    # Add MPU6500 sensor
    mpu = MPU6500Sensor()
    manager.add_sensor("accel", mpu, buffer_size=20, send_interval=400)
    manager.add_sensor("gyro", mpu, buffer_size=20, send_interval=400)
    
    # To add an EKG sensor (uncomment when ready)
    ekg = EKGSensor(pin=26)  # Adjust pin as needed
    manager.add_sensor("ekg", ekg, buffer_size=20, send_interval=400)
    
    # Start the main loop
    manager.run(sample_interval_ms=400)
from machine import I2C, Pin
from mpu9250 import MPU9250
from gaming_health_data.data_loggers.keyboard_logger import InputDeviceLogger

class MouseSensorLogger(InputDeviceLogger):

    def __init__(self, file_name, sda_pin, scl_pin):
        super().__init__(file_name)
        self.i2c = I2C(sda=Pin(sda_pin), scl=Pin(scl_pin))
        self.sensor = MPU9250(self.i2c)

    def read_acceleration(self):
        try:
            accel = self.sensor.acceleration
            self.logger.log('MouseSensor', 'Acceleration', f'X: {accel[0]}, Y: {accel[1]}, Z: {accel[2]}')
        except Exception as e:
            print(f"Error reading acceleration: {e}")

    def read_gyro(self):
        gyro = self.sensor.gyro
        self.logger.log('MouseSensor', 'Gyro', f'X: {gyro[0]}, Y: {gyro[1]}, Z: {gyro[2]}')


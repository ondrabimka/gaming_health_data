from gaming_health_data.data_loggers.input_logger import InputLogger
import mouse
import threading
import time

class MouseLogger:
    def __init__(self, file_name):
        self.logger = InputLogger(file_name)
        self.thread = threading.Thread(target=self.start_listener, daemon=True)
        self.click_thread = threading.Thread(target=self.click_listener, daemon=True)

    def start_listener(self):
        while True:
            x, y = mouse.get_position()
            self.logger.log('Mouse', 'Move', f'X: {x}, Y: {y}')
            time.sleep(0.1)  # Adjust sleep time as needed
            
    def click_listener(self):
        mouse.on_click(self.logger.log, ('Mouse', 'Click', 'Left'))
        mouse.on_right_click(self.logger.log, ('Mouse', 'Click', 'Right'))
        
    def start(self):
        self.thread.start()
        self.click_thread.start()

    def join(self):
        pass  # No need to join an infinite loop

    def close(self):
        self.logger.close()


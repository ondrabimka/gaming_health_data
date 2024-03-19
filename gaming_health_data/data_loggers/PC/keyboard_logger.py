from gaming_health_data.data_loggers.input_logger import InputLogger
import keyboard
import threading

class KeyboardLogger:
    def __init__(self, file_name):
        self.logger = InputLogger(file_name)
        self.thread = threading.Thread(target=self.start_listener, daemon=True)
        self.pressed_keys = set()

    def start_listener(self):
        keyboard.hook(self.log_key)

    def log_key(self, key_event):
        action = 'Press' if key_event.event_type == keyboard.KEY_DOWN else 'Release'
        key = key_event.name
        
        if action == 'Press' and key not in self.pressed_keys:
            self.logger.log('Keyboard', action, f'Key: {key}')
            self.pressed_keys.add(key)
        if action == 'Release':
            self.pressed_keys.remove(key)
            self.logger.log('Keyboard', action, f'Key: {key}')
        

    def start(self):
        self.thread.start()

    def join(self):
        pass  # No need to join an infinite loop

    def close(self):
        self.logger.close()
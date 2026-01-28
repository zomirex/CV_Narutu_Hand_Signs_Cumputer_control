# hand/executor.py
import pyautogui
from modularV0.utils import logger

# برای دستورات خاص می‌توانید این دیکشنری را تغییر دهید
DEFAULT_COMMANDS = {
    "open_palm":    lambda: pyautogui.press('volumeup'),
    "thumbs_up":         lambda: pyautogui.press('volumedown'),
    "fist":   lambda : pyautogui.hotkey('alt', 'tab'),
    "stop":         lambda: pyautogui.press('stop'),
    "scroll_up":    lambda: pyautogui.press('scroll_up'),
    "scroll_down":  lambda: pyautogui.press('scroll_down'),
}

class GestureExecutor:
    def __init__(self, command_map=None):
        """
        command_map: dict mapping gesture_name -> callable
        If a string is provided instead of a callable, it will be wrapped
        into a lambda that calls pyautogui.press(string).
        """
        # If no map supplied, use the default
        self.command_map = command_map or DEFAULT_COMMANDS.copy()

        # Normalize: make sure every value is callable
        for gesture, action in list(self.command_map.items()):
            if not callable(action):
                # Assume action is a string key for pyautogui.press
                if isinstance(action, str):
                    self.command_map[gesture] = lambda key=action: pyautogui.press(key)
                else:
                    logger.warning(
                        f"Unsupported action type for gesture '{gesture}': {type(action)}"
                    )
                    # Remove or replace with no‑op
                    self.command_map[gesture] = lambda: None

    def execute(self, gesture_name):
        """
        Execute the command mapped to the given gesture_name.
        """
        action = self.command_map.get(gesture_name)
        if action is None:
            logger.debug(f"No command mapped for gesture: {gesture_name}")
            return

        logger.info(f"Executing command for gesture: {gesture_name}")
        try:
            action()
        except Exception as e:
            logger.error(f"Error executing command for {gesture_name}: {e}")

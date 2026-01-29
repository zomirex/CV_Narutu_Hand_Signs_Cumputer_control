# hand/executor.py
import pyautogui
from modularV0.utils import logger

# برای دستورات خاص می‌توانید این دیکشنری را تغییر دهید
DEFAULT_COMMANDS = {
    "open_palm": lambda key="volumeup": pyautogui.press(key),
    "thumbs_up": lambda key="volumedown": pyautogui.press(key),
    "fist": lambda key="space": pyautogui.press(key),
    "scroll_up": lambda amount=200: pyautogui.scroll(amount),
    "scroll_down": lambda amount=-200: pyautogui.scroll(amount),
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
            print(self.command_map["fist"])
            action()
        except Exception as e:
            logger.error(f"Error executing command for {gesture_name}: {e}")

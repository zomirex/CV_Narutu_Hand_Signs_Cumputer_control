# hand/executor.py
import pyautogui
from modularV0.utils import logger

# برای دستورات خاص می‌توانید این دیکشنری را تغییر دهید
DEFAULT_COMMANDS = {
    "open_palm": lambda: pyautogui.press('volumeup'),
    "thumbs_up": lambda: pyautogui.press('volumedown'),
    "fist": lambda: pyautogui.press('space'),
    "scroll_up": lambda: pyautogui.scroll(200),
    "scroll_down": lambda: pyautogui.scroll(-200),
}

class GestureExecutor:
    def __init__(self, command_map=None):
        self.command_map = command_map or {
            "open_palm": self.volume_up,
            "thumbs_up": self.volume_down,
            "fist": self.alt_tab,
            "index_up": self.scroll_up,
            "scroll_down": self.scroll_down,
        }

    def volume_up(self):
        pyautogui.press("volumeup")

    def volume_down(self):
        pyautogui.press("volumedown")

    def alt_tab(self):
        pyautogui.hotkey("alt", "tab")

    def scroll_up(self):
        pyautogui.scroll(200)

    def scroll_down(self):
        pyautogui.scroll(-200)

    def execute(self, gesture_name):
        action = self.command_map.get(gesture_name)

        if not action:
            return
        logger.info(f"Executing command for gesture: {gesture_name}")
        try:
            action()
        except Exception as e:
            logger.error(f"Error executing command for {gesture_name}: {e}")
            print(e)
# hand/executor.py
import pyautogui
from modularV0.utils import logger

# برای دستورات خاص می‌توانید این دیکشنری را تغییر دهید
Mouse_Commands = {
    "Lclick": lambda pressed,x,y: handle_lclick(pressed,x,y,'left'),
    "Rclick": lambda pressed,x,y: handle_lclick(pressed,x,y,'right'),
    "Mclick": lambda pressed,x,y: handle_lclick(pressed,x,y,'middle'),
    "move": lambda x,y: pyautogui.move(x,y),
    "scroll_up": lambda amount: pyautogui.scroll(amount),
    "scroll_down": lambda amount: pyautogui.scroll(-1*amount),
}
Hardware_Commands={
    "volumeup": lambda key="volumeup": pyautogui.press(key),
    "volumedown": lambda key="volumedown": pyautogui.press(key),
    "brightnessup": lambda key="brightnessup": pyautogui.press(key),
    "brightnessdown": lambda key="brightnessdown": pyautogui.press(key),
}


def handle_lclick(pressed,x,y,buttonn):
    try:
        if pressed:
            pyautogui.mouseDown(button=buttonn)
            pyautogui.move(x, y)
            print(buttonn+" mouse DOWN")
        else:
            pyautogui.mouseUp(button=buttonn)
            pyautogui.move(x, y)
            print(buttonn+" mouse UP")
    except Exception as e:
        print(f"Error: {e}")


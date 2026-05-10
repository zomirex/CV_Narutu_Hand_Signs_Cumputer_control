# hand/executor.py
import time
from typing import List, Tuple

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

class GestureExecutor:
    def __init__(self, command_map=None):
        self.prev_hand_x= None
        self.prev_hand_y = None
        self.last_tab_time=0
        self.MOVE_THRESHOLD_NORM=0.008
        self._COOLDOWN=0.1
        self.alt_held=False
        self.alt_mode_counter = 0
        self.exit_counter = 0
        # اگر کاربر مقداری داد یا پیش‌فرض را گرفت
        raw_map = command_map or Hardware_Commands

        # تبدیل تمام مقادیر به callable
        self.command_map = {}
        for k, v in raw_map.items():
            if callable(v):
                self.command_map[k] = v
            elif isinstance(v, str):
                # فرض می‌کنیم رشته همان نام کلید است
                self.command_map[k] = lambda key=v: pyautogui.press(key)
            else:
                raise ValueError(f"Unsupported command for {k!r}: {type(v)}")
    # def execute_XMove(self,HandGesture,IncreaseCommand,DecreaseCommand,Hands_xy):
    def execute_XMove_alt(self,HandGesture,Hands_xy):
        current_hand_x=Hands_xy[0][0]

        # print(self.alt_mode_counter )
        # print(self.exit_counter )
        # print("fds")
        if HandGesture=="pinch":
            self.alt_mode_counter += 1
            self.exit_counter = 0

            if self.alt_mode_counter >= 2 and not self.alt_held:
                pyautogui.keyDown('alt')
                self.alt_held = True
                self.prev_hand_x = current_hand_x
                print("🡐 Alt فشرده شد")
        elif self.alt_held:
            self.exit_counter += 1
            self.alt_mode_counter=0
            if self.exit_counter >= 2:
                print("✓ Alt رها شد")
                pyautogui.keyUp('alt')
                self.alt_held = False
                self.prev_hand_x = None
                self.exit_counter = 0
        else:
            self.alt_mode_counter=0
            self.exit_counter = 0
        if self.prev_hand_x is not None:
            diff = current_hand_x - self.prev_hand_x

            current_time = time.time()
            if (current_time - self.last_tab_time) > self._COOLDOWN and self.alt_held==True:
                if diff > self.MOVE_THRESHOLD_NORM :
                    pyautogui.press('tab')
                    self.prev_hand_x = current_hand_x
                    self.last_tab_time = current_time
                    print(current_hand_x)
                    print("→ برنامه بعدی")
                elif diff < -self.MOVE_THRESHOLD_NORM :
                    pyautogui.hotkey('shift', 'tab')
                    self.prev_hand_x = current_hand_x
                    self.last_tab_time = current_time
                    print(current_hand_x)
                    print("← برنامه قبلی")
        else:
            self.prev_hand_x = current_hand_x

    def process_gesture(self,
                        gesture: str,
                        hands_xy: List[Tuple[float, float]]) -> None:
        """
        gesture : string مثل "pinch", "open", "scroll"
        hands_xy: لیست (x, y) برای دست‌ها؛ در این مثال فقط دست اول (index 0) مهم است
        """
        if not hands_xy:
            return

        cur_x, cur_y,z = hands_xy[0]

        # ----------------- تغییر صدا (X movement) -----------------
        if gesture == "fist":
            if self.prev_hand_x is not None:
                diff_x = cur_x - self.prev_hand_x
                if abs(diff_x) >= self.MOVE_THRESHOLD_NORM:
                   if(diff_x>0):
                       pyautogui.keyDown('volumeup')
                       print(diff_x)
                   else:
                       pyautogui.keyDown('volumedown')
                       print(diff_x)
            self.prev_hand_x = cur_x

        # ----------------- تغییر روشنایی (Y movement) -----------------
        elif gesture == "spiderman":
            if self.prev_hand_y is not None:
                diff_y = cur_y - self.prev_hand_y
                if abs(diff_y) >= self.MOVE_THRESHOLD_NORM:
                    if (diff_y > 0):
                        pyautogui.keyDown('brightnessup')
                    else:
                        pyautogui.keyDown('brightnessdown')
            self.prev_hand_y = cur_y

        # ----------------- اسکرول (scroll gesture) -----------------
        elif gesture == "index_up":
            if self.prev_hand_y is not None:
                diff_y = cur_y - self.prev_hand_y
                if abs(diff_y) >= self.MOVE_THRESHOLD_NORM:
                    if (diff_y > 0):
                        pyautogui.scroll(10)
                    else:
                        pyautogui.scroll(-10)
            self.prev_hand_y = cur_y







import time
from typing import List, Tuple

import pyautogui
from modularV0.utils import logger

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
        """
        پارامتر `command_map` می‌تواند یک دیکشنری باشد که کلیدهای
        Gesture را به توابع یا کلیدهای pyautogui نگاشت می‌دهد.
        اگر مقدار داده نشود، از پیش‌فرض `Hardware_Commands` استفاده می‌شود.

        ویژگی‌های نمونه:
            - prev_hand_x / prev_hand_y : مختصات قبلی دست برای محاسبه حرکت
            - last_tab_time : زمان آخرین فشرده‌سازی کلید Tab (برای cooldown)
            - MOVE_THRESHOLD_NORM : حداقل تغییر مختصات که باعث اجرای فرمان می‌شود
            - _COOLDOWN : زمان خنک‌سازی بین فرمان‌های مشابه
            - alt_held : آیا کلید Alt فشرده است یا خیر
            - alt_mode_counter / exit_counter : شمارش حالات Alt
        """
        self.prev_hand_x   = None
        self.prev_hand_y   = None
        self.last_tab_time = 0
        self.MOVE_THRESHOLD_NORM = 0.008
        self._COOLDOWN = 0.1
        self.alt_held = False
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

    # ------------------------------------------------------------------
    # متد اجرای حرکت X با Alt برای انتخاب پنجره (مثلاً Tab/Shift+Tab)
    # ------------------------------------------------------------------
    def execute_XMove_alt(self, HandGesture, Hands_xy):
        """
        پارامترها:
            HandGesture : نوع Gesture (مثلاً "pinch")
            Hands_xy    : لیست مختصات (x, y) برای دست‌ها؛ فقط دست اول استفاده می‌شود
        """
        current_hand_x = Hands_xy[0][0]      # مختصات X دست

        # اگر Gesture «pinch» باشد، Alt را فشرده و آماده‌ی کنترل می‌کنیم
        if HandGesture == "pinch":
            self.alt_mode_counter += 1
            self.exit_counter = 0

            # پس از دو بار pinch و در صورتی که هنوز Alt فشرده نیست
            if self.alt_mode_counter >= 2 and not self.alt_held:
                pyautogui.keyDown('alt')
                self.alt_held = True
                self.prev_hand_x = current_hand_x
                print("🡐 Alt فشرده شد")

        # اگر Alt فعلاً فشرده است ولی Gesture دیگر نیست
        elif self.alt_held:
            self.exit_counter += 1
            self.alt_mode_counter = 0
            if self.exit_counter >= 2:
                print("✓ Alt رها شد")
                pyautogui.keyUp('alt')
                self.alt_held = False
                self.prev_hand_x = None
                self.exit_counter = 0
        else:
            self.alt_mode_counter = 0
            self.exit_counter = 0

        # اگر مختصات قبلی وجود دارد، تفاوت را محاسبه می‌کنیم
        if self.prev_hand_x is not None:
            diff = current_hand_x - self.prev_hand_x

            current_time = time.time()
            # در صورتی که cooldown گذشت و Alt فشرده است
            if (current_time - self.last_tab_time) > self._COOLDOWN and self.alt_held:
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
            # اگر هنوز مختصات قبلی ثبت نشده بود، آن را ذخیره می‌کنیم
            self.prev_hand_x = current_hand_x

    # ------------------------------------------------------------------
    # متد عمومی برای پردازش هر Gesture (مثلاً تغییر صدا، روشنایی، اسکرول)
    # ------------------------------------------------------------------
    def process_gesture(self,
                        gesture: str,
                        hands_xy: list) -> None:
        """
        پارامترها:
            gesture : نام Gesture شناسایی شده (مثلاً "fist", "spiderman", "index_up")
            hands_xy: لیست (x, y) برای دست‌ها؛ در این مثال فقط دست اول (index 0) مهم است

        توضیح کلی:
            - برای هر Gesture نوعی عملیات خاص تعریف می‌شود
            - برای حرکت X یا Y از اختلاف مختصات قبلی استفاده می‌شود
            - در صورت گذشت زمان کافی، دستورات pyautogui اجرا می‌شوند
        """
        if not hands_xy:
            return

        cur_x, cur_y, z = hands_xy[0]

        # ------------------------------------------------------------------
        # ۱. تغییر صدا با حرکت دست در محور X (Gesture «fist»)
        # ------------------------------------------------------------------
        if gesture == "fist":
            if self.prev_hand_x is not None:
                diff_x = cur_x - self.prev_hand_x
                if abs(diff_x) >= self.MOVE_THRESHOLD_NORM:
                    if diff_x > 0:
                        pyautogui.keyDown('volumeup')
                        print(diff_x)
                    else:
                        pyautogui.keyDown('volumedown')
                        print(diff_x)
            self.prev_hand_x = cur_x

        # ------------------------------------------------------------------
        # ۲. تغییر روشنایی با حرکت دست در محور Y (Gesture «spiderman»)
        # ------------------------------------------------------------------
        elif gesture == "spiderman":
            if self.prev_hand_y is not None:
                diff_y = cur_y - self.prev_hand_y
                if abs(diff_y) >= self.MOVE_THRESHOLD_NORM:
                    if diff_y > 0:
                        pyautogui.keyDown('brightnessup')
                    else:
                        pyautogui.keyDown('brightnessdown')
            self.prev_hand_y = cur_y

        # ------------------------------------------------------------------
        # ۳. اسکرول با حرکت دست در محور X (Gesture «index_up»)
        # ------------------------------------------------------------------
        elif gesture == "index_up":
            if self.prev_hand_x is not None:
                diff_x = cur_x - self.prev_hand_x
                if abs(diff_x) >= self.MOVE_THRESHOLD_NORM:
                    if diff_x > 0:
                        pyautogui.scroll(10)
                    else:
                        pyautogui.scroll(-10)
            self.prev_hand_x = cur_x




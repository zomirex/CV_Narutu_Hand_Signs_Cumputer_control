# config.py
# ---------------------------------------------
# تنظیمات اصلی (به راحتی قابل تغییر)
# ---------------------------------------------
CAMERA_ID = 0                     # شماره‌ی دوربین (0 = وب‌کم داخلی)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DETECTION_CONFIDENCE = 0.4
TRACKING_CONFIDENCE = 0.4

# ---------------------------------------------
# گِیِست‌ها و دستورات متناظر
# ---------------------------------------------
# در این دیکشنری، کلید گِیِست و مقدار دستور متناظر است
GESTURE_COMMANDS = {
    "thumbs_up": "volume_up",          # بالا بردن حجم
    "thumbs_down": "volume_down",      # کم کردن حجم
    "open_palm": "play_pause",         # پلی/پاز
    "fist": "stop",                    # متوقف کردن
    # اضافه کنید گِیِست‌های خودتان
}
# config.py
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Configurations:
    # دسترسی‌ها
    max_hands: int = 2
    detection_conf: float = 0.7
    tracking_conf: float = 0.7
    CAMERA_ID: int = 0                
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480

    # پارامترهای دقت (در صورت تمایل می‌توانید از 2 مجموعه استفاده کنید)
    DETECTION_CONFIDENCE: float = 0.4
    TRACKING_CONFIDENCE: float = 0.4

    # آستانه‌ی تشخیص انگشت (مقدار به میلی‌متر)
    FOLD_THRESHOLD: Dict[str, int] = field(
        default_factory=lambda: {
            'Thumb': 100,
            'Index': 40,
            'Middle': 40,
            'Ring': 40,
            'Pinky': 40,
        }
    )

    theme: str = "dark"      # dark / light

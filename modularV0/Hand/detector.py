
"""
کلاس `HandDetector`:
- با استفاده از MediaPipe، دست‌های موجود در تصویر را تشخیص می‌دهد،
- نقاط کلیدی (landmarks) را استخراج می‌کند و طرف دست (Left/Right) را برمی‌گرداند،
- قابلیت تغییر پارامترهای اولیه (max_hands، detection_conf، tracking_conf) را در زمان اجرا فراهم می‌کند.
"""

import cv2
import mediapipe as mp


class HandDetector:
    """
    تشخیص دست، استخراج نقاط کلیدی (landmarks) و تشخیص طرف (left/right).
    """

    # ------------------------------------------------------------------
    # سازنده
    # ------------------------------------------------------------------
    def __init__(self, max_hands=2, detection_conf=0.7, tracking_conf=0.7):
        """
        پارامترها:
            max_hands        : حداکثر تعداد دستی که می‌خواهیم تشخیص دهیم.
            detection_conf   : دقت حداقل برای تشخیص دست.
            tracking_conf    : دقت حداقل برای ردیابی بعدی دست.
        """
        self.max_hands = max_hands

        # شیء MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        # ابزار رسم هندسه
        self.mp_draw = mp.solutions.drawing_utils

    # ------------------------------------------------------------------
    # تغییر پارامترهای تنظیمات در زمان اجرا
    # ------------------------------------------------------------------
    def Config(self, Config):
        """
        پارامتر `Config` یک شیء (معمولاً از فایل JSON خوانده‌شده) است که
        شامل ویژگی‌های `max_hands`, `detection_conf`, `tracking_conf` می‌شود.
        این متد، شیء `self.hands` را با مقادیر جدید بازسازی می‌کند.
        """
        self.max_hands = Config.max_hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=Config.max_hands,
            min_detection_confidence=Config.detection_conf,
            min_tracking_confidence=Config.tracking_conf,
        )

    # ------------------------------------------------------------------
    # پردازش یک فریم BGR
    # ------------------------------------------------------------------
    def process(self, frame):
        """
        ورودی: تصویر به فرمت BGR (OpenCV)
        خروجی: (فریم) + لیست دست‌ها و نقاط کلیدی

        فرایند:
            1. تبدیل تصویر به RGB برای MediaPipe.
            2. فراخوانی `self.hands.process`.
            3. اگر دستی پیدا شد، نقاط کلیدی را به صورت لیست ۳‑بعدی استخراج می‌کنیم
               و طرف دست (Left/Right) را دریافت می‌کنیم.
            4. هندسه‌های MediaPipe را روی فریم رسم می‌کنیم تا نمایش بهتری داشته باشیم.
        """
        # تبدیل BGR → RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        # لیست داده‌های دست‌ها
        hand_data = []  # هر ورودی: {'landmarks': np.array, 'handedness': 'Left'|'Right'}

        if results.multi_hand_landmarks:
            # برای هر دست، نقاط و طرف آن را استخراج می‌کنیم
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                lm_list = []
                for lm in hand_landmarks.landmark:
                    lm_list.append([lm.x, lm.y, lm.z])  # مختصات نرمالیزه

                hand_data.append({
                    'landmarks': lm_list,
                    'handedness': handedness.classification[0].label,  # "Left" یا "Right"
                })

                # رسم نقاط و اتصالات روی فریم (خاکی به صورت سبز و سرخ)
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2)
                )

        # بازگشت فریم (تبدیل شده) و داده‌های دست
        return frame, hand_data

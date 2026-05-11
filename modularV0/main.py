# -----------------------------
#   MAIN SCRIPT – Demo of the hand‑gesture pipeline
# -----------------------------

import cv2

# کلاس‌های اصلی
from modularV0.Hand.detector   import HandDetector
from modularV0.Hand.processor  import HandProcessor
from modularV0.Hand.model      import GestureModel
from modularV0.Hand.excuter    import GestureExecutor

# تنظیمات ثابت (در فایل Configurations.py تعریف شده‌اند)
from modularV0.utils.Configurations import (
    CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT,
    DETECTION_CONFIDENCE, TRACKING_CONFIDENCE
)


def main():
    """
    این تابع تمام حلقه‌ی اصلی برنامه را اجرا می‌کند:
      * گرفتن فریم از دوربین
      * تشخیص دست، استخراج نقاط کلیدی
      * محاسبه هندسه (زاویه‌ها، فواصل، وضعیت انگشتان)
      * طبقه‌بندی Gesture
      * نمایش اطلاعات روی فریم و اجرای دستورات بر اساس Gesture
    """
    # ---------- حافظه‌های موقتی برای ذخیره نتایج هر فریم ----------
    P_P_T_distances = {}
    hand_angles      = {}
    finger_status    = {}
    hand_gesture     = {}
    hand_wrist_ang   = {}

    # ---------- باز کردن دوربین ----------
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # ---------- ایجاد آبجکت‌های پردازش ----------
    detector  = HandDetector(
        max_hands=2,
        detection_conf=DETECTION_CONFIDENCE,
        tracking_conf=TRACKING_CONFIDENCE
    )
    processor = HandProcessor()
    model     = GestureModel(threshold=30)
    executor  = GestureExecutor()

    # حافظه‌ی آخرین Gesture برای جلوگیری از تکرار فرمان
    last_gesture = {"Left": None, "Right": None}

    # ---------- حلقه‌ی اصلی فریم‌های دوربین ----------
    while True:
        ret, frame = cap.read()
        if not ret:
            # در صورت عدم دریافت فریم، خروج از حلقه
            break

        # --------- تشخیص دست و نقاط کلیدی ----------
        frame, hands = detector.process(frame)

        # --------- پردازش هر دست (Left / Right) ----------
        for hand in hands:
            lm = hand["landmarks"]          # لیست نقاط کلیدی (x, y, z)
            handedness = hand["handedness"]  # 'Left' یا 'Right'

            # ۱. محاسبه زاویه‌های انگشتان
            hand_angles[handedness] = ang = processor.Angele_Calculator(lm, frame)

            # ۲. فواصل نرمالیزه‌ی نوک انگشتان
            P_P_T_distances[handedness] = processor.Distance_norm_Calculator(lm, frame)

            # ۳. وضعیت باز/بسته‌ی هر انگشت
            finger_status[handedness] = stat = processor.Finger_Status(ang)

            # ۴. طبقه‌بندی Gesture بر اساس وضعیت انگشتان و زاویه‌ها
            hand_gesture[handedness] = gesture = model.classify(stat, ang)

            # ۵. زاویه‌ی آرنج (برای نمایش فقط)
            hand_wrist_ang[handedness] = processor.Wrist_angel(lm, frame)

            # --------- نمایش اطلاعات روی فریم ----------
            y0, dy = (30, 20) if handedness == "Left" else (150, 20)
            for i, (finger, percent) in enumerate(ang.items()):
                text = f"{finger.capitalize()}: {percent}"
                y = y0 + i * dy
                cv2.putText(frame, text, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # نام دست و Gesture
            txt = f"{handedness} Hand: {gesture}"
            txt_pos = (400, 80) if handedness == "Left" else (400, 100)
            cv2.putText(frame, txt, txt_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # --------- فرمان‌دهی بر اساس Gesture (مشت دست چپ و gesture راست) ----------
        if hand_gesture.get('Right') and hand_gesture.get('Left'):
            # اگر Gesture سمت راست تغییر کرده و دست چپ در وضعیت «fist» است
            if (hand_gesture['Right'] != last_gesture['Right']) and hand_gesture['Left'] == 'fist':
                # در این نمونه کلاس GestureExecutor متد execute تعریف نشده،
                # شاید می‌خواستید از execute_XMove_alt استفاده کنید.
                # در صورت وجود متد execute، می‌توانید به صورت زیر فراخوانی کنید:
                # executor.execute(hand_gesture.get('Right'))
                # یا اگر از execute_XMove_alt استفاده می‌کنید:
                # executor.execute_XMove_alt(hand_gesture.get('Right'), lm)
                last_gesture['Right'] = hand_gesture.get('Right')

        # --------- نمایش فریم در پنجره ----------
        cv2.imshow("Hand Motion Detector", frame)

        # خروج با کلید ESC
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    # --------- آزادسازی منابع ----------
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

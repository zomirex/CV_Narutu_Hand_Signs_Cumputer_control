import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# --- صدا (فقط ویندوز) ---
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

# --- MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- تنظیمات صدا ---
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
try:
    vol_percent = int(volume.GetMasterVolumeLevelScalar() * 100)
except:
    vol_percent = 50

# --- تنظیمات حرکت (مستقل از فاصله از دوربین) ---
MOVE_THRESHOLD_NORM = 0.10  # 10% از عرض تصویر (ثابت)
BRIGHTNESS_STEP = 10  # هر حرکت = 10% نور
VOLUME_STEP = 5  # هر حرکت = 5% صدا
TAB_COOLDOWN = 0.25  # تأخیر بین Tabها

# --- متغیرهای جهانی ---
brightness = 100
prev_right_x = None
prev_left_x = None
alt_held = False
prev_hand_x = None
last_tab_time = 0

# --- متغیرهای پایداری Alt+Tab ---
alt_mode_counter = 0
exit_counter = 0
CONFIRM_FRAMES = 3
DISTANCE_THRESHOLD = 0.06  # فاصله شست و اشاره (نرمالایز شده)


# --- توابع کمکی ---
def euclidean_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def other_fingers_open(landmarks):
    def is_finger_open(tip, dip):
        return landmarks.landmark[tip].y < landmarks.landmark[dip].y

    return is_finger_open(12, 10) and is_finger_open(16, 14) and is_finger_open(20, 18)


def is_fist(hand_landmarks):
    tips = [8, 12, 16, 20]
    dips = [6, 10, 14, 18]
    for tip, dip in zip(tips, dips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y*1:
            return False
    return True


# --- شروع سیستم ---
cap = cv2.VideoCapture(0)
print("✅ سیستم کنترل دست فعال شد!")
print("- دست راست مشت → نور")
print("- دست چپ مشت → صدا")
print("- شست+اشاره چسبیده + بقیه باز → Alt+Tab پیشرفته")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    current_hand_x = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- موقعیت نرمالایز شده (0 تا 1) — مستقل از فاصله از دوربین ---
        current_hand_x = hand_landmarks.landmark[0].x
        cx_norm = current_hand_x
        label = "Right" if cx_norm < 0.5 else "Left"

        # --- تشخیص Alt+Tab با تأیید چند فریمی ---
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        distance = euclidean_distance(thumb_tip, index_tip)
        others_open = other_fingers_open(hand_landmarks)

        if distance < DISTANCE_THRESHOLD and others_open:
            alt_mode_counter += 1
            exit_counter = 0
            if alt_mode_counter >= CONFIRM_FRAMES and not alt_held:
                print("🡐 Alt فشرده شد (منوی سوئیچ باز شد)")
                pyautogui.keyDown('alt')
                pyautogui.press('tab')
                alt_held = True
                prev_hand_x = current_hand_x
        elif alt_held:
            exit_counter += 1
            alt_mode_counter = 0
            if exit_counter >= CONFIRM_FRAMES:
                print("✓ Alt رها شد (برنامه انتخاب شد)")
                pyautogui.keyUp('alt')
                alt_held = False
                prev_hand_x = None
                exit_counter = 0
        else:
            alt_mode_counter = 0
            exit_counter = 0

        # --- کنترل نور و صدا (فقط خارج از حالت Alt+Tab) ---
        if not alt_held:
            cx = hand_landmarks.landmark[0].x
            label = "Right" if cx < 0.5 else "Left"
            cx_px = cx * w

            if label == "Right" and is_fist(hand_landmarks):
                if prev_right_x is not None:
                    delta = (cx_px - prev_right_x) / w
                    brightness = np.clip(brightness + delta * 500, 0, 100)
                prev_right_x = cx_px

            elif label == "Left" and is_fist(hand_landmarks):
                if prev_left_x is not None:
                    delta = (cx_px - prev_left_x) / w
                    new_vol = vol_percent + delta * 250
                    vol_percent = np.clip(new_vol, 0, 100)
                    volume.SetMasterVolumeLevelScalar(vol_percent / 100.0, None)
                prev_left_x = cx_px

            else:
                if label == "Right":
                    prev_right_x = cx_px
                else:
                    prev_left_x = cx_px

        # --- حرکت بین برنامه‌ها در حالت Alt+Tab ---
        else:
            if prev_hand_x is not None:
                diff = current_hand_x - prev_hand_x
                current_time = time.time()
                if (current_time - last_tab_time) > TAB_COOLDOWN:
                    if diff > MOVE_THRESHOLD_NORM:
                        print("→ برنامه بعدی")
                        pyautogui.press('left')
                        prev_hand_x = current_hand_x
                        last_tab_time = current_time
                    elif diff < -MOVE_THRESHOLD_NORM:
                        print("← برنامه قبلی")
                        pyautogui.press('right')
                        prev_hand_x = current_hand_x
                        last_tab_time = current_time
            else:
                prev_hand_x = current_hand_x

    else:
        # --- دست نیست → ریست همه چیز ---
        if alt_held:
            pyautogui.keyUp('alt')
            alt_held = False
        prev_right_x = None
        prev_left_x = None
        prev_hand_x = None
        alt_mode_counter = 0
        exit_counter = 0

    # --- شبیه‌سازی نور (لایه سیاه) ---
    overlay = image.copy()
    alpha = 1.0 - brightness / 100.0
    alpha = np.clip(alpha, 0, 1)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # --- نمایش وضعیت ---
    cv2.putText(image, f"brightness: {int(brightness)}%", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"volume: {int(vol_percent)}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    status = "🡐 Alt+Tab" if alt_held else "🟢 عادی"
    cv2.putText(image, status, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Hand Control - Final Version", image)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC
        break

# --- پاک‌سازی نهایی ---
if alt_held:
    pyautogui.keyUp('alt')
cap.release()
cv2.destroyAllWindows()
print("⏹️ سیستم بسته شد.")
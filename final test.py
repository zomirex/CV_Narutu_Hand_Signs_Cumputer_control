import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import screen_brightness_control as sbc  # ✅ کتابخانه جدید

# --- صدا (فقط ویندوز) ---
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

# --- MediaPipe ---
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

# --- دریافت نور فعلی مانیتور ---
try:
    current_brightness = sbc.get_brightness()[0]  # اولین مانیتور
except Exception as e:
    print("❌ خطا در خواندن نور مانیتور:", e)
    print("💡 مطمئن شوی DDC/CI در مانیتور فعال باشه!")
    current_brightness = 50

# --- تنظیمات حرکت ---
MOVE_THRESHOLD_NORM = 0.01
BRIGHTNESS_STEP = 10  # هر حرکت = 10%
VOLUME_STEP = 5
TAB_COOLDOWN = 0.25
# --- متغیرهای جهانی ---
prev_right_x = None
prev_left_x = None
alt_held = False
prev_hand_x = None
last_tab_time = 0
alt_mode_counter = 0
exit_counter = 0
CONFIRM_FRAMES = 2
DISTANCE_THRESHOLD = 0.06

# --- توابع کمکی ---
def euclidean_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def other_fingers_open(landmarks):
    def is_finger_open(tip, dip):
        return landmarks.landmark[tip].y < landmarks.landmark[dip].y
    return is_finger_open(12, 10) and is_finger_open(16, 14) and is_finger_open(20, 18)

def is_fist(hand_landmarks):
    tips = [8, 12, 16, 20]
    dips = [6, 10, 14, 18]
    for tip, dip in zip(tips, dips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y:
            return False
    return True

# --- شروع ---
cap = cv2.VideoCapture(0)
print("✅ سیستم کنترل دست فعال شد!")
print("- دست راست مشت → نور مانیتور")
print("- دست چپ مشت → صدا")
print("- شست+اشاره چسبیده → Alt+Tab")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    current_hand_x = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        current_hand_x = hand_landmarks.landmark[0].x
        cx_norm = current_hand_x
        label = "Right" if cx_norm < 0.5 else "Left"

        # --- Alt+Tab ---
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        distance = euclidean_distance(thumb_tip, index_tip)
        others_open = other_fingers_open(hand_landmarks)

        if distance*1.1 < DISTANCE_THRESHOLD and others_open:
            alt_mode_counter += 1
            exit_counter = 0
            if alt_mode_counter >= CONFIRM_FRAMES and not alt_held:
                print("🡐 Alt فشرده شد")
                pyautogui.keyDown('alt')
                alt_held = True
                prev_hand_x = current_hand_x
        elif alt_held:
            exit_counter += 1
            alt_mode_counter = 0
            if exit_counter >= CONFIRM_FRAMES:
                print("✓ Alt رها شد")
                pyautogui.keyUp('alt')
                alt_held = False
                prev_hand_x = None
                exit_counter = 0
        else:
            alt_mode_counter = 0
            exit_counter = 0

        # --- کنترل نور و صدا ---
        if not alt_held:
            if label == "Right" and is_fist(hand_landmarks):
                if prev_right_x is not None:
                    diff = cx_norm - prev_right_x
                    if diff < -MOVE_THRESHOLD_NORM:
                        current_brightness = min(100, current_brightness + BRIGHTNESS_STEP)
                        sbc.set_brightness(current_brightness)
                        prev_right_x = cx_norm
                        print(f"💡 نور مانیتور: {current_brightness}%")
                    elif diff > MOVE_THRESHOLD_NORM:
                        current_brightness = max(0, current_brightness - BRIGHTNESS_STEP)
                        sbc.set_brightness(current_brightness)
                        prev_right_x = cx_norm
                        print(f"💡 نور مانیتور: {current_brightness}%")
                else:
                    prev_right_x = cx_norm

            elif label == "Left" and is_fist(hand_landmarks):
                if prev_left_x is not None:
                    diff = cx_norm - prev_left_x
                    if diff > MOVE_THRESHOLD_NORM:
                        vol_percent = min(100, vol_percent + VOLUME_STEP)
                        volume.SetMasterVolumeLevelScalar(vol_percent / 100.0, None)
                        prev_left_x = cx_norm
                        print(f"🔊 صدا: {vol_percent}%")
                    elif diff < -MOVE_THRESHOLD_NORM:
                        vol_percent = max(0, vol_percent - VOLUME_STEP)
                        volume.SetMasterVolumeLevelScalar(vol_percent / 100.0, None)
                        prev_left_x = cx_norm
                        print(f"🔊 صدا: {vol_percent}%")

                else:
                    prev_left_x = cx_norm

        # --- Alt+Tab حرکت ---
        else:
            if prev_hand_x is not None:
                diff = current_hand_x - prev_hand_x
                current_time = time.time()
                if (current_time - last_tab_time) > TAB_COOLDOWN:
                    if diff > MOVE_THRESHOLD_NORM*2:
                        pyautogui.press('tab')
                        prev_hand_x = current_hand_x
                        last_tab_time = current_time
                        print("→ برنامه بعدی")
                    elif diff < -MOVE_THRESHOLD_NORM*2:
                        pyautogui.hotkey('shift', 'tab')
                        prev_hand_x = current_hand_x
                        last_tab_time = current_time
                        print("← برنامه قبلی")
            else:
                prev_hand_x = current_hand_x

    else:
        if alt_held:
            pyautogui.keyUp('alt')
            alt_held = False
        prev_right_x = None
        prev_left_x = None
        prev_hand_x = None
        alt_mode_counter = 0
        exit_counter = 0

    # --- نمایش وضعیت در تصویر ---
    try:
        current_disp = sbc.get_brightness()[0]
    except:
        current_disp = current_brightness
    cv2.putText(image, f"brightness: {current_disp}%", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"volume: {vol_percent}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    status = "Alt+Tab on" if alt_held else "off"
    cv2.putText(image, status, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Hand Control - final version", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# --- پاک‌سازی ---
if alt_held:
    pyautogui.keyUp('alt')
cap.release()
cv2.destroyAllWindows()
print("⏹️ سیستم بسته شد.")
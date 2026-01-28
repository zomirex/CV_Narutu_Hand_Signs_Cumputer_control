import cv2
import mediapipe as mp
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import math
import time

# ======== تنظیمات اولیه ولوم ========
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

# ======== متغیرهای کنترلی ========
control_mode = False  # آیا در حالت کنترل هستیم؟
fist_timestamp = None  # زمان آخرین مشت (برای تشخیص سرعت)
prev_right_fingers = [0] * 5  # وضعیت قبلی انگشتان دست راست
prev_left_fingers = [0] * 5   # وضعیت قبلی انگشتان دست چپ
cooldown = 0.3  # حداقل فاصله زمانی بین افزایش‌های متوالی (ثانیه)
last_volume_change = 0
last_brightness_change = 0
now = time.time()

# ======== mediapipe setup ========
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# توابع کمکی برای تشخیص باز/بسته بودن انگشت
def finger_is_open(hand_landmarks, tip_id, pip_id):
    # اگر نوک انگشت بالاتر (کوچک‌تر در y) از مفصل وسط باشد → باز است (برای دست عمودی در تصویر)
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

def get_finger_states(hand_landmarks):
    # ترتیب: شست (4)، اشاره (8)، وسط (12)، انگشت حلقه (16)، کوچک (20)
    # مفاصل PIP: 3, 6, 10, 14, 18
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    states = []
    for tip, pip in zip(tips, pips):
        if tip == 4:  # شست — منطق متفاوت
            # اگر شست از کف دست خارج شده باشد (x شدت بیشتری داشته باشد)
            states.append(hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x - 0.02)
        else:
            states.append(finger_is_open(hand_landmarks, tip, pip))
    return states

def is_fist(finger_states):
    # مشت = همه انگشتان بسته
    return all(not s for s in finger_states)

cap = cv2.VideoCapture(0)
print("در حال اجرا... دست را در مقابل دوربین نگه دارید.")
print("راهنما:")
print("- دست چپ: باز کردن هر انگشت → +20% نور")
print("- دست راست: باز کردن هر انگشت → +20% ولوم")
print("- مشت کردن سریع (هر دو دست یا یک دست دوباره) → خروج از حالت کنترل")
print("- بستن سریع همه انگشتان (مشت) → ورود به حالت کنترل")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    left_fingers = [0] * 5
    right_fingers = [0] * 5
    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # تشخیص دست چپ/راست
            handedness = results.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = get_finger_states(hand_landmarks)
            if handedness == 'Left':
                left_hand = hand_landmarks
                left_fingers = fingers
            elif handedness == 'Right':
                right_hand = hand_landmarks
                right_fingers = fingers

    # ----- منطق ورود/خروج از حالت کنترل -----
    now = time.time()
    fist_left = is_fist(left_fingers)
    fist_right = is_fist(right_fingers)

    # اگر هر دو دست مشت باشند یا یک دست مشت شد و زمان کوتاهی از آخرین مشت گذشته:
    if fist_left or fist_right:
        if fist_timestamp is None:
            fist_timestamp = now
        else:
            # اگر دو مشت سریع پشت سر هم (مثلاً زیر 0.6 ثانیه):
            if now - fist_timestamp < 0.6:
                control_mode = not control_mode
                print(f"✅ حالت کنترل {'فعال' if control_mode else 'غیرفعال'} شد.")
                fist_timestamp = None  # reset
            else:
                fist_timestamp = now
    else:
        fist_timestamp = None  # اگر مشت رها شد، زمان صفر شود

    # ----- اگر در حالت کنترل بودیم: افزایش ولوم/نور -----
    if control_mode:
        # تغییر دست راست → ولوم
        for i in range(5):
            if right_fingers[i] and not prev_right_fingers[i]:  # انگشت i تازه باز شده
                if now - last_volume_change > cooldown:
                    # ولوم فعلی را بگیر و 20% افزایش بده (در محدوده 0-100)
                    current_vol = volume.GetMasterVolLevelScalar() * 100  # 0.0 ~ 1.0 → 0~100
                    new_vol = min(100, current_vol + 20)
                    volume.SetMasterVolumeLevelScalar(new_vol / 100, None)
                    print(f"🔊 ولوم: {int(new_vol)}%")
                    last_volume_change = now

        # تغییر دست چپ → نور
        for i in range(5):
            if left_fingers[i] and not prev_left_fingers[i]:  # انگشت i تازه باز شده
                if now - last_brightness_change > cooldown:
                    try:
                        current_br = sbc.get_brightness(display=0)[0]
                        new_br = min(100, current_br + 20)
                        sbc.set_brightness(new_br)
                        print(f"💡 نور: {int(new_br)}%")
                        last_brightness_change = now
                    except Exception as e:
                        print("خطا در تنظیم نور:", e)

    # به‌روزرسانی وضعیت قبلی
    prev_right_fingers = right_fingers[:]
    prev_left_fingers = left_fingers[:]

    # نمایش وضعیت روی تصویر
    status = "CONTROL ON" if control_mode else "CONTROL OFF"
    color = (0, 255, 0) if control_mode else (0, 0, 255)
    cv2.putText(image, f"Mode: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Hand Control - Volume & Brightness', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import pyautogui
import time

# --- تنظیمات ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- تنظیمات اسکرول ---
SCROLL_SENSITIVITY = 30       # مقدار اسکرول در هر بار
SCROLL_DELAY = 0.05          # تأخیر بین اسکرول‌ها
last_scroll_time = 0

# --- تشخیص: فقط اشاره و وسطی باز باشند ---
def is_scroll_gesture(hand_landmarks):
    def is_finger_open(tip, dip):
        # y کمتر = بالاتر = باز
        return hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y

    index_open = is_finger_open(8, 6)
    middle_open = is_finger_open(12, 10)
    ring_closed = not is_finger_open(16, 14)
    pinky_closed = not is_finger_open(20, 18)

    return index_open and middle_open and ring_closed and pinky_closed

# --- تشخیص جهت اسکرول: فقط بر اساس تفاوت y نوک و مفصل ---
def get_scroll_direction(hand_landmarks):
    # تفاوت y بین نوک و مفصل (مثبت = نوک پایین‌تر = خم به پایین)
    index_diff = hand_landmarks.landmark[8].y - hand_landmarks.landmark[6].y
    middle_diff = hand_landmarks.landmark[12].y - hand_landmarks.landmark[10].y
    avg_diff = (index_diff + middle_diff) / 2
    print(avg_diff)
    # آستانه کوچک‌تر برای حساسیت بیشتر
    if avg_diff > 0.03:   # انگشت‌ها به پایین خم شدند
        return "down"
    elif avg_diff < -0.03:  # انگشت‌ها صاف یا به بالا
        return "up"
    else:
        return None

# --- شروع ---
cap = cv2.VideoCapture(0)
print("✅ اسکرول با انگشت فعال شد!")
print("- فقط اشاره و وسطی باز → انگشت‌ها را خم کنید")
print("- بالا خم کنید → اسکرول بالا | پایین خم کنید → اسکرول پایین")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    scroll_status = "ready"
    current_time = time.time()

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if is_scroll_gesture(hand_landmarks):
            direction = get_scroll_direction(hand_landmarks)
            if direction and (current_time - last_scroll_time) > SCROLL_DELAY:
                if direction == "up":
                    pyautogui.scroll(SCROLL_SENSITIVITY)
                    scroll_status = "upا"
                else:  # "down"
                    pyautogui.scroll(-SCROLL_SENSITIVITY)
                    scroll_status = "down"
                last_scroll_time = current_time
        else:
            scroll_status = "off"
    else:
        scroll_status = "nothing"

    # --- نمایش ---
    cv2.putText(image, f"state: {scroll_status}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("اسکرول با انگشت - بهینه‌شده", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("⏹️ بسته شد.")
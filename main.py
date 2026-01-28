import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
import time


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max = volume.GetVolumeRange()[:2]


prev_right_x = None
prev_left_x = None
brightness = 100  # درصد روشنایی فقط برای نمایش)
vol_percent = int(volume.GetMasterVolumeLevelScalar() * 100)
smoothing = 0.2  # برای جلوگیری از تغییرات ناگهانی


def is_fist(hand_landmarks):

    tips = [8, 12, 16, 20]
    dips = [6, 10, 14, 18]
    for tip, dip in zip(tips, dips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y:
            return False
    return True


def get_hand_label(hand_results, handedness):
    if len(hand_results.multi_handedness) == 1:
        return hand_results.multi_handedness[0].classification[0].label
    elif len(hand_results.multi_handedness) == 2:

        x1 = hand_results.multi_hand_landmarks[0].landmark[0].x
        x2 = hand_results.multi_hand_landmarks[1].landmark[0].x
        if x1 < x2:
            return "Right", "Left"
        else:
            return "Left", "Right"
    return None, None


while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    h, w, _ = image.shape

    if results.multi_hand_landmarks:
        hand_labels = []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cx = hand_landmarks.landmark[0].x
            label = "Right" if cx < 0.5 else "Left"
            hand_labels.append((label, hand_landmarks))

        for label, landmarks in hand_labels:
            cx = landmarks.landmark[0].x * w
            fist = is_fist(landmarks)

            if label == "Right" and fist:
                if prev_right_x is not None:
                    delta = (cx - prev_right_x) / w
                    brightness = np.clip(brightness + delta * 300, 0, 100)
                prev_right_x = cx

            elif label == "Left" and fist:
                if prev_left_x is not None:
                    delta = (cx - prev_left_x) / w
                    new_vol = vol_percent - delta * 200
                    vol_percent = np.clip(new_vol, 0, 100)
                    vol_scalar = vol_percent / 100.0

                    volume.SetMasterVolumeLevelScalar(vol_scalar, None)

                prev_left_x = cx
            else:

                if label == "Right":
                    prev_right_x = cx
                else:
                    prev_left_x = cx


    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    alpha = 1.0 - brightness / 100.0
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


    cv2.putText(image, f"Brightness: {int(brightness)}%", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"Volume: {int(vol_percent)}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Hand Control - Volume & Brightness", image)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
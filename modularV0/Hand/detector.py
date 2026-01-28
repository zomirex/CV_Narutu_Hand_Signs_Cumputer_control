# hand/detector.py
import cv2
import mediapipe as mp

class HandDetector:
    """
    تشخیص دست، استخراج نقاط کلیدی (landmarks) و تشخیص طرف (left/right).
    """

    def __init__(self, max_hands=2,
                 detection_conf=0.5,
                 tracking_conf=0.5):
        self.max_hands = max_hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame):
        """
        ورودی: تصویر BGR (OpenCV)
        خروجی: (فریم) + لیست دست‌ها و نقاط کلیدی
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        hand_data = []  # هر ورودی: {'landmarks': np.array, 'handedness': 'Left'|'Right'}

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                lm_list = []
                for lm in hand_landmarks.landmark:
                    lm_list.append([lm.x, lm.y, lm.z])
                hand_data.append({
                    'landmarks': lm_list,
                    'handedness': handedness.classification[0].label,  # "Left" یا "Right"
                })
                # رسم روی فریم
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2)
                )

        return frame, hand_data

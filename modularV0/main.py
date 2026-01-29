

import cv2
import pyautogui

from modularV0.Hand.detector import HandDetector
from modularV0.Hand.processor import HandProcessor
from modularV0.Hand.model import GestureModel
from modularV0.Hand.excuter import GestureExecutor
from modularV0.Config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE, GESTURE_COMMANDS

def main():
    #main attributes
    P_P_T_distances = {}
    hand_angles = {}
    finger_status = {}


    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    detector = HandDetector(
    max_hands=2,
    detection_conf=DETECTION_CONFIDENCE,
    tracking_conf=TRACKING_CONFIDENCE
    )
    processor = HandProcessor()
    model = GestureModel(threshold=30)
    executor = GestureExecutor()

    last_gesture = {"Left": None, "Right": None}

    while True:
        ret, frame = cap.read()
        if not ret:
        # logger.error("Could not read frame from camera")
            break

        frame, hands = detector.process(frame)

        # پردازش هر دست
        for hand in hands:
            lm = hand["landmarks"]
            handedness = hand["handedness"]  # 'Left' یا 'Right'
            # print(handedness)
            # print(lm[0])
            ang=processor.Angele_Calculator(lm,frame)
            processor.Distance_norm_Calculator(lm,frame)
            # print(x)
            stat=processor.Finger_Status(ang)
            # print(y)

            gesture = model.classify(stat,ang)
            # print(gesture)
            c=processor.Wrist_angel(lm,frame)
            print(c)

             # جلوگیری از فراخوانی مکرر همان دستور
            if gesture != last_gesture[handedness]:
                executor.execute(gesture)
                last_gesture[handedness] = gesture

            # نمایش درجه ها برای هر دست ارنج رو باید اضاف کنم
            if handedness =="Left":
                y0, dy = 30, 20
            else:
                y0, dy = 150, 20
            for i, (finger, percent) in enumerate(ang.items()):
                text = f"{finger.capitalize()}: {percent}"
                y = y0 + i * dy
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
                if handedness == "Left":
                    cv2.putText(frame, f"{handedness} Hand: {gesture}",
                                (400, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 2)
                else:
                    cv2.putText(frame, f"{handedness} Hand: {gesture}",
                                (400, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 2)


        cv2.imshow("Hand Motion Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
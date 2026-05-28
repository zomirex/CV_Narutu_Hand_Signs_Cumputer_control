import cv2
import mediapipe as mp

# ۱. ویدیو
cap = cv2.VideoCapture(0)

# ۲. راه‌اندازی Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,   # برای عینک یا چشم‌های جزئی
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# اندیس‌های landmark برای چشم‌ها (0‑based)
LEFT_EYE_INDICES  = [474,475,476,477,478]
RIGHT_EYE_INDICES = [263, 362, 387, 386, 385, 384, 398, 381, 380, 379, 374, 373]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تبدیل BGR به RGB (مطلوب برای Mediapipe)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # پردازش
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            # رسم نقاط چشم
            for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                lm = face_landmarks.landmark[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

    cv2.imshow("Mediapipe Eye Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------------------------
#   نصب پیش‌نیازها
# ---------------------------------------------
# pip install mediapipe opencv-python numpy

import cv2
import mediapipe as mp
import numpy as np
import math

# ---------------------------------------------
#   تنظیمات MediaPipe
# ---------------------------------------------
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 5 انگشت (index: 0-4)
FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

# indexهای نقاط مهم در MediaPipe
MCP_IDX = [2, 5, 9, 13, 17]   # MCP برای هر انگشت
TIP_IDX = [4, 8, 12, 16, 20]  # Tip برای هر انگشت
BASE_IDX = [0]               # wrist (landmark 0)

# آستانه‌ی زاویه برای هر انگشت (به درجه)
FOLD_THRESHOLD = {
    'Thumb' : 100,   # مثال: انگشت کوچک‌تر را ممکن است کمتر ببینید
    'Index' : 40,
    'Middle': 40,
    'Ring'  : 40,
    'Pinky' : 40
}
# ---------------------------------------------
#   توابع کمکی
# ---------------------------------------------
def dist(p1, p2):
    """فاصله‌ی اقیانوسی در فضای تصویر (pixel)."""
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def angle_from_sides(a, b, c):
    """محاسبه زاویه‌ی بالای مثلث با استفاده از قضیه‌ی کسینوس."""
    # جلوگیری از خطای تقسیم بر صفر
    if a == 0 or b == 0:
        return 0
    cos_val = (a*a + b*b - c*c) / (2 * a * b)
    # محدود کردن مقدار داخل [-1,1] برای استحکام
    cos_val = max(min(cos_val, 1.0), -1.0)
    return math.degrees(math.acos(cos_val))


def point_line_distance(p, a, b):
    """فاصله‌ی نقطه‌ی p از خط AB (خط بی‌نهایت)."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    p = np.array(p, dtype=float)

    # بردارهای خط و نقطه
    ab = b - a
    ap = p - a

    # مقدار انتزاعی حاصل‌ضرب کراس
    cross_val = np.abs(np.cross(ab, ap))          # در 2D  np.cross برمی‌گرداند اسکالر
    denom      = np.linalg.norm(ab)

    if denom == 0:  # اگر a==b
        return np.linalg.norm(ap)
    return cross_val / denom

# ---------------------------------------------
#   حلقه‌ی اصلی پردازش ویدیوی وب‌کم
# ---------------------------------------------
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8) as hands, \
     mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- تصویر را به رنگ BGR -> RGB تبدیل می‌کنیم
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # --- تشخیص دست
        hand_results = hands.process(image_rgb)

        # --- تشخیص بازو
        pose_results = pose.process(image_rgb)

        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        P_P_T_distances={}
        P_P_T_distances_Raw = {}
        P_L_P_distances = {}
        hand_angles = {}
        finger_status = {}



        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            print(len(hand_landmarks.landmark))
            # print(hand_results.multi_hand_landmarks[0])
            h, w, _ = frame.shape

            # تبدیل نقاط به مختصات pixel
            pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark])

            # --- محاسبه زاویه‌ی هر انگشت
            for i, name in enumerate(FINGER_NAMES):



                p_dip = pts[MCP_IDX[i]]
                p_wrist = pts[0]
                cv2.line(image_bgr, tuple(p_dip), tuple(p_wrist), (255, 0, 0), 2)

                # میانه خط اصلی و برد
                mid = ((p_dip[0] + p_wrist[0]) // 2, (p_dip[1] + p_wrist[1]) // 2)
                vec = np.array([p_wrist[0] - p_dip[0], p_wrist[1] - p_dip[1]], dtype=float)
                norm = np.linalg.norm(vec)

                if norm == 0:
                    perp = np.array([0, 0])
                else:
                    unit_vec = vec / norm
                    perp = np.array([-unit_vec[1], unit_vec[0]])

                # طول خط عمود (نصف طول خط اصلی)
                length = int(norm / 2)
                if length == 0:
                    length = 50

                pt1 = (int(mid[0] + perp[0] * length), int(mid[1] + perp[1] * length))
                pt2 = (int(mid[0] - perp[0] * length), int(mid[1] - perp[1] * length))


                # اطمینان از داخل حاشیه‌ی تصویر
                def inside(pt):
                    x, y = pt
                    return max(0, min(w - 1, x)), max(0, min(h - 1, y))


                pt1 = inside(pt1)
                pt2 = inside(pt2)

                # نمایش خط عمود
                cv2.line(image_bgr, pt1, pt2, (0, 255, 0), 2)

                # --- فاصله‌ی نوک انگشت از این خط عمود ----------------
                p_tip = pts[12]  # نوک انگشت وسط
                dist_tip_to_perp = point_line_distance(p_tip, pt1, pt2)
                cv2.circle(image_bgr, tuple(p_tip), 4, (0, 0, 255), -1)
                cv2.putText(image_bgr, f"{dist_tip_to_perp:.1f} px",
                            (p_tip[0] + 10, p_tip[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                # 1. محاسبه‌ی سه ضلع
                a = point_line_distance(pts[MCP_IDX[i]], pt1, pt2)
                b = dist(pts[MCP_IDX[i]], pts[TIP_IDX[i]])
                c = point_line_distance(pts[TIP_IDX[i]], pt1, pt2)

                # 2. زاویه‌ی انگشت
                theta = angle_from_sides(a, b, c)
                hand_angles[name] = theta

                # 3. وضعیت انگشت – استفاده از آستانه‌ی مخصوص
                threshold = FOLD_THRESHOLD.get(name, 40)  # اگر نام وجود نداشت، به ۳۰ برگرد
                finger_status[name] = 'Folded' if theta <= threshold else 'Extended'
            # --- محاسبه فاصله هر انگشت
            for i, name_i in enumerate(FINGER_NAMES):
                for j, name_j in enumerate(FINGER_NAMES):
                    # فقط یکبار (یا i < j) محاسبه کنید تا تکرار (symmetry) نداشته باشید
                    if i < j:
                        pt_i = pts[TIP_IDX[i]]
                        pt_j = pts[TIP_IDX[j]]
                        P_P_T_distances_Raw[(name_i, name_j)] = dist(pt_i,pt_j)  # dist به‌صورت (x1,y1),(x2,y2) تعریف شده‌اید

            D_ref = dist(pts[0], pts[9])  # مچ → نوک وسط
            if D_ref < 1e-6:  # جلوگیری از تقسیم بر صفر
                D_ref = 1e-6
            P_P_T_distances = {k: v / D_ref for k, v in P_P_T_distances_Raw.items()}
            # print(P_P_T_distances)

            # نمایش مختصات و زاویه‌ها روی تصویر
            for name in FINGER_NAMES:
                angle_str = f'{hand_angles[name]:.0f}°'
                cv2.putText(image_bgr, f'{name}:{angle_str}', (10, 30 + 30*FINGER_NAMES.index(name)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if finger_status[name]=='Extended' else (0, 0, 255), 2)

            # تشخیص وضعیت کل دست
            if all(status == 'Folded' for status in finger_status.values()):
                hand_state = 'Closed Fist'
                color = (0, 0, 255)  # قرمز
            elif all(status == 'Extended' for status in finger_status.values()):
                hand_state = 'Open Hand'
                color = (0, 255, 0)  # سبز
            else:
                hand_state = 'Mixed'
                color = (255, 255, 0)  # آبی


            cv2.putText(image_bgr, f'Hand: {hand_state}', (10, 30 + 30*len(FINGER_NAMES)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

            # رسم هندسه‌ی دست
            mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # --------------------------------------------------------------------

        # عمود منصف ورژن 0
        # --------------------------------------------------------------------
        # if hand_results.multi_hand_landmarks:
        #     hand_landmarks = hand_results.multi_hand_landmarks[0]
        #     h, w, _ = frame.shape
        #     # pts: (21,2) – مختصات pixel برای تمام landmarks
        #     pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark], dtype=int)
        #
        #     # ----------------------------------------------------------------
        #     # 1️⃣  خط از مفصل پایین انگشت وسط (landmark 11) به نقطه‌ی کف دست (landmark 0)
        #     # ----------------------------------------------------------------
        #     p_dip = pts[9]  # مفصل پایین انگشت وسط (DIP)
        #     p_wrist = pts[0]  # نقطه‌ی کف دست (Wrist)
        #
        #     # رسم خط اصلی (مقدار رنگ و ضخامت را می‌توانید تغییر دهید)
        #     cv2.line(image_bgr, tuple(p_dip), tuple(p_wrist), (255, 0, 0), 2)
        #
        #     # ----------------------------------------------------------------
        #     # 2️⃣  محاسبه‌ی نقطه‌ی میانه و برد خط اصلی
        #     # ----------------------------------------------------------------
        #     mid = ((p_dip[0] + p_wrist[0]) // 2, (p_dip[1] + p_wrist[1]) // 2)
        #     vec = np.array([p_wrist[0] - p_dip[0], p_wrist[1] - p_dip[1]], dtype=float)
        #
        #     # برد واحد خط اصلی
        #     norm = np.linalg.norm(vec)
        #     if norm == 0:
        #         perp = np.array([0, 0])
        #     else:
        #         unit_vec = vec / norm
        #         # برد عمود (90 درجه چرخش، چپ-چرخش)
        #         perp = np.array([-unit_vec[1], unit_vec[0]])
        #
        #     # ----------------------------------------------------------------
        #     # 3️⃣  طول خط عمود – می‌توانید دلخواه خودتان را تعیین کنید
        #     # ----------------------------------------------------------------
        #     # مثال: نصف طول خط اصلی (حداقل 50px اگر صفر شد)
        #     length = int(norm / 2)
        #     if length == 0:
        #         length = 50
        #
        #     # دو نقطه انتهایی خط عمود
        #     pt1 = (int(mid[0] + perp[0] * length), int(mid[1] + perp[1] * length))
        #     pt2 = (int(mid[0] - perp[0] * length), int(mid[1] - perp[1] * length))
        #
        #
        #     # اطمینان از داخل حاشیه‌ی تصویر (اختیاری)
        #     def inside(pt):
        #         x, y = pt
        #         return max(0, min(w - 1, x)), max(0, min(h - 1, y))
        #
        #
        #     pt1 = inside(pt1)
        #     pt2 = inside(pt2)
        #
        #     # رسم خط عمود (سبز رنگ، ضخامت 2 پیکسل)
        #     cv2.line(image_bgr, pt1, pt2, (0, 255, 0), 2)
#-------------------
        # if hand_results.multi_hand_landmarks:#عمود منصف ورژن1
        #     hand_landmarks = hand_results.multi_hand_landmarks[0]
        #     h, w, _ = frame.shape
        #     pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark], dtype=int)
        #
        #     # خط اصلی: مفصل پایین انگشت وسط (9) → مچ (0)
        #     p_dip = pts[9]
        #     p_wrist = pts[0]
        #     cv2.line(image_bgr, tuple(p_dip), tuple(p_wrist), (255, 0, 0), 2)
        #
        #     # میانه خط اصلی و برد
        #     mid = ((p_dip[0] + p_wrist[0]) // 2, (p_dip[1] + p_wrist[1]) // 2)
        #     vec = np.array([p_wrist[0] - p_dip[0], p_wrist[1] - p_dip[1]], dtype=float)
        #     norm = np.linalg.norm(vec)
        #
        #     if norm == 0:
        #         perp = np.array([0, 0])
        #     else:
        #         unit_vec = vec / norm
        #         perp = np.array([-unit_vec[1], unit_vec[0]])
        #
        #     # طول خط عمود (نصف طول خط اصلی)
        #     length = int(norm / 2)
        #     if length == 0:
        #         length = 50
        #
        #     pt1 = (int(mid[0] + perp[0] * length), int(mid[1] + perp[1] * length))
        #     pt2 = (int(mid[0] - perp[0] * length), int(mid[1] - perp[1] * length))
        #
        #
        #     # اطمینان از داخل حاشیه‌ی تصویر
        #     def inside(pt):
        #         x, y = pt
        #         return max(0, min(w - 1, x)), max(0, min(h - 1, y))
        #
        #
        #     pt1 = inside(pt1)
        #     pt2 = inside(pt2)
        #
        #     # نمایش خط عمود
        #     cv2.line(image_bgr, pt1, pt2, (0, 255, 0), 2)
        #
        #     # --- فاصله‌ی نوک انگشت از این خط عمود ----------------
        #     p_tip = pts[12]  # نوک انگشت وسط
        #     dist_tip_to_perp = point_line_distance(p_tip, pt1, pt2)
        #     cv2.circle(image_bgr, tuple(p_tip), 4, (0, 0, 255), -1)
        #     cv2.putText(image_bgr, f"{dist_tip_to_perp:.1f} px",
        #                 (p_tip[0] + 10, p_tip[1] - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ---------------------------------------------
        #   (اختیاری) تشخیص زاویه‌ی آرنج
        # ---------------------------------------------

        if pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks
            h, w, _ = frame.shape
            pose_pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in pose_landmarks.landmark])

            # آرنج چپ (شانه:11, آرنج:13, مچ:15)
            shoulder = pose_pts[11]
            elbow    = pose_pts[13]
            wrist    = pose_pts[15]

            # آرنج راست (شانه:12, آرنج:14, مچ:16)
            shoulder_r = pose_pts[12]
            elbow_r    = pose_pts[14]
            wrist_r    = pose_pts[16]

            def elbow_angle(shoulder, elbow, wrist):
                a = dist(shoulder, elbow)
                b = dist(elbow, wrist)
                c = dist(shoulder, wrist)
                return angle_from_sides(a, b, c)

            angle_left  = elbow_angle(shoulder, elbow, wrist)
            angle_right = elbow_angle(shoulder_r, elbow_r, wrist_r)

            cv2.putText(image_bgr, f'Elbow L: {angle_left:.0f}°', (10, 30 + 30*(len(FINGER_NAMES)+1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(image_bgr, f'Elbow R: {angle_right:.0f}°', (10, 60 + 30*(len(FINGER_NAMES)+1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # نمایش خروجی
        cv2.imshow('Hand & Arm Closure Detection', image_bgr)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC
            break

cap.release()
cv2.destroyAllWindows()
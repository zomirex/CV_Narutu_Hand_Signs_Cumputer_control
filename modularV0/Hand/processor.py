#   نصب پیش‌نیازها
# ---------------------------------------------
# pip install mediapipe opencv-python numpy

import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Tuple
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
Point = Tuple[float, float]

# توابع کمکی
def angle_between_lines(
    p1: Point, p2: Point, p3: Point, p4: Point, in_degrees: bool = True
) -> float:
    """
    Compute the angle between the line (p1,p2) and the line (p3,p4).

    Parameters
    ----------
    p1, p2 : Point
        Two distinct points defining the first line.
    p3, p4 : Point
        Two distinct points defining the second line.
    in_degrees : bool, default True
        If True, return the angle in degrees; otherwise return radians.

    Returns
    -------
    float
        The acute angle (0 ≤ θ ≤ 180) between the two lines.

    Raises
    ------
    ValueError
        If any pair of points defining a line are identical.
    """

    # Helper to build a direction vector
    def vec(a: Point, b: Point) -> Tuple[float, float]:
        return (b[0] - a[0], b[1] - a[1])

    # Direction vectors of the two lines
    v1 = vec(p1, p2)
    v2 = vec(p3, p4)

    # Sanity check – lines must be defined by two distinct points
    if v1 == (0.0, 0.0):
        raise ValueError("Points p1 and p2 cannot be identical.")
    if v2 == (0.0, 0.0):
        raise ValueError("Points p3 and p4 cannot be identical.")

    # Dot product and magnitudes
    dot_prod = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])

    # Guard against floating‑point rounding errors that could push the
    # cosine slightly out of the [-1, 1] interval.
    cos_theta = max(-1.0, min(1.0, dot_prod / (mag1 * mag2)))

    # The angle in radians
    theta_rad = math.acos(cos_theta)

    # Return in requested unit
    return math.degrees(theta_rad) if in_degrees else theta_rad

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

class HandProcessor():
    def __init__(self, fold_thr=FOLD_THRESHOLD, wrist_idx=BASE_IDX):
        self.fold_thr = fold_thr
        self.wrist_idx = wrist_idx
    def Angele_Calculator(self,hand_landmarks,frame):
        hand_angles={}
        h, w, _ = frame.shape
        pts = np.array([[int(lm[0] * w), int(lm[1] * h)] for lm in hand_landmarks])
        for i, name in enumerate(FINGER_NAMES):

            p_dip = pts[MCP_IDX[i]]
            p_wrist = pts[0]


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
            # cv2.line(image_bgr, pt1, pt2, (0, 255, 0), 2)

            # --- فاصله‌ی نوک انگشت از این خط عمود ----------------
            p_tip = pts[12]  # نوک انگشت وسط
            dist_tip_to_perp = point_line_distance(p_tip, pt1, pt2)
            # cv2.circle(image_bgr, tuple(p_tip), 4, (0, 0, 255), -1)
            # cv2.putText(image_bgr, f"{dist_tip_to_perp:.1f} px",
            #             (p_tip[0] + 10, p_tip[1] - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 1. محاسبه‌ی سه ضلع
            a = point_line_distance(pts[MCP_IDX[i]], pt1, pt2)
            b = dist(pts[MCP_IDX[i]], pts[TIP_IDX[i]])
            c = point_line_distance(pts[TIP_IDX[i]], pt1, pt2)

            # 2. زاویه‌ی انگشت
            theta = angle_from_sides(a, b, c)
            hand_angles[name] = round(theta,1)
        # print(hand_angles)
        return hand_angles
    def Distance_norm_Calculator(self,hand_landmarks,frame):
        P_P_T_distances_Raw={}
        h, w, _ = frame.shape
        pts = np.array([[int(lm[0] * w), int(lm[1] * h)] for lm in hand_landmarks])
        #--- محاسبه فاصله هر انگشت
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
        return P_P_T_distances
    def Finger_Status(self,angs):

        finger_status={}
        # for name,theta in angs :
        for i, name in enumerate(FINGER_NAMES):
            threshold = FOLD_THRESHOLD.get(name, 40)  # اگر نام وجود نداشت، به ۳۰ برگرد
            finger_status[name] = 'Folded' if angs[name] <= threshold else 'Extended'
        return finger_status
    def Wrist_angel(self,hand_landmarks,frame):
            hand_angles = {}
            h, w, _ = frame.shape
            pts = np.array([[int(lm[0] * w), int(lm[1] * h)] for lm in hand_landmarks])


            p_l = pts[2]
            p_r = pts[17]

            # 2. زاویه‌ی انگشت
            theta = angle_between_lines(p_l,p_r,[0,0],[0,100])
            Wrist_ang = round(theta, 1)
            return Wrist_ang
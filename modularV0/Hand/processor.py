
"""
کلاس `HandProcessor`  وظیفه‌ی محاسبه هندسه‌ی دست را بر عهده دارد:
- زاویه‌ی هر انگشت (به کمک قانون کسینوس و متدهای کمکی)
- فاصله‌ی نرمالیزه بین نوک انگشتان
- وضعیت باز/بسته‌ی هر انگشت بر مبنای آستانه‌ی `FOLD_THRESHOLD`
- زاویه‌ی آرنج (مقداری اختیاری برای نمایش)

توابع کمکی در بالای فایل تعریف شده‌اند؛
در ادامه با توضیحات فارسی، کد اصلی را توضیح می‌دهیم.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Tuple
from modularV0.utils.Config_Loader import load_config

# -------------------------------------------------------------
#  تنظیمات MediaPipe (به‌صورت سراسری در برنامه)
# -------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# -------------------------------------------------------------
#  ثابت‌های دست (فهرست انگشتان و ایندکس‌های نقاط کلیدی)
# -------------------------------------------------------------
FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

# ایندکس‌های نقاط مهم در MediaPipe
MCP_IDX = [2, 5, 9, 13, 17]   # MCP (Middle‑Carp) برای هر انگشت
TIP_IDX = [4, 8, 12, 16, 20]  # نوک انگشت (Tip)
BASE_IDX = [0]               # wrist (landmark 0)

# آستانه‌ی زاویه برای هر انگشت (به درجه) – از فایل تنظیمات خوانده می‌شود
FOLD_THRESHOLD = load_config("x.json").FOLD_THRESHOLD
Point = Tuple[float, float]

# ------------------------------------------------------------------
#  توابع کمکی (محاسبه زاویه، فاصله و ...) – مستقیماً در این فایل
# ------------------------------------------------------------------
def angle_between_lines(
    p1: Point, p2: Point, p3: Point, p4: Point, in_degrees: bool = True
) -> float:
    """
    زاویه‌ی بین دو خط (p1‑p2) و (p3‑p4) را محاسبه می‌کند.
    اگر `in_degrees` True باشد مقدار به درجه بازگردانده می‌شود، در غیر اینصورت به رادیان.
    """
    def vec(a: Point, b: Point) -> Tuple[float, float]:
        return (b[0] - a[0], b[1] - a[1])

    v1 = vec(p1, p2)
    v2 = vec(p3, p4)

    if v1 == (0.0, 0.0):
        raise ValueError("Points p1 and p2 cannot be identical.")
    if v2 == (0.0, 0.0):
        raise ValueError("Points p3 and p4 cannot be identical.")

    dot_prod = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])

    cos_theta = max(-1.0, min(1.0, dot_prod / (mag1 * mag2)))
    theta_rad = math.acos(cos_theta)

    return math.degrees(theta_rad) if in_degrees else theta_rad


def dist(p1, p2):
    """فاصله‌ی اقیانوسی (Euclidean) بین دو نقطه‌ی تصویر (pixel)."""
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def angle_from_sides(a, b, c):
    """محاسبه زاویه‌ی بالای مثلث با استفاده از قضیه‌ی کسینوس."""
    if a == 0 or b == 0:
        return 0
    cos_val = (a*a + b*b - c*c) / (2 * a * b)
    cos_val = max(min(cos_val, 1.0), -1.0)
    return math.degrees(math.acos(cos_val))


def point_line_distance(p, a, b):
    """فاصله‌ی نقطه‌ی p از خط AB (خط بی‌نهایت)."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    p = np.array(p, dtype=float)

    ab = b - a
    ap = p - a
    cross_val = np.abs(np.cross(ab, ap))
    denom = np.linalg.norm(ab)

    if denom == 0:      # اگر a==b
        return np.linalg.norm(ap)
    return cross_val / denom


# ------------------------------------------------------------------
#  کلاس اصلی که در برنامه‌ی اصلی فراخوانی می‌شود
# ------------------------------------------------------------------
class HandProcessor:
    def __init__(self, fold_thr=FOLD_THRESHOLD, wrist_idx=BASE_IDX):
        """
        پارامترها:
            fold_thr : دیکشنری آستانه‌ی زاویه برای هر انگشت
            wrist_idx: ایندکس نقطه‌ی آرنج (در این کد استفاده نشد، ولی برای سازگاری نگه‌داشته شده)
        """
        self.fold_thr = fold_thr
        self.wrist_idx = wrist_idx

    # ------------------------------------------------------------------
    #  بروز رسانی تنظیمات (برای مثال پس از تغییر در فایل JSON)
    # ------------------------------------------------------------------
    def Config(self, Config):
        self.fold_thr = Config.FOLD_THRESHOLD

    # ------------------------------------------------------------------
    #  محاسبه زاویه‌ی هر انگشت
    # ------------------------------------------------------------------
    def Angele_Calculator(self, hand_landmarks, frame):
        """
        ورودی:
            hand_landmarks : لیست نقاط کلیدی (x, y, z) در فرمت MediaPipe
            frame          : تصویر BGR (از OpenCV)
        خروجی:
            dict: {'Thumb': 123.4, 'Index': 98.2, …}
            (زاویه‌ی هر انگشت به درجه)
        """
        hand_angles = {}
        h, w, _ = frame.shape
        # تبدیل نقاط نسبت به اندازه‌ی تصویر
        pts = np.array([[int(lm[0] * w), int(lm[1] * h)] for lm in hand_landmarks])

        for i, name in enumerate(FINGER_NAMES):
            # ۱. نقطه‌ی MCP (مرکز انگشت) و آرنج
            p_dip = pts[MCP_IDX[i]]
            p_wrist = pts[0]

            # ۲. مرکز خط اصلی و برد
            mid = ((p_dip[0] + p_wrist[0]) // 2,
                   (p_dip[1] + p_wrist[1]) // 2)
            vec = np.array([p_wrist[0] - p_dip[0], p_wrist[1] - p_dip[1]], dtype=float)
            norm = np.linalg.norm(vec)

            # بردار عمود (مقدار واحد)
            if norm == 0:
                perp = np.array([0, 0])
            else:
                unit_vec = vec / norm
                perp = np.array([-unit_vec[1], unit_vec[0]])

            # طول خط عمود (نصف طول خط اصلی)
            length = int(norm / 2) or 50   # حداقل طول

            # نقطه‌های انتهایی خط عمود
            pt1 = (int(mid[0] + perp[0] * length), int(mid[1] + perp[1] * length))
            pt2 = (int(mid[0] - perp[0] * length), int(mid[1] - perp[1] * length))

            # اطمینان از داخل حاشیه‌ی تصویر
            def inside(pt):
                x, y = pt
                return max(0, min(w - 1, x)), max(0, min(h - 1, y))

            pt1 = inside(pt1)
            pt2 = inside(pt2)

            # ------------------------------------------------------------------
            # 3. فاصله‌ی نوک انگشت (برای مقایسه‌ی بعدی)
            # ------------------------------------------------------------------
            p_tip = pts[TIP_IDX[i]]            # نوک انگشت (مهم‌ترین نقطه‌ی انگشت)
            dist_tip_to_perp = point_line_distance(p_tip, pt1, pt2)

            # ------------------------------------------------------------------
            # 4. محاسبه‌ی سه ضلع و زاویه‌ی انگشت
            # ------------------------------------------------------------------
            a = point_line_distance(pts[MCP_IDX[i]], pt1, pt2)  # فاصله‌ی MCP تا خط عمود
            b = dist(pts[MCP_IDX[i]], pts[TIP_IDX[i]])        # طول انگشت
            c = point_line_distance(pts[TIP_IDX[i]], pt1, pt2) # فاصله‌ی نوک تا خط عمود

            theta = angle_from_sides(a, b, c)  # زاویه‌ی انگشت
            hand_angles[name] = round(theta, 1)

        return hand_angles

    # ------------------------------------------------------------------
    #  محاسبه فاصله‌ی نرمالیزه‌ی هر دو نوک انگشت
    # ------------------------------------------------------------------
    def Distance_norm_Calculator(self, hand_landmarks, frame):
        """
        خروجی: دیکشنری فواصل نرمالیزه بین نوک انگشتان
        کلید: (name_i, name_j)
        مقدار: فاصله‌ی نرمالیزه (بین 0 و 1)
        """
        P_P_T_distances_Raw = {}
        h, w, _ = frame.shape
        pts = np.array([[int(lm[0] * w), int(lm[1] * h)] for lm in hand_landmarks])

        # محاسبه فواصل بین نوک انگشتان
        for i, name_i in enumerate(FINGER_NAMES):
            for j, name_j in enumerate(FINGER_NAMES):
                if i < j:  # هر ترکیب فقط یکبار
                    pt_i = pts[TIP_IDX[i]]
                    pt_j = pts[TIP_IDX[j]]
                    P_P_T_distances_Raw[(name_i, name_j)] = dist(pt_i, pt_j)

        # مرجع: فاصله‌ی مچ دست تا نوک انگشت وسط (برای نرمالیزه)
        D_ref = dist(pts[0], pts[9])  # 0=mcp_wrist, 9=MiddleTip
        if D_ref < 1e-6:
            D_ref = 1e-6
        # نرمالیزه کردن
        P_P_T_distances = {k: v / D_ref for k, v in P_P_T_distances_Raw.items()}
        return P_P_T_distances

    # ------------------------------------------------------------------
    #  تعیین وضعیت (باز/بسته) هر انگشت بر اساس زاویه و آستانه
    # ------------------------------------------------------------------
    def Finger_Status(self, angs):
        """
        ورودی:
            angs : dict {name: زاویه‌ی انگشت}
        خروجی:
            dict {name: 'Folded' یا 'Extended'}
        """
        finger_status = {}
        for i, name in enumerate(FINGER_NAMES):
            threshold = self.fold_thr.get(name, 70)   # آستانه‌ی پیش‌فرض 70°
            finger_status[name] = 'Folded' if angs[name] <= threshold else 'Extended'
        return finger_status

    # ------------------------------------------------------------------
    #  محاسبه زاویه‌ی آرنج (به‌صورت اختیاری برای نمایش)
    # ------------------------------------------------------------------
    def Wrist_angel(self, hand_landmarks, frame):
        """
        زاویه‌ی آرنج (بین دو نقطه‌ی MCP-Thumb و MCP-Pinky)
        به‌صورت عددی (درجه) بازگردانده می‌شود.
        """
        h, w, _ = frame.shape
        pts = np.array([[int(lm[0] * w), int(lm[1] * h)] for lm in hand_landmarks])

        p_l = pts[2]   # MCP Thumb
        p_r = pts[17]  # MCP Pinky

        theta = angle_between_lines(p_l, p_r, [0, 0], [0, 100])  # ردیف vertical reference
        Wrist_ang = round(theta, 1)
        return Wrist_ang

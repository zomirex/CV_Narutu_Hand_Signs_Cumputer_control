"""
مدارک اصلی برنامه‌ی تشخیص دست و Gesture
در این فایل تمام ویجت‌های PyQt، پردازش دوربین و
مدیریت کانفیگ‌ها تعریف شده‌اند.
"""

import os
import sys
import json
import cv2
import mediapipe as mp
import numpy as np

# --------------------- ماژول‌های PyQt ---------------------
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
    QHBoxLayout, QGroupBox, QGridLayout, QStyleFactory, QMessageBox,
    QSlider, QToolButton, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QIcon

# --------------------- ماژول‌های شخصی ---------------------
from modularV0.Hand.detector   import HandDetector
from modularV0.Hand.processor  import HandProcessor
from modularV0.Hand.model      import GestureModel
from modularV0.Hand.excuter    import GestureExecutor
from modularV0.utils.Configurations import (
    CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT,
    DETECTION_CONFIDENCE, TRACKING_CONFIDENCE
)
import modularV0.utils.Config_Loader as x

# ---------------------------------------------------------
# مسیر فایل تنظیمات (به‌جای متغیرهای ثابت می‌توانید این را به محیطی دیگر هم تغییر دهید)
CONFIG_FILE = "x.json"

# ----------------------------------------------------------------
# بارگذاری تنظیمات اولیه (پیش‌فرض‌ها در صورت عدم وجود فایل)
# ----------------------------------------------------------------
cfg = x.load_config(CONFIG_FILE)


cfg =x.load_config (CONFIG_FILE)

class CameraWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)

        # باز کردن دوربین
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.setText("❌ دوربین پیدا نشد")
            return

        # پارامترهای اولیه تنظیمات تصویر
        self.brightness = 0
        self.contrast   = 0
        self.gamma      = 0

        # تایمر برای دریافت فریم‌ها
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)   # تقریبا 30fps

        # حافظه‌های برای اطلاعات دست (در این ویجت به‌صورت ساده استفاده نمی‌شوند)
        self.P_P_T_distances = {}
        self.hand_angles = {}
        self.finger_status = {}
        self.hand_gesture = {}
        self.hand_wrist_ang = {}

    # ------------------------------------------------------------------
    # دریافت فریم، اعمال تنظیمات، تبدیل به QImage و نمایش در QLabel
    # ------------------------------------------------------------------
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # تبدیل از BGR به RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # اعمال brightness / contrast / gamma
        frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(256)]).astype("uint8")
        frame = cv2.LUT(frame, table)

        # تبدیل به QImage
        h, w, ch = frame.shape
        bytesPerLine = ch * w
        qImg = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qImg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pix)

    # ------------------------------------------------------------------
    # متدهای تنظیمات دوربین (کنتراست، روشنایی، گاما)
    # ------------------------------------------------------------------
    def set_brightness(self, v):
        self.brightness = v

    def set_contrast(self, v):
        self.contrast = v

    def set_gamma(self, v):
        self.gamma = v

    # ------------------------------------------------------------------
    # اطمینان از آزادسازی منابع دوربین در زمان بسته شدن ویجت
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)

# ============================================================
# ویجت پردازش دست (دوربین + MediaPipe + مدل‌های Gesture)
# ============================================================
class HandProcessorWidget(QLabel):
    """
    این ویجت تمامی مراحل پردازش دست را در خود جای می‌دهد:
    - خواندن فریم از دوربین
    - تشخیص دست توسط MediaPipe
    - محاسبه هندسه انگشتان
    - طبقه‌بندی Gesture
    - اجرای فرمان بر اساس Gesture
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)

        # ------------------------------------------------------------------
        # باز کردن دوربین و تنظیم پارامترهای رزولوشن
        # ------------------------------------------------------------------
        self.cap = cv2.VideoCapture(CAMERA_ID)
        if not self.cap.isOpened():
            self.setText("❌ دوربین پیدا نشد")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # تنظیمات اولیه تصویر
        self.brightness = 0
        self.contrast = 1.0
        self.gamma = 1.0

        # ------------------------------------------------------------------
        # ایجاد اشیای پردازش
        # ------------------------------------------------------------------
        self.detector  = HandDetector(
            max_hands=cfg.max_hands,
            detection_conf=cfg.detection_conf,
            tracking_conf=cfg.tracking_conf
        )
        self.processor = HandProcessor()
        self.model     = GestureModel()
        self.executor  = GestureExecutor()

        # حافظهٔ وضعیت (برای جلوگیری از فراخوانی مکرر دستور)
        self.last_gesture = {"Left": None, "Right": None}

        # تایمر فریم
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)  # تقریباً 20fps

    # ------------------------------------------------------------------
    # دریافت فریم، پردازش دست‌ها، نمایش، و اجرای فرمان‌ها
    # ------------------------------------------------------------------
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.setText("❌ دریافت فریم از دوربین امکان‌پذیر نیست")
            return

        # به‌روزرسانی کانفیگ در هر حلقه (اگر فایل تنظیمات تغییر کرده باشد)
        cfg = x.load_config(CONFIG_FILE)

        # تبدیل فریم به RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # اعمال brightness / contrast / gamma
        frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(256)]).astype("uint8")
        frame = cv2.LUT(frame, table)

        # ------------------------------------------------------------------
        # تشخیص دست‌ها توسط MediaPipe
        # ------------------------------------------------------------------
        frame, hands = self.detector.process(frame)

        # حافظه‌های موقتی برای نمایش (برای هر دست)
        hand_angles = {}
        hand_gesture = {}
        P_P_T_distances = {}
        finger_status = {}
        hand_wrist_ang = {}

        # بروزرسانی تنظیمات (در صورت تغییر در فایل)
        self.processor.Config(x.load_config("x.json"))
        self.detector.Config(x.load_config("x.json"))

        # ------------------------------------------------------------------
        # حلقه‌ی پردازش هر دست
        # ------------------------------------------------------------------
        for hand in hands:
            lm = hand["landmarks"]          # لیست نقاط (x, y, z)
            handedness = hand["handedness"]  # 'Left' یا 'Right'

            # ۱. محاسبه زوایای انگشتان
            ang = self.processor.Angele_Calculator(lm, frame)
            hand_angles[handedness] = ang

            # ۲. محاسبه فاصله‌های (Palm‑Palm، Palm‑Thumb و غیره)
            P_P_T_distances[handedness] = self.processor.Distance_norm_Calculator(lm, frame)

            # ۳. وضعیت (مفت یا بسته) هر انگشت
            finger_status[handedness] = stat = self.processor.Finger_Status(ang)

            # ۴. طبقه‌بندی Gesture
            hand_gesture[handedness] = gesture = self.model.classify(
                stat, ang, P_P_T_distances[handedness]
            )

            # ۵. زاویه‌ی آرنج (فقط برای نمایش)
            hand_wrist_ang[handedness] = self.processor.Wrist_angel(lm, frame)

            # ------------------------------------------------------------------
            # نمایش متنی بر روی فریم (زاویه انگشتان و Gesture)
            # ------------------------------------------------------------------
            y0, dy = (30, 20) if handedness == "Left" else (150, 20)
            for i, (finger, percent) in enumerate(ang.items()):
                text = f"{finger.capitalize()}: {percent}"
                y = y0 + i * dy
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)

            txt = f"{handedness} Hand: {gesture}"
            x_pos = 400
            y_pos = 80 if handedness == "Left" else 100
            cv2.putText(frame, txt, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 2)

        # ------------------------------------------------------------------
        # فرمان‌دهی: وقتی دست چپ مشت است، دست راست را کنترل می‌کند
        # ------------------------------------------------------------------
        if hand_gesture.get('Left') and hand_gesture.get('Right'):
            if hand_gesture['Left'] == 'fist':
                # ارسال فرمان به سمت راست (شما می‌توانید متد را تغییر دهید)
                self.executor.execute_XMove_alt(hand_gesture.get('Right'), lm)
                self.last_gesture['Right'] = hand_gesture.get('Right')
                print(lm[0])  # نقطه‌ی اول (به‌عنوان نمونه)

        # ------------------------------------------------------------------
        # تبدیل فریم به QImage و نمایش در QLabel
        # ------------------------------------------------------------------
        h, w, ch = frame.shape
        bytesPerLine = ch * w
        qImg = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qImg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pix)

    # ------------------------------------------------------------------
    # متدهای تنظیمات (brightness، contrast، gamma)
    # ------------------------------------------------------------------
    def set_brightness(self, v):
        self.brightness = v

    def set_contrast(self, v):
        self.contrast = v

    def set_gamma(self, v):
        self.gamma = v

    # ------------------------------------------------------------------
    # آزادسازی منابع در هنگام بسته شدن
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)

# ------------------------------------------------------------------
# ۳. MainWindow
# ------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Computer Controller")
        self.resize(400, 600)
        self.setAutoFillBackground(True)
        # self.setWindowFlags(Qt.tit | Qt.Window)
        # self.setAttribute(Qt.WA_TranslucentBackground)

        # ----------- سبک Fusion -----------------
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        # (رنگ‌آمیزی مشابه نمونه‌ی بالا، اختیاری)

        # ----------- ویجت‌های اصلی -----------------
        # self.cam_widget = CameraWidget(self)
        self.hand_widget = HandProcessorWidget(self)
        # self.settings_panel = SettingsPanel(self.cam_widget, self)
        self.settings_panel = SettingsPanel(self.hand_widget, self)


        # ---- دکمه تعویض تم ----
        self.theme_btn = QToolButton()
        self.theme_btn.setText("تغییر تم")
        self.theme_btn.setToolTip("تغییر رنگ (روشن یا دارک)")
        self.theme_btn.setIcon(QIcon.fromTheme("weather-clear-night"))
        self.theme_btn.setCheckable(True)
        self.theme_btn.setChecked(cfg.theme == "dark")
        self.theme_btn.toggled.connect(self.toggle_theme)
        # تنظیمات اولیه
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=cfg.max_hands,
            min_detection_confidence=cfg.detection_conf,
            min_tracking_confidence=cfg.tracking_conf,
        )

        # مدل گِیِست
        self.gesture_model = GestureModel(threshold=30)

        # دکمه تنظیمات
        self.settings_btn = QPushButton("تنظیمات")
        self.settings_btn.clicked.connect(self.open_settings)

        # لایوت
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        top_bar.addWidget(self.theme_btn)

        # self.hand_widget.setSizePolicy(
        #     QSizePolicy.Preferred,  # horizontal:  Preferred (یا Fixed)
        #     QSizePolicy.Fixed  # vertical:  Fixed
        # )
        # self.hand_widget.setFixedSize(640,480)


        layout = QVBoxLayout()
        # layout.addWidget(self.cam_widget, stretch=3)
        layout.addWidget(self.hand_widget)

        layout.addLayout(top_bar)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.settings_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        layout.addWidget(self.settings_panel)



        self.setLayout(layout)

        # Apply initial theme
        apply_theme(QApplication.instance(), cfg.theme)

    def toggle_theme(self, checked):
        """کلیک روی دکمه تغییر تم"""
        theme = "dark" if checked else "light"
        apply_theme(QApplication.instance(), theme)

        # ذخیره در config
        cfg.theme = theme
        x.save_config(cfg,CONFIG_FILE)
    def open_settings(self):
        """باز کردن پنجره‌ی تنظیمات"""
        from settings_dialog import SettingsDialog
        cfg = x.load_config(CONFIG_FILE)
        dlg = SettingsDialog(cfg, self)
        dlg.settings_changed.connect(self.apply_new_settings)
        dlg.exec_()

    def apply_new_settings(self, new_cfg):
        """اعمال تغییرات به مدل‌ها و ذخیره در حافظه"""
        global CONFIG
        CONFIG = new_cfg

        # ۱. دست‌نمونهٔ MediaPipe را بروزرسانی می‌کنیم
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=cfg.max_hands,
            min_detection_confidence=cfg.detection_conf,
            min_tracking_confidence=cfg.tracking_conf,
        )

        # ۲. Threshold ها را به GestureModel ارسال می‌کنیم
        self.gesture_model.thresholds = cfg.FOLD_THRESHOLD
        # (در تابع `classify` خود را مطابق با dict جدید بروزرسانی کنید)

        QMessageBox.information(self, "تنظیمات ذخیره شد",
                                "مقادیر جدید اعمال شد و فایل تنظیمات ذخیره شد.")
class SettingsPanel(QGroupBox):
    def __init__(self, cam_widget, parent=None):
        super().__init__("تنظیمات", parent)
        self.cam = cam_widget

        layout = QGridLayout()

        # روشنایی
        layout.addWidget(QLabel("روشنایی:"), 0, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.cam.set_brightness)
        layout.addWidget(self.brightness_slider, 0, 1)

        # کنتراست
        layout.addWidget(QLabel("کنتراست:"), 1, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(1, 300)   # 1 به 3.0
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(
            lambda v: self.cam.set_contrast(v / 100.0)
        )
        layout.addWidget(self.contrast_slider, 1, 1)

        # گاما
        layout.addWidget(QLabel("گاما:"), 2, 0)
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(1, 300)   # 1 به 3.0
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(
            lambda v: self.cam.set_gamma(v / 100.0)
        )
        layout.addWidget(self.gamma_slider, 2, 1)

        self.setLayout(layout)

# ------------------------------------------------------------------
# ۴. اجرای برنامه
# ------------------------------------------------------------------

def set_light_palette(app):
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#ffffff"))
    pal.setColor(QPalette.WindowText, Qt.black)
    pal.setColor(QPalette.Base, QColor("#f0f0f0"))
    pal.setColor(QPalette.AlternateBase, QColor("#e0e0e0"))
    pal.setColor(QPalette.ToolTipBase, Qt.white)
    pal.setColor(QPalette.ToolTipText, Qt.black)
    pal.setColor(QPalette.Text, Qt.black)
    pal.setColor(QPalette.Button, QColor("#dcdcdc"))
    pal.setColor(QPalette.ButtonText, Qt.black)
    pal.setColor(QPalette.BrightText, Qt.red)
    pal.setColor(QPalette.Link, QColor("#2a7ae2"))
    pal.setColor(QPalette.Highlight, QColor("#4a90e2"))
    pal.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(pal)


def set_dark_palette(app):
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#2d2d2d"))
    pal.setColor(QPalette.WindowText, Qt.white)
    pal.setColor(QPalette.Base, QColor("#3c3c3c"))
    pal.setColor(QPalette.AlternateBase, QColor("#4b4b4b"))
    pal.setColor(QPalette.ToolTipBase, Qt.white)
    pal.setColor(QPalette.ToolTipText, Qt.white)
    pal.setColor(QPalette.Text, Qt.white)
    pal.setColor(QPalette.Button, QColor("#3c3c3c"))
    pal.setColor(QPalette.ButtonText, Qt.white)
    pal.setColor(QPalette.BrightText, Qt.red)
    pal.setColor(QPalette.Link, QColor("#4a90e2"))
    pal.setColor(QPalette.Highlight, QColor("#5e5e5e"))
    pal.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(pal)


def apply_theme(app, theme_name):
    if theme_name == "light":
        with open('light.qss', 'r', encoding='utf-8') as f:
            app.setStyleSheet(f.read())
            set_light_palette(app)
    else:
        with open('dark.qss', 'r', encoding='utf-8') as f:
            app.setStyleSheet(f.read())
            set_dark_palette(app)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

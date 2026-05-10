# main.py
import os
import sys
import json
import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
    QHBoxLayout, QGroupBox, QGridLayout, QStyleFactory, QMessageBox, QSlider, QToolButton
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QIcon

from modularV0.Hand.detector   import HandDetector
from modularV0.Hand.processor  import HandProcessor
from modularV0.Hand.model      import GestureModel
from modularV0.Hand.excuter    import GestureExecutor
from modularV0.utils.Configurations import (
    CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT,
    DETECTION_CONFIDENCE, TRACKING_CONFIDENCE
)
import modularV0.utils.Config_Loader as x

# CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".hand_gesture_app.json")
CONFIG_FILE="x.json"
# ------------- تنظیمات خوانده شده -----------------
# try:
#     with open(CONFIG_FILE, "r", encoding="utf-8") as fp:
#         CONFIG = json.load(fp)
# except Exception:
#     # اگر فایل وجود نداشته باشد یا خراب باشد، از مقدار پیش‌فرض استفاده می‌کنیم
#     CONFIG = {
#         "max_hands": 2,
#         "detection_conf": 0.5,
#         "tracking_conf": 0.5,
#         "FOLD_THRESHOLD": {
#             "Thumb": 100,
#             "Index": 40,
#             "Middle": 40,
#             "Ring": 40,
#             "Pinky": 40
#         }
#     }




cfg =x.load_config (CONFIG_FILE)

class CameraWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)

        # --- باز کردن دوربین ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.setText("❌ دوربین پیدا نشد")
            return

        # --- تنظیمات اولیه ---
        self.brightness = 0
        self.contrast   = 0
        self.gamma      = 0

        # --- تایمر فریم گرفتن ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)   # ~30fps

        self.P_P_T_distances = {}
        self.hand_angles = {}
        self.finger_status = {}
        self.hand_gesture = {}
        self.hand_wrist_ang = {}

    # ------------------------------------------------------------------
    # ۱. فریم دریافت و تبدیل به QImage
    # ------------------------------------------------------------------
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # تبدیل BGR → RGB
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
    # ۲. متدهای تنظیمات
    # ------------------------------------------------------------------
    def set_brightness(self, v):
        self.brightness = v

    def set_contrast(self, v):
        self.contrast = v

    def set_gamma(self, v):
        self.gamma = v

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)


class HandProcessorWidget(QLabel):
    """
    ویجتی که تمام حلقه‌ی دست‌ها را درون خود می‌گیرد
    و در هر فریم به صورت غیر مسدود کننده اجرا می‌شود.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)

        # ----------- ضبط و تنظیمات دوربین -------------
        self.cap = cv2.VideoCapture(CAMERA_ID)
        if not self.cap.isOpened():
            self.setText("❌ دوربین پیدا نشد")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        # --- تنظیمات اولیه ---
        self.brightness = 0
        self.contrast = 1.0
        self.gamma = 1.0

        # ----------- آبجکت‌های پردازش ---------------
        self.detector  = HandDetector(
            max_hands=cfg.max_hands,
            detection_conf=cfg.detection_conf,
            tracking_conf=cfg.tracking_conf
        )
        self.processor = HandProcessor()
        self.model     = GestureModel(threshold=30)
        self.executor  = GestureExecutor()

        # ----------- حافظهٔ وضعیت -------------
        self.last_gesture = {"Left": None, "Right": None}

        # ----------- تایمر فریم -------------
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)  # ~30fps

    # def update_frame(self):
    #     ret, frame = self.cap.read()
    #     if not ret:
    #         logger.error("Could not read frame from camera")
    #
    #
    #     frame, hands = self.detector.process(frame)
    #
    #     # پردازش هر دست
    #     for hand in hands:
    #         lm = hand["landmarks"]
    #         handedness = hand["handedness"]  # 'Left' یا 'Right'
    #         # print(handedness)
    #         # print(lm[0])
    #         self.hand_angles[handedness] = ang = processor.Angele_Calculator(lm, frame)
    #         self.P_P_T_distances[handedness] = processor.Distance_norm_Calculator(lm, frame)
    #         # print(x)
    #         self.finger_status[handedness] = stat = processor.Finger_Status(ang)
    #         # print(y)
    #
    #         self.hand_gesture[handedness] = gesture = model.classify(stat, ang)
    #         print(self.hand_gesture)
    #         hand_wrist_ang = c = processor.Wrist_angel(lm, frame)
    #         # print(c)
    #
    #         # جلوگیری از فراخوانی مکرر همان دستور
    #
    #         # نمایش درجه ها برای هر دست ارنج رو باید اضاف کنم
    #         if handedness == "Left":
    #             y0, dy = 30, 20
    #         else:
    #             y0, dy = 150, 20
    #         for i, (finger, percent) in enumerate(ang.items()):
    #             text = f"{finger.capitalize()}: {percent}"
    #             y = y0 + i * dy
    #             cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
    #                         0.7, (0, 255, 255), 2)
    #             if handedness == "Left":
    #                 cv2.putText(frame, f"{handedness} Hand: {gesture}",
    #                             (400, 80), cv2.FONT_HERSHEY_SIMPLEX,
    #                             0.6, (0, 0, 0), 2)
    #             else:
    #                 cv2.putText(frame, f"{handedness} Hand: {gesture}",
    #                             (400, 100), cv2.FONT_HERSHEY_SIMPLEX,
    #                             0.6, (0, 0, 0), 2)
    #
    #
    #     # ----------- نمایش فریم در QLabel ---------------------------
    #     h, w, ch = frame.shape
    #     bytesPerLine = ch * w
    #     qImg = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
    #     pix = QPixmap.fromImage(qImg).scaled(
    #         self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    #     self.setPixmap(pix)
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            # اگر دوربین خاموش شود، نمایش خطا
            self.setText("❌ دریافت فریم از دوربین امکان‌پذیر نیست")
            return

        # اپدیت کردن کانفیگ ها در هر حلقه
        cfg = x.load_config(CONFIG_FILE)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # اعمال brightness / contrast / gamma
        frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)

        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(256)]).astype("uint8")
        frame = cv2.LUT(frame, table)
        # ---- پردازش دست‌ها ----
        frame, hands = self.detector.process(frame)

        # حافظه‌های موقتی برای نمایش
        hand_angles = {}
        hand_gesture = {}
        P_P_T_distances = {}
        finger_status = {}
        hand_wrist_ang = {}


        #Update config + khode tasvir
        self.processor.Config(x.load_config("x.json"))
        self.detector.Config(x.load_config("x.json"))


        # --------------------- حلقهٔ دست‌ها ---------------------
        for hand in hands:
            lm = hand["landmarks"]
            # print(hands)
            handedness = hand["handedness"]  # 'Left' یا 'Right'

            # ۱. زاویه
            ang = self.processor.Angele_Calculator(lm, frame)
            hand_angles[handedness] = ang

            # ۲. فاصله
            P_P_T_distances[handedness] = self.processor.Distance_norm_Calculator(lm, frame)
            # print(P_P_T_distances)
            # ۳. وضعیت انگشتان
            finger_status[handedness] = stat = self.processor.Finger_Status(ang)

            # ۴. گِیِست
            hand_gesture[handedness] = gesture = self.model.classify(stat, ang,P_P_T_distances[handedness])

            # ۵. زاویه‌ی آرنج (فقط برای نمایش)
            hand_wrist_ang[handedness] = self.processor.Wrist_angel(lm, frame)

            # ----- نمایش متنی در فریم -----
            y0, dy = (30, 20) if handedness == "Left" else (150, 20)
            for i, (finger, percent) in enumerate(ang.items()):
                text = f"{finger.capitalize()}: {percent}"
                y = y0 + i * dy
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)

            # گِیِست روی فریم
            txt = f"{handedness} Hand: {gesture}"
            x_pos = 400
            y_pos = 80 if handedness == "Left" else 100
            cv2.putText(frame, txt, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 2)
        # ----------- فرمان‌دهی دست راست وقتی دست چپ مشت است -----------
        if  hand_gesture.get('Left') and hand_gesture.get('Right'):
            if   hand_gesture['Left'] == 'fist':
                self.executor.execute_XMove_alt(hand_gesture.get('Right'),lm)
                # self.executor.process_gesture(hand_gesture.get('Right'),lm)
                self.last_gesture['Right'] = hand_gesture.get('Right')
                print(lm[0])

        # ----------- نمایش فریم در QLabel ---------------------------
        h, w, ch = frame.shape
        bytesPerLine = ch * w
        qImg = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qImg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pix)
        # print (cfg.FOLD_THRESHOLD)

    def set_brightness(self, v):
        self.brightness = v

    def set_contrast(self, v):
        self.contrast = v

    def set_gamma(self, v):
        self.gamma = v


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
        self.setWindowTitle("Hand Gesture CV")
        self.resize(900, 600)

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
        self.theme_btn.setToolTip("تغییر رنگ (آتش یا تاریکی)")
        self.theme_btn.setIcon(QIcon.fromTheme("weather-clear-night"))  # آیکون اختیاری
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
        set_light_palette(app)
    else:
        set_dark_palette(app)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_theme(app,'dark')        # یا set_light_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

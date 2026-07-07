# # # مثال با PySide6
# # from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout,
# #                                QLineEdit, QCheckBox, QPushButton, QApplication)
# #
# # class SettingsWidget(QWidget):
# #     def __init__(self):
# #         super().__init__()
# #         self.setWindowTitle("تنظیمات برنامه")
# #
# #         form = QFormLayout()
# #         form.addRow("مقدار آدرس:", QLineEdit())
# #         form.addRow("اجازهٔ کارکرد:", QCheckBox("فعال"))
# #         form.addRow("حداکثر زمان:", QLineEdit())
# #
# #         btn_save = QPushButton("ذخیره")
# #         btn_cancel = QPushButton("لغو")
# #
# #         btn_layout = QVBoxLayout()
# #         btn_layout.addWidget(btn_save)
# #         btn_layout.addWidget(btn_cancel)
# #
# #         main = QVBoxLayout()
# #         main.addLayout(form)
# #         main.addLayout(btn_layout)
# #
# #         self.setLayout(main)
# #
# # if __name__ == "__main__":
# #     import sys
# #     app = QApplication(sys.argv)
# #     win = SettingsWidget()
# #     win.show()
# #     sys.exit(app.exec())
# # university_hover_timer.py
# # import sys
# # from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
# # from PyQt5.QtCore import Qt, QTimer
# #
# # class HoverLabel(QLabel):
# #
# #     def __init__(self, normal_text, hover_text, delay_ms=2000, parent=None):
# #         super().__init__(normal_text, parent)
# #
# #         self.normal_text = normal_text
# #         self.hover_text = hover_text
# #         self.delay_ms = delay_ms
# #
# #         self.setAttribute(Qt.WA_Hover, True)
# #
# #         self.timer = QTimer(self)
# #         self.timer.setSingleShot(True)
# #         self.timer.timeout.connect(self._show_hover_text)
# #
# #         self.setStyleSheet("""
# #             QLabel {
# #                 font-size: 18px;
# #                 padding: 10px;
# #                 background-color: #f9f9f9;
# #                 border: 1px solid #bbb;
# #                 border-radius: 5px;
# #             }
# #         """)
# #
# #     def enterEvent(self, event):
# #
# #         if self.text() == self.normal_text:
# #             self.timer.start(self.delay_ms)
# #         super().enterEvent(event)
# #
# #     def leaveEvent(self, event):
# #
# #         if self.timer.isActive():
# #             self.timer.stop()
# #         self.setText(self.normal_text)
# #         super().leaveEvent(event)
# #
# #     def _show_hover_text(self):
# #         self.setText(self.hover_text)
# #
# # class MainWindow(QWidget):
# #     def __init__(self):
# #         super().__init__()
# #         self.setWindowTitle("Shamsi Por University")
# #         self.setGeometry(100, 100, 400, 200)
# #
# #         self.label = HoverLabel(
# #             normal_text="دانشگاه ملی",
# #             hover_text="دانشگاه ملی شهید شمسی پور",
# #             delay_ms=2000,
# #             parent=self
# #         )
# #
# #         layout = QVBoxLayout()
# #         layout.addWidget(self.label)
# #         layout.setAlignment(Qt.AlignCenter)
# #         self.setLayout(layout)
# #
# # if __name__ == "__main__":
# #     app = QApplication(sys.argv)
# #     win = MainWindow()
# #     win.show()
# #     sys.exit(app.exec_())
# # def merge_sort(arr):
# #
# #     if len(arr) <= 1:
# #         return arr
# #
# #     mid = len(arr) // 2
# #     left_half = merge_sort(arr[:mid])
# #     right_half = merge_sort(arr[mid:])
# #
# #     return merge(left_half, right_half)
# #
# # def merge(left, right):
# #
# #     merged = []
# #     i = j = 0
# #
# #     while i < len(left) and j < len(right):
# #         if left[i] <= right[j]:
# #             merged.append(left[i])
# #             i += 1
# #         else:
# #             merged.append(right[j])
# #             j += 1
# #
# #
# #     if i < len(left):
# #         merged.extend(left[i:])
# #     if j < len(right):
# #         merged.extend(right[j:])
# #
# #     return merged
# #
# #
# #
# #
# # if __name__ == "__main__":
# #     unsorted = [34, 7, 23, 32, 5, 62, 0, -1, 19,25,40,88,76]
# #     print("قبل از مرتب‌سازی:", unsorted)
# #
# #     sorted_list = merge_sort(unsorted)
# #     print("بعد از مرتب‌سازی :", sorted_list)
#
#
#
#
# import sys
# import cv2
# import numpy as np
# from PyQt5.QtCore import Qt, QTimer, QEvent
# from PyQt5.QtGui import QImage, QPalette, QColor, QKeySequence, QPixmap
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
#     QSlider, QHBoxLayout, QGroupBox, QGridLayout, QShortcut, QStyleFactory
# )
#
# # ------------------------------------------------------------------
# # ۱. کلاس برای نمایش دوربین
# # ------------------------------------------------------------------
# class CameraWidget(QLabel):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setAlignment(Qt.AlignCenter)
#
#         # دوربین را باز می‌کنیم
#         self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             self.setText("❌ دوربین پیدا نشد")
#             return
#
#         # تایمر برای گرفتن فریم
#         self.timer = QTimer(self)
#         self.timer.timeout.connect(self.update_frame)
#         self.timer.start(30)          # ~30fps
#
#         # تنظیمات اولیه
#         self.brightness = 0
#         self.contrast   = 1.0
#         self.gamma      = 1.0
#
#     def update_frame(self):
#         ret, frame = self.cap.read()
#         if not ret:
#             return
#
#         # تبدیل BGR (OpenCV) به RGB (Qt)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # اعمال تنظیمات ساده
#         frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)
#
#         # gamma correction (اختیاری)
#         invGamma = 1.0 / self.gamma
#         table = np.array([((i / 255.0) ** invGamma) * 255
#                           for i in np.arange(256)]).astype("uint8")
#         frame = cv2.LUT(frame, table)
#
#         # تبدیل به QImage
#         h, w, ch = frame.shape
#         bytesPerLine = ch * w
#         qImg = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
#
#         # تبدیل به QPixmap، اسکیل کردن و نمایش
#         pix = QPixmap.fromImage(qImg).scaled(
#             self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         self.setPixmap(pix)
#
#     # ---- متدهای تنظیمات ----
#     def set_brightness(self, v):  self.brightness = v
#     def set_contrast(self, v):    self.contrast   = v
#     def set_gamma(self, v):       self.gamma      = v
#
#     def closeEvent(self, event):
#         if self.cap.isOpened():
#             self.cap.release()
#         super().closeEvent(event)
# # ------------------------------------------------------------------
# # ۲. ویجت تنظیمات
# # ------------------------------------------------------------------
# class SettingsPanel(QGroupBox):
#     def __init__(self, cam_widget, parent=None):
#         super().__init__("تنظیمات", parent)
#         self.cam = cam_widget
#         layout = QGridLayout()
#
#         # روشنایی
#         layout.addWidget(QLabel("روشنایی:"), 0, 0)
#         self.brightness_slider = QSlider(Qt.Horizontal)
#         self.brightness_slider.setRange(-100, 100)
#         self.brightness_slider.setValue(0)
#         self.brightness_slider.valueChanged.connect(self.cam.set_brightness)
#         layout.addWidget(self.brightness_slider, 0, 1)
#
#         # کنتراست
#         layout.addWidget(QLabel("کنتراست:"), 1, 0)
#         self.contrast_slider = QSlider(Qt.Horizontal)
#         self.contrast_slider.setRange(1, 300)   # 1 به 3.0
#         self.contrast_slider.setValue(100)
#         self.contrast_slider.valueChanged.connect(
#             lambda v: self.cam.set_contrast(v / 100.0)
#         )
#         layout.addWidget(self.contrast_slider, 1, 1)
#
#         # گاما
#         layout.addWidget(QLabel("گاما:"), 2, 0)
#         self.gamma_slider = QSlider(Qt.Horizontal)
#         self.gamma_slider.setRange(1, 300)   # 1 به 3.0
#         self.gamma_slider.setValue(100)
#         self.gamma_slider.valueChanged.connect(
#             lambda v: self.cam.set_gamma(v / 100.0)
#         )
#         layout.addWidget(self.gamma_slider, 2, 1)
#
#         self.setLayout(layout)
#
# # ------------------------------------------------------------------
# # ۳. کلاس اصلی پنجره
# # ------------------------------------------------------------------
# class MainWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("ComputerVision")
#         self.resize(900, 600)
#
#         # ------------------------------------------------------------------
#         # ۳.۱. سبک و رنگ ویندوز (System)
#         # ------------------------------------------------------------------
#         QApplication.setStyle(QStyleFactory.create('Fusion'))
#         palette = self.palette()
#         # رنگ‌های پیش‌فرض ویندوز در Fusion
#         palette.setColor(QPalette.Window, QColor(240, 240, 240))
#         palette.setColor(QPalette.Base, QColor(255, 255, 255))
#         palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
#         palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
#         palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
#         palette.setColor(QPalette.Text, QColor(0, 0, 0))
#         palette.setColor(QPalette.Button, QColor(240, 240, 240))
#         palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
#         palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
#         palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
#         palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
#         self.setPalette(palette)
#
#         # ------------------------------------------------------------------
#         # ۳.۲. ویجت‌ها
#         # ------------------------------------------------------------------
#         self.cam_widget = CameraWidget(self)
#
#         self.settings_panel = SettingsPanel(self.cam_widget, self)
#
#         # دکمه‌های کنترل
#         self.start_btn = QPushButton("شروع")
#         self.stop_btn = QPushButton("توقف")
#         self.start_btn.clicked.connect(self.start_camera)
#         self.stop_btn.clicked.connect(self.stop_camera)
#
#         # shortcut: Space = start/stop
#         QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_camera)
#
#         # ------------------------------------------------------------------
#         # ۳.۳. Layout
#         # ------------------------------------------------------------------
#         layout = QVBoxLayout()
#         layout.addWidget(self.cam_widget, stretch=3)
#
#         hbox = QHBoxLayout()
#         hbox.addWidget(self.start_btn)
#         hbox.addWidget(self.stop_btn)
#         hbox.addStretch()
#         hbox.addWidget(self.settings_panel)
#
#         layout.addLayout(hbox)
#         self.setLayout(layout)
#
#     # ------------------------------------------------------------------
#     # ۳.۴. متدهای کنترل دوربین
#     # ------------------------------------------------------------------
#     def start_camera(self):
#         if not self.cam_widget.timer.isActive():
#             self.cam_widget.timer.start(30)
#
#     def stop_camera(self):
#         if self.cam_widget.timer.isActive():
#             self.cam_widget.timer.stop()
#
#     def toggle_camera(self):
#         if self.cam_widget.timer.isActive():
#             self.stop_camera()
#         else:
#             self.start_camera()
#
#     def closeEvent(self, event):
#         self.cam_widget.cap.release()
#         super().closeEvent(event)
#
# # ------------------------------------------------------------------
# # ۴. اجرای برنامه
# # ------------------------------------------------------------------
# def main():
#     app = QApplication(sys.argv)
#     win = MainWindow()
#     win.show()
#     sys.exit(app.exec_())
#
# if __name__ == "__main__":
#     main()



















































# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#

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
# import sys
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QStackedWidget,
#     QVBoxLayout, QFormLayout, QLineEdit, QSpinBox,
#     QPushButton, QLabel, QMessageBox
# )
# import matplotlib as mp
# from PyQt5.QtCore import Qt
#
# class StudentApp(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("محاسبه معدل دانشجو")
#         self.setFixedSize(400, 150)
#
#         # ------------------------------------------------------------------
#         # داده‌های ذخیره‌شده در طول فرم
#         self.student_name  = ""
#         self.student_id    = ""
#         self.courses       = []      # لیست از سه نام درس
#         self.grades        = []      # لیست از سه نمره
#         # ------------------------------------------------------------------
#
#         # بنای صفحه‌ها
#         self.stack = QStackedWidget()
#         self.page1 = self.create_page1()
#         self.page2 = self.create_page2()
#         self.page3 = self.create_page3()
#
#         self.stack.addWidget(self.page1)
#         self.stack.addWidget(self.page2)
#         self.stack.addWidget(self.page3)
#
#         main_layout = QVBoxLayout()
#         main_layout.addWidget(self.stack)
#         self.setLayout(main_layout)
#
#     # ----------------------------------------------------------------------
#     # صفحه‌ی اول : نام و شماره دانشجویی
#     # ----------------------------------------------------------------------
#     def create_page1(self):
#         page = QWidget()
#         layout = QFormLayout()
#
#         self.name_edit = QLineEdit()
#         self.id_edit   = QLineEdit()
#
#         layout.addRow("نام و نام خانوادگی :", self.name_edit)
#         layout.addRow("شماره دانشجویی :", self.id_edit)
#
#
#         next_btn = QPushButton("دنبال‌ی‌نویس")
#         next_btn.clicked.connect(self.go_to_page2)
#         layout.addRow(next_btn)
#
#         page.setLayout(layout)
#         return page
#
#     def go_to_page2(self):
#         name = self.name_edit.text().strip()
#         sid  = self.id_edit.text().strip()
#
#         # اعتبارسنجی ساده
#         if not name:
#             QMessageBox.warning(self, "خطا", "لطفاً نام را وارد کنید.")
#             return
#         if not sid.isdigit():
#             QMessageBox.warning(self, "خطا", "شماره دانشجویی باید عددی باشد.")
#             return
#
#         self.student_name = name
#         self.student_id   = sid
#
#         # پاک کردن فیلدها برای موارد بعدی
#         self.name_edit.clear()
#         self.id_edit.clear()
#         self.stack.setCurrentIndex(1)
#
#     # ----------------------------------------------------------------------
#     # صفحه‌ی دوم : نام سه درس
#     # ----------------------------------------------------------------------
#     def create_page2(self):
#         page = QWidget()
#         layout = QFormLayout()
#
#         self.course_edits = []
#         for i in range(3):
#             le = QLineEdit()
#             self.course_edits.append(le)
#             layout.addRow(f"درس {i+1} :", le)
#
#         next_btn = QPushButton("دنبال‌ی‌نویس")
#         next_btn.clicked.connect(self.go_to_page3)
#         layout.addRow(next_btn)
#
#         page.setLayout(layout)
#         return page
#
#     def go_to_page3(self):
#         courses = [le.text().strip() for le in self.course_edits]
#
#         if any(not c for c in courses):
#             QMessageBox.warning(self, "خطا", "لطفاً نام همه‌ی سه درس را وارد کنید.")
#             return
#
#         self.courses = courses
#
#         # پاک کردن فیلدها
#         for le in self.course_edits:
#             le.clear()
#         self.stack.setCurrentIndex(2)
#
#     # ----------------------------------------------------------------------
#     # صفحه‌ی سوم : نمره هر درس
#     # ----------------------------------------------------------------------
#     def create_page3(self):
#         page = QWidget()
#         layout = QFormLayout()
#
#         self.grade_spins = []
#         for i in range(3):
#             spin = QSpinBox()
#             spin.setRange(0, 20)
#             spin.setSingleStep(0.5)
#             self.grade_spins.append(spin)
#             layout.addRow(f"نمره درس {i+1} :", spin)
#
#         finish_btn = QPushButton("محاسبه معدل")
#         finish_btn.clicked.connect(self.show_result)
#         layout.addRow(finish_btn)
#
#         page.setLayout(layout)
#         return page
#
#     def show_result(self):
#         grades = [spin.value() for spin in self.grade_spins]
#
#         # محاسبه معدل (میانگین ساده)
#         avg = sum(grades) / len(grades) if grades else 0
#
#         msg = (
#             f"نام و نام خانوادگی : {self.student_name}\n"
#             f"شماره دانشجویی : {self.student_id}\n"
#             f"معدل : {avg:.2f}"
#         )
#         QMessageBox.information(self, "نتیجه", msg)
#
#         # بازنشانی برای ورود اطلاعات جدید
#         self.grade_spins = []
#         self.stack.setCurrentIndex(0)
#
# # --------------------------------------------------------------------------
# # اجرای برنامه
# # --------------------------------------------------------------------------
# def main():
#     app = QApplication(sys.argv)
#     win = StudentApp()
#     win.show()
#     sys.exit(app.exec_())
#
# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout,
    QLineEdit, QComboBox, QPushButton, QLabel, QMessageBox
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt

# Matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --------------------------------------------------------------------
# دسته‌بندی‌های BMI (نام، حداقل، حداکثر)
BMI_CATEGORIES = [
    ("lake weight",      0,  18.5),
    ("normal", 18.5, 24.9),
    ("almost fat", 25.0, 29.9),
    ("fat",       30.0, 100)   # سقف بالا
]

# --------------------------------------------------------------------
class BMICanvas(FigureCanvas):
    """Matplotlib canvas داخل PyQt5."""
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.update_bmi(0)          # نمودار اولیه (بدون نشانگر)

    def update_bmi(self, bmi_value: float):
        """نمودار را با دسته‌های BMI رسم می‌کند و bar مناسب را روشن می‌کند."""
        self.ax.clear()
        bars = []
        colors = []

        # رسم دسته‌های BMI
        for label, low, high in BMI_CATEGORIES:
            bars.append(high)
            if low <= bmi_value < high:
                colors.append("#5cb85c")   # سبز
            else:
                colors.append("#d3d3d3")   # خاکستری روشن

        y_pos = range(len(BMI_CATEGORIES))
        self.ax.barh(y_pos, bars, color=colors, align='center')
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels([c[0] for c in BMI_CATEGORIES])
        self.ax.set_xlabel("BMI")
        self.ax.set_xlim(0, 40)   # حداکثر عرض برای نمایش همه دسته‌ها

        # نقطه‌ای روی مقدار واقعی BMI
        if 0 < bmi_value < 40:
            idx = next(i for i, cat in enumerate(BMI_CATEGORIES)
                       if cat[1] <= bmi_value < cat[2])
            self.ax.plot(bmi_value, idx, "ro")  # نقطه قرمز

        self.ax.grid(axis='x', linestyle='--', alpha=0.7)
        self.draw()


# --------------------------------------------------------------------
class BMICalculator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("محاسبه BMI و وضعیت بدن")
        self.setFixedSize(480, 550)

        # لایوت اصلی
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # فرم ورودی
        form = QFormLayout()

        self.weight_edit = QLineEdit()
        self.weight_edit.setPlaceholderText("مثال: 70")
        self.weight_edit.setValidator(QDoubleValidator(0.1, 500.0, 1))
        form.addRow("وزن (kg):", self.weight_edit)

        self.height_edit = QLineEdit()
        self.height_edit.setPlaceholderText("مثال: 170")
        self.height_edit.setValidator(QDoubleValidator(10.0, 250.0, 1))
        form.addRow("ارتفاع (cm):", self.height_edit)

        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["مرد", "زن"])
        form.addRow("جنسیت:", self.gender_combo)

        main_layout.addLayout(form)

        # دکمه محاسبه
        self.calc_btn = QPushButton("محاسبه BMI")
        self.calc_btn.clicked.connect(self.calculate_bmi)
        main_layout.addWidget(self.calc_btn, alignment=Qt.AlignCenter)

        # نمایش نتایج
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(self.result_label)

        # نمودار
        self.canvas = BMICanvas(self, width=5, height=3)
        main_layout.addWidget(self.canvas, stretch=1)

    # ------------------------------------------------------------------
    def calculate_bmi(self):
        """محاسبه BMI، تعیین وضعیت و به‌روزرسانی UI و نمودار."""
        try:
            weight = float(self.weight_edit.text())
            height_cm = float(self.height_edit.text())
            if weight <= 0 or height_cm <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(
                self, "ورودی نامعتبر",
                "لطفاً وزن و ارتفاع معتبر (عدد مثبت) وارد کنید."
            )
            return

        height_m = height_cm / 100.0
        bmi = weight / (height_m ** 2)
        bmi = round(bmi, 2)

        # تعیین وضعیت
        status = next(
            label for label, low, high in BMI_CATEGORIES
            if low <= bmi < high
        )

        # نمایش متن
        self.result_label.setText(f"وزن‌فشاری (BMI): {bmi}\nوضعیت: {status}")

        # به‌روزرسانی نمودار
        self.canvas.update_bmi(bmi)


# --------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    win = BMICalculator()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

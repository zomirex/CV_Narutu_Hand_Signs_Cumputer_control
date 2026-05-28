# settings_dialog.py
"""
پنجره‌ی تنظیمات (SettingsDialog) که در برنامه‌ی اصلی
از طریق یک دکمه‌ی «تنظیمات» باز می‌شود.
اینجا مقادیر اولیه (تعداد دست، دقت تشخیص/ردیابی،
threshold هر انگشت، و تم رنگی) را می‌توانید ویرایش کنید
و سپس با ذخیره‌ی تغییرات، فایل JSON `x.json`
به‌روز می‌شود و سیگنال `settings_changed` به
`MainWindow` ارسال می‌شود تا مدل‌ها و ویجت‌ها
به‌روز شوند.
"""

import json
import os

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QPushButton, QHeaderView, QMessageBox,
    QAbstractItemView, QWidget, QSpacerItem, QSizePolicy, QInputDialog
)
from PyQt5.QtCore import Qt, pyqtSignal

# مسیر فایل تنظیمات (به‌صورت نسبی یا مطلق می‌تواند باشد)
CONFIG_PATH = 'x.json'


class SettingsDialog(QDialog):
    """پنجره‌ی تنظیمات

    این کلاس یک `QDialog` ساده است که درون آن فیلدهای عددی
    برای `max_hands`, `detection_conf`, `tracking_conf`
    و یک جدول برای تنظیم thresholds هر انگشت قرار دارد.
    در زمان ذخیره‌ی تغییرات، فایل `x.json` بازنویسی می‌شود
    و یک سیگنال به نام `settings_changed` ارسال می‌شود
    تا پنجره‌ی اصلی بتواند تغییرات را اعمال کند.
    """

    # سیگنال برای ارسال دیکشنری جدید تنظیمات به `MainWindow`
    settings_changed = pyqtSignal(dict)

    def __init__(self, current_settings, parent=None):
        """
        پارامتر `current_settings` یک شیء (معمولاً از `cfg` در `main.py`)
        است که دارای ویژگی‌هایی مثل `max_hands`,
        `detection_conf`, `tracking_conf`, و `FOLD_THRESHOLD` می‌باشد.
        """
        super().__init__(parent)
        self.setWindowTitle("تنظیمات")
        self.setMinimumWidth(300)

        # ------------------------------------------------------------------
        # 1️⃣ فیلدهای عددی اولیه
        # ------------------------------------------------------------------
        self.max_hands_sb = QSpinBox()
        self.max_hands_sb.setRange(1, 10)              # حداقل ۱ تا حداکثر ۱۰ دست
        self.max_hands_sb.setValue(current_settings.max_hands)

        self.det_conf_sb = QDoubleSpinBox()
        self.det_conf_sb.setRange(0.0, 1.0)           # 0 تا 1 برای دقت
        self.det_conf_sb.setSingleStep(0.05)
        self.det_conf_sb.setValue(current_settings.detection_conf)

        self.track_conf_sb = QDoubleSpinBox()
        self.track_conf_sb.setRange(0.0, 1.0)
        self.track_conf_sb.setSingleStep(0.05)
        self.track_conf_sb.setValue(current_settings.tracking_conf)

        # ------------------------------------------------------------------
        # 2️⃣ جدول thresholds (Finger | Threshold)
        # ------------------------------------------------------------------
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Finger", "Threshold"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # بارگذاری مقادیر اولیه در جدول
        self.load_thresholds(current_settings.FOLD_THRESHOLD)

        # ------------------------------------------------------------------
        # 3️⃣ دکمه‌های ذخیره/لغو
        # ------------------------------------------------------------------
        ok_btn = QPushButton("ذخیره")
        cancel_btn = QPushButton("لغو")
        ok_btn.clicked.connect(self.save_settings)   # فراخوانی متد ذخیره
        cancel_btn.clicked.connect(self.reject)      # ببند به‌صورت رد

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        # ------------------------------------------------------------------
        # 4️⃣ Layout کلی
        # ------------------------------------------------------------------
        layout = QVBoxLayout()
        # فیلدهای عددی
        layout.addWidget(QLabel("حداکثر دست‌ها (max_hands):"))
        layout.addWidget(self.max_hands_sb)
        layout.addWidget(QLabel("حداقل تشخیص (0-1):"))
        layout.addWidget(self.det_conf_sb)
        layout.addWidget(QLabel("حداقل ردیابی (0-1):"))
        layout.addWidget(self.track_conf_sb)

        layout.addSpacing(15)
        layout.addWidget(QLabel("Thresholds (دیکشنری FOLD_THRESHOLD):"))
        layout.addWidget(self.table)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    # ----------------------------------------------------------------------
    # Helper: بارگذاری thresholds به جدول
    # ----------------------------------------------------------------------
    def load_thresholds(self, thresholds):
        """
        `thresholds` یک دیکشنری است به شکل
        {"Thumb": 100, "Index": 40, …}
        که در جدول دو ستون نمایش داده می‌شود.
        ستون اول نام انگشت (قابل ویرایش نیست) و ستون دوم
        یک `QSpinBox` برای تغییر مقدار threshold است.
        """
        self.table.setRowCount(0)
        for finger, value in thresholds.items():
            row = self.table.rowCount()
            self.table.insertRow(row)

            # ستون 0: نام انگشت (متن، قابل ویرایش نیست)
            txt_item = QTableWidgetItem(finger)
            txt_item.setFlags(txt_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, txt_item)

            # ستون 1: spinbox برای مقدار threshold
            spin = QSpinBox()
            spin.setRange(0, 200)            # مقادیر دلخواه
            spin.setValue(int(value))
            self.table.setCellWidget(row, 1, spin)

    # ----------------------------------------------------------------------
    # Helper: ذخیره تنظیمات، نوشتن در فایل JSON و انتشار سیگنال
    # ----------------------------------------------------------------------
    def save_settings(self):
        """
        ۱. مقادیر جدول را به دیکشنری `thresholds` تبدیل می‌کنیم.
        ۲. دیکشنری جدید شامل `max_hands`, `detection_conf`,
           `tracking_conf`, `FOLD_THRESHOLD` و `theme` ساخته می‌شود.
        ۳. فایل JSON بازنویسی می‌شود.
        ۴. سیگنال `settings_changed` با پارامتر دیکشنری جدید ارسال می‌شود.
        ۵. پنجره‌ی تنظیمات بسته می‌شود (با `accept()`).
        """
        # thresholds جدید
        thresholds = {}
        for row in range(self.table.rowCount()):
            finger = self.table.item(row, 0).text()
            spin = self.table.cellWidget(row, 1)
            thresholds[finger] = spin.value()

        # ساخت دیکشنری نهایی
        new_cfg = {
            "max_hands": self.max_hands_sb.value(),
            "detection_conf": self.det_conf_sb.value(),
            "tracking_conf": self.track_conf_sb.value(),
            "FOLD_THRESHOLD": thresholds,
            "theme": "dark"   # می‌توانید تغییر دهید (مثلاً `light`)
        }

        # نوشتن در فایل
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as fp:
                json.dump(new_cfg, fp, indent=4, ensure_ascii=False)
        except IOError as e:
            QMessageBox.critical(self, "خطا", f"نمی‌توان فایل را ذخیره کرد: {e}")
            return

        # انتشار سیگنال
        self.settings_changed.emit(new_cfg)
        self.accept()   # بسته شدن پنجره (به صورت پذیرش)

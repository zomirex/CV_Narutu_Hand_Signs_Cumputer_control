# settings_dialog.py
import json
import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QPushButton, QHeaderView, QMessageBox,
    QAbstractItemView, QWidget, QSpacerItem, QSizePolicy, QInputDialog
)
from PyQt5.QtCore import Qt, pyqtSignal

# CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".hand_gesture_config.json")
CONFIG_PATH = 'x.json'


class SettingsDialog(QDialog):
    """پنجره تنظیمات: مقادیر اولیه و thresholds انگشتان"""

    # سیگنال برای ارسال تغییرات به MainWindow
    settings_changed = pyqtSignal(dict)

    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("تنظیمات")
        self.setMinimumWidth(420)

        # 1️⃣ فیلدهای عددی اولیه
        self.max_hands_sb = QSpinBox()
        self.max_hands_sb.setRange(1, 10)
        self.max_hands_sb.setValue(current_settings.max_hands)

        self.det_conf_sb = QDoubleSpinBox()
        self.det_conf_sb.setRange(0.0, 1.0)
        self.det_conf_sb.setSingleStep(0.05)
        self.det_conf_sb.setValue(current_settings.detection_conf)

        self.track_conf_sb = QDoubleSpinBox()
        self.track_conf_sb.setRange(0.0, 1.0)
        self.track_conf_sb.setSingleStep(0.05)
        self.track_conf_sb.setValue(current_settings.tracking_conf)

        # 2️⃣ جدول thresholds (Finger | Threshold)
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Finger", "Threshold"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.load_thresholds(current_settings.FOLD_THRESHOLD)

        # 3️⃣ دکمه‌های ذخیره/لغو
        ok_btn = QPushButton("ذخیره")
        cancel_btn = QPushButton("لغو")
        ok_btn.clicked.connect(self.save_settings)
        cancel_btn.clicked.connect(self.reject)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        # 4️⃣ Layout کلی
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

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------
    def load_thresholds(self, thresholds):
        """بارگذاری thresholds به جدول"""
        self.table.setRowCount(0)
        for finger, value in thresholds.items():
            row = self.table.rowCount()
            self.table.insertRow(row)

            # ستون 0: نام انگشت (متن قابل ویرایش نیست)
            txt_item = QTableWidgetItem(finger)
            txt_item.setFlags(txt_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, txt_item)

            # ستون 1: spinbox برای مقدار threshold
            spin = QSpinBox()
            spin.setRange(0, 200)            # مقادیر دلخواه
            spin.setValue(int(value))
            self.table.setCellWidget(row, 1, spin)

    def save_settings(self):
        """ذخیره در JSON و انتشار سیگنال"""
        # thresholds جدید
        thresholds = {}
        for row in range(self.table.rowCount()):
            finger = self.table.item(row, 0).text()
            spin = self.table.cellWidget(row, 1)
            thresholds[finger] = spin.value()

        new_cfg = {
            "max_hands": self.max_hands_sb.value(),
            "detection_conf": self.det_conf_sb.value(),
            "tracking_conf": self.track_conf_sb.value(),
            "FOLD_THRESHOLD": thresholds,
            "theme": "dark"
        }

        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as fp:
                json.dump(new_cfg, fp, indent=4, ensure_ascii=False)
        except IOError as e:
            QMessageBox.critical(self, "خطا", f"نمی‌توان فایل را ذخیره کرد: {e}")
            return

        self.settings_changed.emit(new_cfg)
        self.accept()
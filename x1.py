#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QGridLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QSizePolicy,
)
from PyQt5.QtCore import Qt


class Calculator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("(تکلیف سوم (ماشین حساب")
        self.setFixedSize(260, 340)
        self.expression = ""
        self.init_ui()


    def init_ui(self):
        # نمایش نتیجه
        self.display = QLineEdit()
        self.display.setReadOnly(True)
        self.display.setAlignment(Qt.AlignRight)
        self.display.setFixedHeight(50)
        self.display.setStyleSheet("font: 18pt; background: #EEE;")


        buttons = {
            'C': (0, 0),
            '←': (0, 1),
            '/': (0, 2),
            '*': (0, 3),
            '7': (1, 0),
            '8': (1, 1),
            '9': (1, 2),
            '-': (1, 3),
            '4': (2, 0),
            '5': (2, 1),
            '6': (2, 2),
            '+': (2, 3),
            '1': (3, 0),
            '2': (3, 1),
            '3': (3, 2),
            '=': (3, 3, 1, 2),   # 2 سطر عمودی
            '0': (4, 0, 1, 2),   # 2 سطر افقی
            '.': (4, 2),
        }

        grid = QGridLayout()
        grid.setSpacing(5)

        for btn_text, pos in buttons.items():
            button = QPushButton(btn_text)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            button.setStyleSheet(
                """
                QPushButton {
                    font: 16pt;
                    background: #FFF;
                }
                QPushButton:hover {
                    background: #DDD;
                }
                """
            )
            if len(pos) == 2:
                grid.addWidget(button, pos[0], pos[1])
            else:
                grid.addWidget(button, pos[0], pos[1], pos[2], pos[3])


            if btn_text == 'C':
                button.clicked.connect(self.clear_display)
            elif btn_text == '←':
                button.clicked.connect(self.backspace)
            elif btn_text == '=':
                button.clicked.connect(self.calculate)
            else:
                button.clicked.connect(lambda _, txt=btn_text: self.add_to_expression(txt))

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.display)
        main_layout.addLayout(grid)

        self.setLayout(main_layout)









    def add_to_expression(self, char: str):
        self.expression += char
        self.display.setText(self.expression)

    def clear_display(self):
        self.expression = ""
        self.display.setText("")

    def backspace(self):
        self.expression = self.expression[:-1]
        self.display.setText(self.expression)

    def calculate(self):
        if not self.expression:
            return

        try:
            result = eval(self.expression)
            self.display.setText(str(result))
            self.expression = str(result)
        except Exception:
            self.display.setText("خطا")
            self.expression = ""



# ---------------------------------------------
# اجرای برنامه
# ---------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    calc = Calculator()
    calc.show()
    sys.exit(app.exec_())

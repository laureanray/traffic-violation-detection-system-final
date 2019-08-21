from PyQt5.QtWidgets import QWidget, QPlainTextEdit, QPushButton, QHBoxLayout, QVBoxLayout
from datetime import datetime


class Logging(QWidget):
    def __init__(self):
        super().__init__()

        # PlainTextEdit for the logtext
        self.line_edit = QPlainTextEdit(self)
        self.line_edit.setReadOnly(True)
        self.line_edit.setFixedHeight(150)


        self.clear_button = QPushButton("Clear")

        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(self.line_edit)
        layout.addWidget(self.clear_button)

        # self.setGeometry(300, 300, 200, 50)
        self.setWindowTitle('Log')


    def log(self, log_origin, log_details):
        now = datetime.now()
        datetime_string = now.strftime("%d/%m/%Y %H:%M:%S")
        self.line_edit.appendPlainText(datetime_string + "[" + log_origin + "] " + log_details)





from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets, uic


class Config(QWidget):
    def __init__(self):
        super(Config, self).__init__()
        # Load GU
        uic.loadUi('./user_interface/config.ui', self)

        # # Connect click event`1
        # self.loadModelButton.clicked.connect(self.onClickedLoadModelButton)
        # self.closeButton.clicked.connect(self.close)
        self.closeButton.clicked.connect(self.close)

    # def onClickedLoadModelButton(self):
    #     options = QFileDialog.Options()
    #     options |= QFileDialog.DontUseNativeDialog
    #     fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
    #                                               "All Files (*);;Python Files (*.py)", options=options)
    #     if fileName:
    #         print(fileName)
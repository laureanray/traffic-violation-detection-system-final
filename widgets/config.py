from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets, uic


class Config(QWidget):
    def __init__(self):
        super(Config, self).__init__()
        # Load GU
        uic.loadUi('./user_interface/ConfigView.ui', self)

        # Connect click event
        self.loadModelButton.clicked.connect(self.onClickedLoadModelButton)
        self.closeButton.clicked.connect(self.close)


    def onClickedLoadModelButton(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)


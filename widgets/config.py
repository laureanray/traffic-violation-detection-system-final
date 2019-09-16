from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import config_manager as config
from application import State


class Config(QWidget):
    def __init__(self):
        super(Config, self).__init__()
        # Load GU
        uic.loadUi('./user_interface/config.ui', self)

        # # Connect click event`1
        # self.loadModelButton.clicked.connect(self.onClickedLoadModelButton)
        # self.closeButton.clicked.connect(self.close)
        # Load the saved config file

        config.loadConfig()

        self.camera1URL.setText(State.config_dict['CAMERA_1'])
        self.camera2URL.setText(State.config_dict['CAMERA_2'])
        self.closeButton.clicked.connect(self.close)
        self.camera1URLButton.clicked.connect(self.loadFootageCamera1)
        self.setWindowTitle('Configuration')

    def loadFootageCamera1(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)


    
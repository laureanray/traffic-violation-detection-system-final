# Imports
import os
import cv2 as cv
import numpy as np
from widgets import Detection, Config
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow
import application as Application

class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        # Load GUI
        uic.loadUi('user_interface/MainWindow.ui', self)
        self.detection = Detection()
        self.configs = Config()
        self.startButton.clicked.connect(self.onStartButton)
        self.actionConfigs.triggered.connect(self.onTriggeredConfig)
        self.detection.show()

    def onStartButton(self):
        print('Start')
        if Application.State.isStarted == True:
            self.startButton.setText("Start")
            Application.State.isStarted = False
            self.detection.close()
        else:
            self.startButton.setText("Stop")
            Application.State.isStarted = True
            self.detection.show()

    def onTriggeredConfig(self):
        print('Configs')
        self.configs.show()

if __name__=='__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.setWindowTitle('Main')
    # window.show()
    sys.exit(app.exec_())
import os
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, uic

class video (QtWidgets.QDialog):
    def __init__(self):
        super(video, self).__init__()
        uic.loadUi('user_interface/Tset2.ui',self)
        self.startButton.clicked.connect(self.start_webcam)
        self.capture.clicked.connect(self.capture_image)
        self.imgLabel.setScaledContents(True)
        self.capture = None
        self.timer = QtCore.QTimer(self, interval=5)
        self.timer.timeout.connect(self.update_frame)
        self._image_counter = 0


if __name__=='__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = video()
    window.setWindowTitle('main code')
    window.show()
    sys.exit(app.exec_())
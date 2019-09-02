# Imports
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow

# Read the graph.
with tf.gfile.FastGFile('/home/lr/Desktop/tvds-integrated-final/object_detection/cars_inference_graph/frozen_inference_graph2.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    class Detection(QMainWindow):
        def __init__(self):
            super(Detection, self).__init__()
            # Load GUI
            uic.loadUi('./user_interface/DetectionView.ui', self)
            self.imageLabel.setScaledContents(True)
            self.capture = None
            self.setWindowTitle('Detection Window')
            self.timer = QtCore.QTimer(self, interval=5)
            self.timer.timeout.connect(self.update_frame)
            self.start_webcam()

        @QtCore.pyqtSlot()
        def start_webcam(self):
            if self.capture is None:
                self.capture = cv.VideoCapture("/home/lr/Desktop/tvds-integrated-final/vid.mp4", 0)
                self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
                self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            self.timer.start()

        @QtCore.pyqtSlot()
        def update_frame(self):
            ret, image = self.capture.read()
            simage = cv.flip(image, 1)
            self.displayImage(image, True)

        @QtCore.pyqtSlot()
        def capture_image(self):
            flag, frame = self.capture.read()
            path = r'J:\Face'
            if flag:
                QtWidgets.QApplication.beep()
                name = "opencv_frame_{}.png".format(self._image_counter)
                cv.imwrite(os.path.join(path, name), frame)
                self._image_counter += 1

        def displayImage(self, img, window=True):
            qformat = QtGui.QImage.Format_Indexed8
            if len(img.shape) == 3:
                if img.shape[2] == 4:
                    qformat = QtGui.QImage.Format_RGBA8888
                else:
                    qformat = QtGui.QImage.Format_RGB888
            outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
            outImage = outImage.rgbSwapped()
            if window:
                self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(outImage))

            # Uncommend this on deploy
            # self.showFullScreen()

            # self.show()

    # if __name__=='__main__':
    #     import sys
    #     app = QtWidgets.QApplication(sys.argv)
    #     window = Detection()
    #     window.setWindowTitle('Detection View')
    #
    #     sys.exit(app.exec_())
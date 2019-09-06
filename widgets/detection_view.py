# Imports
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow
import numpy as np
from matplotlib import pyplot as plt
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
            self.trafficLight.setScaledContents(True)
            self.traffic_light = None
            self.setWindowTitle('Detection Window')
            self.timer = QtCore.QTimer(self, interval=100)
            self.timer.timeout.connect(self.update_frame)
            self.start_webcam()
            self.stopButton.clicked.connect(self.stop)

        def stop(self):
            self.traffic_light.release()

        @QtCore.pyqtSlot()
        def start_webcam(self):
            if self.traffic_light is None:
                self.traffic_light = cv.VideoCapture("/home/lr/Desktop/traffic_light3.avi", 0)
                self.traffic_light.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
                self.traffic_light.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            self.timer.start()

        @QtCore.pyqtSlot()
        def update_frame(self):
            ret, image = self.traffic_light.read()
            resized = cv.resize(image, (1280, 720));
            x = 301*2;
            y = 182*2;
            w = 7*2;
            h = 16*2;
            roi = resized[y:y+h, x:x+w]
            # cv.rectangle(resized, (x, y), (x + w, y + h), (255, 255, 00), 2)
            # cv.imshow('roi', roi)
            simage = cv.flip(resized, 1)
            # print(resized.shape);
            # print(roi.shape);
            self.displayImage(resized, True, self.imageLabel)
            state = self.getTrafficLightStatus(roi)
            if state == "GREEN":
                self.trafficLightStatus.setText(state)
                self.trafficLightStatus.setStyleSheet('color: green')
            elif state == "AMBER":
                self.trafficLightStatus.setText(state)
                self.trafficLightStatus.setStyleSheet('color: yellow')
            elif state == "RED":
                self.trafficLightStatus.setText(state)
                self.trafficLightStatus.setStyleSheet('color: red')


            # self.displayImage(roi, True, self.trafficLight)

        @QtCore.pyqtSlot()
        def capture_image(self):
            flag, frame = self.capture.read()
            path = r'J:\Face'
            if flag:
                QtWidgets.QApplication.beep()
                name = "opencv_frame_{}.png".format(self._image_counter)
                cv.imwrite(os.path.join(path, name), frame)
                self._image_counter += 1

        def displayImage(self, img, window=True, label=None):
            qformat = QtGui.QImage.Format_Indexed8
            if len(img.shape) == 3:
                if img.shape[2] == 4:
                    qformat = QtGui.QImage.Format_RGBA8888
                else:
                    qformat = QtGui.QImage.Format_RGB888
            outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
            outImage = outImage.rgbSwapped()
            if window:
                label.setPixmap(QtGui.QPixmap.fromImage(outImage))

            # Uncomment this on deploy
            # self.showFullScreen()

            # self.show()

        def bincount_app(self, a):
            a2D = a.reshape(-1, a.shape[-1])
            col_range = (256, 256, 256)  # generically : a2D.max(0)+1
            a1D = np.ravel_multi_index(a2D.T, col_range)
            return np.unravel_index(np.bincount(a1D).argmax(), col_range)

        def getTrafficLightStatus(self, frame):
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            red = gray[6][6]
            red_full = frame[6][6]
            amber = gray[16][6]
            green = gray[25][6]
            state = ""
            if green > 80:
                state = "GREEN"
            elif amber > 80:
                state = "AMBER"
            elif red > 80 or red_full[2] > 170:
                state = "RED"
            return state


    # if __name__=='__main__':
    #     import sys
    #     app = QtWidgets.QApplication(sys.argv)
    #     window = Detection()
    #     window.setWindowTitle('Detection View')
    #
    #     sys.exit(app.exec_())
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
            self.trafficLight.setScaledContents(True)
            self.capture = None
            self.setWindowTitle('Detection Window')
            self.timer = QtCore.QTimer(self, interval=50)
            self.timer.timeout.connect(self.update_frame)
            self.start_webcam()

        @QtCore.pyqtSlot()
        def start_webcam(self):
            if self.capture is None:
                self.capture = cv.VideoCapture("/home/lr/Desktop/traffic_light.avi", 0)
                self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
                self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            self.timer.start()

        @QtCore.pyqtSlot()
        def update_frame(self):
            ret, image = self.capture.read()
            resized = cv.resize(image, (1280, 720));
            x = 299*2;
            y = 181*2;
            w = 10*2;
            h = 18*2;
            roi = resized[y:y+h, x:x+w]
            cv.rectangle(resized, (x, y), (x + w, y + h), (255, 255, 00), 2)
            # cv.imshow('roi', roi)
            simage = cv.flip(resized, 1)
            # print(resized.shape);
            # print(roi.shape);
            self.displayImage(resized, True, self.imageLabel)
            print(self.getTrafficLightStatus(roi))
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
        def getTrafficLightStatus(self, frame):
            # 20 x 36
            # Red ROI
            red_x = 0
            red_y = 0
            red_w = 20
            red_h = 12
            # Amber ROI
            amber_x = 0
            amber_y = 12
            amber_w = 20
            amber_h = 12
            # Green ROI
            green_x = 0
            green_y = 24
            green_w = 20
            green_h = 12
            cv.rectangle(frame, (red_x, red_y), (red_x + red_w, red_y + red_h), (0, 0, 255), 1)
            cv.rectangle(frame, (amber_x, amber_y), (amber_x + amber_w, amber_y + amber_h), (0, 255, 255), 1)
            cv.rectangle(frame, (green_x, green_y), (green_x + green_w, green_y + green_h), (0, 255, 0), 1)
            print(frame.shape)

            red = frame[red_y:red_y+red_h, red_x:red_x+red_w]

            print(red)
            # cv.waitKey(0)
            cv.imshow('frame', frame)
            return "Test"


    # if __name__=='__main__':
    #     import sys
    #     app = QtWidgets.QApplication(sys.argv)
    #     window = Detection()
    #     window.setWindowTitle('Detection View')
    #
    #     sys.exit(app.exec_())
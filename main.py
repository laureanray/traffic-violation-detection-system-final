# Imports
import os
import sys
import cv2 as cv
import numpy as np
import tensorflow as tf
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QMainWindow
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)
from application import State, Details
from widgets.config import Config
import config_manager as config

# Read the graph


# Read the graph.
with tf.gfile.FastGFile('/home/lr/Downloads/new-trained/car_inference_graph/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    application = QtWidgets.QApplication(sys.argv)

    sess = tf.Session()
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')


    class Main(QMainWindow):
        def __init__(self):
            super(Main, self).__init__()
            tf.import_graph_def(graph_def, name='')

            # Load GUI
            uic.loadUi('user_interface/main.ui', self)
            self.imageLabel.setScaledContents(True)
            self.trafficLight.setScaledContents(True)
            self.config = Config()
            self.config.setWindowModality(QtCore.Qt.ApplicationModal)
            self.config.setFixedSize(QSize(630, 470))
            self.traffic_light = None
            self.traffic_camera = None
            self.setWindowTitle('Detection Window')
            self.timer = QtCore.QTimer(self, intervagil=1)
            self.timer.timeout.connect(self.update_traffic_light_frame)
            self.timer.timeout.connect(self.update_traffic_frame)
            self.camera_selected = True

            # Load config before starting camera
            config.load_config()
            # Start camera
            self.start_camera()
            self.stopButton.clicked.connect(self.stop)
            self.changeCameraButton.clicked.connect(self.changeCamera)
            self.configButton.clicked.connect(self.showConfig)
            self.camera1_frame = None
            self.camera2_frame = None

        def stop(self):
            self.traffic_light.release()
            self.traffic_camera.release()
            sys.exit()

        @QtCore.pyqtSlot()
        def start_camera(self):

            if self.traffic_light is None:
                self.traffic_light = cv.VideoCapture(State.config_dict['CAMERA_1'], 0)
                self.traffic_light.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
                self.traffic_light.set(cv.CAP_PROP_FRAME_WIDTH, 640)

            if self.traffic_camera is None:
                self.traffic_camera = cv.VideoCapture(State.config_dict['CAMERA_2'], 0)
                self.traffic_camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
                self.traffic_camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

            self.timer.start()

        @QtCore.pyqtSlot()
        def update_traffic_light_frame(self):

            ret, frame = self.traffic_light.read()
            self.camera1_frame = cv.resize(frame, (1280, 720))
            x = 301 * 2
            y = 182 * 2
            w = 7 * 2
            h = 16 * 2
            roi = self.camera1_frame[y:y + h, x:x + w]

            displayImage(self.camera1_frame, True, self.imageLabel)
            state = getTrafficLightStatus(roi)
            if state == "GREEN":
                self.trafficLightStatus.setText(state)
                self.trafficLightStatus.setStyleSheet('color: green')
            elif state == "AMBER":
                self.trafficLightStatus.setText(state)
                self.trafficLightStatus.setStyleSheet('color: yellow')
            elif state == "RED":
                self.trafficLightStatus.setText(state)
                self.trafficLightStatus.setStyleSheet('color: red')

            image_ = cv.resize(roi, (64, 28))
            displayTraffic(image_, True, self.trafficLight)

        @QtCore.pyqtSlot()
        def update_traffic_frame(self):
            flag, self.camera2_frame = self.traffic_camera.read()
            resized = cv.resize(self.camera2_frame, (1280, 720))

            x = 301 * 2
            y = 182 * 2
            w = 7 * 2
            h = 16 * 2

            if self.camera_selected and sess:

                img = cv.resize(resized, (1280, 720))
                rows = img.shape[0]
                cols = img.shape[1]
                inp = cv.resize(img, (300, 300))
                inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

                # Run the model
                out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                                sess.graph.get_tensor_by_name('detection_scores:0'),
                                sess.graph.get_tensor_by_name('detection_boxes:0'),
                                sess.graph.get_tensor_by_name('detection_classes:0')],
                               feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

                # Visualize detected bounding boxes.
                num_detections = int(out[0][0])
                roi = []

                for i in range(num_detections):
                    classId = int(out[3][0][i])
                    score = float(out[1][0][i])

                    bbox = [float(v) for v in out[2][0][i]]
                    if score > 0.3:
                        x = bbox[1] * cols
                        y = bbox[0] * rows
                        right = bbox[3] * cols
                        bottom = bbox[2] * rows
                        print(classId)
                        if classId == 1:
                            print('Car')
                        elif classId == 2:
                            print('Truck')
                        # else:
                        #     print('Truck')

                        # print(score * 100)
                        cv.rectangle(img, (int(x), int(y)), (int(right),
                                                             int(bottom)), (125, 255, 51), thickness=2)

                        roi = img[int(y):int(y) + int(bottom), int(x):int(x) + int(right)]
                        print(int(right))
                displayImage(img, True, self.imageLabel)
            else:
                displayImage(self.camera1_frame, True, self.imageLabel)

        @QtCore.pyqtSlot()
        def capture_image(self):
            flag, frame = self.trafficLightStatus.read()
            path = r'J:\Face'
            if flag:
                QtWidgets.QApplication.beep()
                name = "opencv_frame_{}.png".format(self._image_counter)
                cv.imwrite(os.path.join(path, name), frame)
                self._image_counter += 1

        def changeCamera(self):
            self.start_camera()
            self.camera_selected = not self.camera_selected

        def showConfig(self):
            print('Show config')
            self.config.show()


# Static Methods


def getTrafficLightStatus(frame):
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


def displayImage(img, window=True, label=None):
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


def displayTraffic(img, window=True, label=None):
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


if __name__ == '__main__':
    import sys

    window = Main()
    window.setWindowTitle(Details.name)
    window.show()
    window.sess = sess
    window.graph_def = graph_def

    # window.showFullScreen()
    # window.show()
    sys.exit(application.exec_())

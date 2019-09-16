# Imports
import os
import sys
import cv2 as cv
import numpy as np
import tensorflow as tf
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QMainWindow
from object_tracker.pyimagesearch.centroidtracker import CentroidTracker
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)
from application import State, Details
from widgets.config import Config
import config_manager as config
import datetime
import time
from network import Net
import web.app as web
import threading, time
# Initialzie the object tracker


# Read the graph.
with tf.gfile.FastGFile('ml/car_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    application = QtWidgets.QApplication(sys.argv)

    sess = tf.Session()
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    ct = CentroidTracker()
    (H, W) = (None, None)


    class Main(QMainWindow):
        def __init__(self):
            super(Main, self).__init__()
            tf.import_graph_def(graph_def, name='')
            
            # Start the webserver first
            self.initializeWebServer()
            
            time.sleep(10)
            # Then start the socketio client to send realtime updates
            self.net = Net()

            # Load GUI
            uic.loadUi('user_interface/main.ui', self)
            # Initialize UI elements
            self.initUI()
          
            self.traffic_light = None
            self.traffic_camera = None

            # Initialize Timers
            self.initializeTimers()
         
            # State and Static Variables
            self.cameraSelected = True
            self.numOfCarsDetected = 0
            self.carsDetected.setText(str(self.numOfCarsDetected))
            self.H = None
            self.W = None
            self.clockTimer.start()
            self.frame = 0
            self.objectsToCount = []
            self.timeTracker = []

            # Load config before starting camera
            config.loadConfig()
            # Start camera
            self.start_camera()
            self.stopButton.clicked.connect(self.stop)
            self.changeCameraButton.clicked.connect(self.changeCamera)
            self.configButton.clicked.connect(self.showConfig)
            self.camera1_frame = None
            self.camera2_frame = None

            self.contours = np.array([[362, 85], [800, 85], [1280, 300], [1280, 720], [470, 720]])

        def initializeTimers(self):
            # This is responseible for the frame timing
            self.frameTimer = QtCore.QTimer(self, interval=1)
            # This updates the clock
            self.clockTimer = QtCore.QTimer(self, interval=1000)
            # This sets the http requests every 5 seconds
            # self.networkTimer = QtCore.QTimer(self, interval=5000)
            
            # Connect the function dependent on the timers
            self.frameTimer.timeout.connect(self.updateTrafficLightFrame)
            self.frameTimer.timeout.connect(self.updateTrafficFrame)
            self.clockTimer.timeout.connect(self.updateClock)
            # self.networkTimer.timeout.connect(self.sendUpdates)
            # self.networkTimer.start()
            

        @QtCore.pyqtSlot()
        def sendUpdates(self):
            self.net.sendRoutineUpdate(self.numOfCarsDetected);
            
        @QtCore.pyqtSlot()
        def initializeWebServer(self):
            print('starting server')
            web.runServerOnThread()
                        

        def initUI(self):
            self.imageLabel.setScaledContents(True)
            self.trafficLight.setScaledContents(True)
            self.config = Config()
            self.config.setWindowModality(QtCore.Qt.ApplicationModal)
            self.config.setFixedSize(QSize(630, 470))
            self.setWindowTitle('Detection Window')

        def stop(self):
            print('stop')
            self.traffic_light.release()
            self.traffic_camera.release()
            # Close the tf session
            net.closeConnection()
            sys.exit()
            
            try:
                sys.exit()
            except:
                print(sys.exc_info()[0])
                
            sess.close()

        @QtCore.pyqtSlot()
        def updateClock(self):
            now = datetime.datetime.now()
            self.day.setText(now.strftime("%A"))
            self.date.setText(now.strftime("%B %d, %Y"))
            self.time.setText(time.strftime("%H:%M:%S", time.localtime()))

        @QtCore.pyqtSlot()
        def start_camera(self):

            if self.traffic_light is None:
                self.traffic_light = cv.VideoCapture(State.config_dict['CAMERA_1'], 0)
                self.traffic_light.set(cv.CAP_PROP_FRAME_HEIGHT, 1920)
                self.traffic_light.set(cv.CAP_PROP_FRAME_WIDTH, 1080)

            if self.traffic_camera is None:
                self.traffic_camera = cv.VideoCapture(State.config_dict['CAMERA_2'], 0)
                self.traffic_camera.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
                self.traffic_camera.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

            self.frameTimer.start()

        @QtCore.pyqtSlot()
        def updateTrafficLightFrame(self):

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
        def updateTrafficFrame(self):
            self.frame += 1
            flag, self.camera2_frame = self.traffic_camera.read()

            if not flag:
                self.camera2_frame = cv.imread('standby.jpg', 0)
                displayImage(self.camera2_frame, True, self.imageLabel)
            else:
                resized = cv.resize(self.camera2_frame, (1280, 720))

                # if self.frame < 130:
                #     return
                # else:
                #     self.frameTimer.setInterval(250)

                roi = None

                if self.cameraSelected and sess:

                    img = cv.resize(resized, (1280, 720))
                    # Add the roi to the image
                    rows = img.shape[0]
                    cols = img.shape[1]
                    
                          # cv.fillPoly(img, pts=[self.contours], color=(255, 255, 255, 125))
                    # alpha = 0.4
                    # img = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                    # cv.imshow('img', img)

                    # 1
                    # bounding = cv.boundingRect(self.contours)
                    # x, y, w, h = bounding
                    # cropped = img[y:y+h, x:x+h].copy()
                    
                    img_copy = img.copy()
                    
                    # Make mask
                    # pts = self.contours - self.contours.min(axis=0)
                    
                    mask = np.zeros(img_copy.shape[:2], np.uint8)
                    cv.drawContours(mask, [self.contours], -1, (255, 255, 255), -1, cv.LINE_AA)
                    # do bit op
                    dst = cv.bitwise_and(img_copy, img_copy, mask=mask)
                    inp = cv.resize(dst, (300, 300))
                    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
                    overlay = img.copy()

                    if self.W is None or self.H is None:
                        (self.H, self.W) = img.shape[:2]

                    # Run the model
                    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                                    sess.graph.get_tensor_by_name('detection_scores:0'),
                                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                                    sess.graph.get_tensor_by_name('detection_classes:0')],
                                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

                    # Visualize detected bounding boxes.
                    num_detections = int(out[0][0])
                    rois = []
                    rects = []

                    # Loop over the detections
                    for i in range(num_detections):
                        classId = int(out[3][0][i])
                        score = float(out[1][0][i])

                        bbox = [float(v) for v in out[2][0][i]]
                        if score > 0.5:
                            x = bbox[1] * cols
                            y = bbox[0] * rows
                            right = bbox[3] * cols
                            bottom = bbox[2] * rows
                            rect = np.array([x, y, right, bottom])
                            rects.append(rect.astype("int"))
                            new_x = int(x * 1.5)
                            new_y = int(y * 1.5)
                            new_right = int(right * 1.5)
                            new_bottom = int(bottom * 1.5)
                            cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)),
                                         (0, 255, 0), 2)
                            
                            
                            roi = self.camera2_frame[new_y:new_y + (new_bottom - new_y), new_x:new_x + (new_right - new_x)]
                            rois.append(roi)
                            
                    
                    for x in range(len(rois)):
                        cv.imshow('roi', rois[x])
                        cv.waitKey(0)                           
                            
                    objects = ct.update(rects)
                                      
                    

                    for (objectID, centroid) in objects.items():
                        # object on the output frame
                        text = "ID {}".format(objectID)
                        cv.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                        # check if the car is already counted if not append to array
                        # print(centroid[1])
                        if centroid[1] >= 100:
                            if len(self.objectsToCount) > 0:
                                for x in range(len(self.objectsToCount)):
                                    # print('[x]: ' + str(x))
                                    # print(self.objectsToCount[x]['id'])
                                    # if objectID != self.objectsToCount[x]['id']:
                                    if not any(d['id'] == objectID for d in self.objectsToCount):
                                        # print(str(objectID) + ' not equal to ' + str(self.objectsToCount[x]['id']))
                                        dict_ = {'coords': centroid, 'is_counted': False, 'id': objectID}
                                        # Check if the objectid that the system is trying to add already in the self.objects_to_track list
                                        check = next((index for (index, d) in enumerate(self.objectsToCount) if d["id"] == objectID), None)
                                        # Check if the id has expired before adding to array
                                        index = next((index for (index, d) in enumerate(self.timeTracker) if d["id"] == objectID), None)
                                        if index is not None and check is not None:
                                            current_time = datetime.datetime.now()
                                            if current_time >= self.timeTracker[index]['time_exp']:
                                                # print(current_time.time() + ' vs ' + self.timeTracker[index]['time_exp'].time(), end=' ')
                                                self.objectsToCount.append(dict_)
                                                self.timeTracker.append({'id': objectID, 'time_exp': datetime.datetime.now() + datetime.timedelta(0,5)})
                                                # delete na yung tracker here?
                                                self.timeTracker.pop(index)
                                        else:
                                            # Dont check the time just add new one
                                            self.objectsToCount.append(dict_)
                                            self.timeTracker.append({'id': objectID, 'time_exp': datetime.datetime.now() + datetime.timedelta(0,5)})

                                    else:
                                        index = next((index for (index, d) in enumerate(self.objectsToCount) if d["id"] == objectID), None)
                                        if index is not None:
                                            self.objectsToCount[index]['coords'] = centroid
                            else:
                                dict_ = {'coords': centroid, 'is_counted': False, 'id': objectID}
                                self.objectsToCount.append(dict_)
                                self.timeTracker.append({'id': objectID, 'time_exp': datetime.datetime.now() + datetime.timedelta(0,5)})

                            cv.rectangle(img, (centroid[0] - 125, centroid[1] - 125), (centroid[0] + 125, centroid[1] + 125),
                                         (255, 255, 0), 1)

                    if len(self.objectsToCount) > 0:
                        for i in range(len(self.objectsToCount)):
                            if self.objectsToCount[i]['is_counted'] == False:
                                (x, y) = self.objectsToCount[i]['coords']
                                # print('y['+str(i)+']: ' + str(y))
                                if y < 200:
                                    self.numOfCarsDetected += 1
                                    self.objectsToCount[i]['is_counted'] = True
                                    
                                    print(self.objectsToCount[i]['coords'])
                                    
                                    index = next((index for (index, d) in enumerate(self.timeTracker) if d["id"] == self.objectsToCount[i]['id']), None)
                                    
                                    del self.timeTracker[index]
                                    
                                    self.sendUpdates()

                    img = cv.line(img, (372, 100), (861, 100), (255, 0, 255), 1)
                    img = cv.line(img, (372, 200), (861, 200), (255, 255, 0), 1)

                    self.carsDetected.setText(str(self.numOfCarsDetected))

                    
                    
                    
                    # # add white background
                    # bg = np.ones_like(img_copy, np.uint8) * 255
                    # cv.bitwise_not(bg, bg, mask=mask)
                    
                    # dst2 = bg + dst
                    
                    cv.imshow('dst2', dst)
                    # cv.waitKey(0)
                    displayImage(img, True, self.imageLabel)
                    cv.imshow('img', img)
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
            self.cameraSelected = not self.cameraSelected

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

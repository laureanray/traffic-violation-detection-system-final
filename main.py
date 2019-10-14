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
from web.app import add_violation
import threading, time
from plate_number import run
from multiprocessing import Process 
import json
# from alpr import Alpr
# Initialzie the object tracker


# # Read the plate number localization graph
# f =  tf.gfile.FastGFile('ml/plate_number_inference_graph.pb', 'rb') 
# pn_graph_def = tf.GraphDef()
# pn_graph_def.ParseFromString(f.read())

    
    

# Read the graph.
with tf.gfile.FastGFile('ml/car_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    application = QtWidgets.QApplication(sys.argv)

    sess = tf.Session()
    sess.graph.as_default()
    # sess.pn_graph.as_default()
    # tf.import_graph_def(pn_graph_def, name='')
    tf.import_graph_def(graph_def, name='')

    ct = CentroidTracker()
    (H, W) = (None, None)


    class Main(QMainWindow):
        def __init__(self):
            super(Main, self).__init__()
            tf.import_graph_def(graph_def, name='')
            # Start the webserver first
            self.initializeWebServer()
            # This gives the server time start before actually trying to connect to it.
            time.sleep(2)
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
            self.numOfVehiclesViolated = 0
            self.carsDetected.setText(str(self.numOfCarsDetected))
            self.H = None
            self.W = None
            self.clockTimer.start()
            self.frame = 0
            self.objectsToCount = []
            self.timeTracker = []
            self.objectsToDetectViolation = []
            self.timeTracker2 = []
            # self.alpr = Alpr("us", "/etc/openalpr/openalpr.conf", )
            # Load config before starting camera
            config.loadConfig()

            # Start camera
            self.start_camera()
            self.stopButton.clicked.connect(self.stop)
            self.changeCameraButton.clicked.connect(self.changeCamera)
            self.configButton.clicked.connect(self.showConfig)
            self.camera1_frame = None
            self.camera2_frame = None
            
            self.contours = np.array([[362, 20], [800, 20], [1280, 300], [1280, 720], [470, 720]], np.int32)
            self.yellow_box = np.array([[365, 86], [601, 86], [900, 210], [394, 210]], np.int32)
            self.plate_number_process = Process(target = self.plateDetectionAndOcr)
            self.trafficLightState = ""
            


        def plateDetectionAndOcr(self, image):
            plate = run(image)
            cv.imshow('plate', plate)
            cv.waitKey(1000)
            return plate

        def initializeTimers(self):
            # This is responseible for the frame timing
            self.frameTimer = QtCore.QTimer(self, interval=25)
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
            self.net.sendRoutineUpdate(self.numOfCarsDetected, self.numOfVehiclesViolated);
            
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
            # Release the source first (ip cam or footage)
            self.traffic_light.release()
            self.traffic_camera.release()
            # Close the tf session
            sess.close()

            # Close the http client connection
            self.net.closeConnection()
            
            # Shutdown thewebserver
            web.shutdownServerOnThread()

            # Exit Gracefully
            sys.exit()

            try:
                sys.exit()
            except:
                print(sys.exc_info()[0])
                

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

            if not ret:
                frame = cv.imread('standby.jpg', 0)
                displayImage(frame, True, self.imageLabel)
            else:

                resized = cv.resize(self.camera2_frame, (1280, 720))    
                self.camera1_frame = cv.resize(frame, (1280, 720))

                x = 301 * 2
                y = 174 * 2
                w = 7 * 2
                h = 16 * 2
                roi = self.camera1_frame[y:y + h, x:x + w]

                displayImage(self.camera1_frame, True, self.imageLabel)
                self.trafficLightState = getTrafficLightStatus(roi)
                if self.trafficLightState == "GREEN":
                    self.trafficLightStatus.setText(self.trafficLightState)
                    self.trafficLightStatus.setStyleSheet('color: green')
                elif self.trafficLightState == "AMBER":
                    self.trafficLightStatus.setText(self.trafficLightState)
                    self.trafficLightStatus.setStyleSheet('color: yellow')
                elif self.trafficLightState == "RED":
                    self.trafficLightStatus.setText(self.trafficLightState)
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

                    # plate_number = sess.run([sess.pn_graph.get_tensor_by_name('num_detections:0'),
                    #                 sess.graph.get_tensor_by_name('detection_scores:0'),
                    #                 sess.graph.get_tensor_by_name('detection_boxes:0'),
                    #                 sess.graph.get_tensor_by_name('detection_classes:0')],
                    #                 feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

                    # Visualize detected bounding boxes.
                    num_detections = int(out[0][0])
                    rois = []
                    rects = []

                    # Loop over the detections
                    for i in range(num_detections):
                        classId = int(out[3][0][i])
                        score = float(out[1][0][i])

                        bbox = [float(v) for v in out[2][0][i]]
                        if score > 0.2:
                            x = bbox[1] * cols
                            y = bbox[0] * rows
                            right = bbox[3] * cols
                            bottom = bbox[2] * rows
  
                            new_x = int(x * 1.5)
                            new_y = int(y * 1.5)
                            new_right = int(right * 1.5)
                            new_bottom = int(bottom * 1.5)
                            # Set width threshold
                            if (int(right) - int(x) <= 450):
                                
                                if(int((y+bottom)/2) >= 201):    
                                    cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)),
                                                    (0, 255, 0), 2)
                                else:
                                    cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)),
                                            (0, 0, 255), 2)
                                
                                roi = self.camera2_frame[new_y:new_y + (new_bottom - new_y), new_x:new_x + (new_right - new_x)]
                                rois.append(roi)
                                rect = np.array([x, y, right, bottom])
                                rects.append(rect.astype("int"))
                               
                    # Run the plate number detection in the rois of the frame
                    # for x in range(len(rois)):
                    #     cv.imshow('roi', rois[x])
                    #     cv.waitKey(0)                           
                            
                    objects = ct.update(rects)
                                      
                    for (objectID, centroid) in objects.items():
                        # object on the output frame
                        text = "ID {}".format(objectID)
                        cv.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                        # check if the car is already counted if not append to array
                        # print(centroid[1])
                        if centroid[1] >= 200:
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
                                                self.timeTracker.append({'id': objectID, 'time_exp': datetime.datetime.now() + datetime.timedelta(0,1)})
                                                # delete na yung tracker here?
                                                self.timeTracker.pop(index)
                                        else:
                                            # Dont check the time just add new one
                                            self.objectsToCount.append(dict_)
                                            self.timeTracker.append({'id': objectID, 'time_exp': datetime.datetime.now() + datetime.timedelta(0,1)})

                                    else:
                                        index = next((index for (index, d) in enumerate(self.objectsToCount) if d["id"] == objectID), None)
                                        if index is not None:
                                            self.objectsToCount[index]['coords'] = centroid
                            else:
                                dict_ = {'coords': centroid, 'is_counted': False, 'id': objectID}
                                self.objectsToCount.append(dict_)
                                self.timeTracker.append({'id': objectID, 'time_exp': datetime.datetime.now() + datetime.timedelta(0,1)})
                        else:
                            if self.trafficLightState == 'RED':
                                # If nag violate
                                if len(self.objectsToDetectViolation) > 0:
                                    for x in range(len(self.objectsToDetectViolation)):
                                        # print('[x]: ' + str(x))
                                        # print(self.objectsToCount[x]['id'])
                                        # if objectID != self.objectsToCount[x]['id']:
                                        if not any(d['id'] == objectID for d in self.objectsToDetectViolation):
                                            # print(str(objectID) + ' not equal to ' + str(self.objectsToDetectViolation[x]['id']))
                                            
                                            # index = next((index for (index, d) in enumerate(self.objectsToDetectViolation) if d["coords"][0] == ()), None)

                                            bbox = []

                                            for x, y, endx, endy in rects:
                                                centroidx = int((x + endx) / 2)
                                                
                                                if centroidx == centroid[0]:
                                                    bbox = [x, y, endx, endy]

                                            dict_ = {'coords': centroid, 'violated': False, 'id': objectID, 'bbox': bbox }
                                            # Check if the objectid that the system is trying to add already in the self.objects_to_track list
                                            check = next((index for (index, d) in enumerate(self.objectsToDetectViolation) if d["id"] == objectID), None)
                                            # Check if the id has expired before adding to array
                                            index = next((index for (index, d) in enumerate(self.timeTracker2) if d["id"] == objectID), None)
                                            if index is not None and check is not None:
                                                current_time = datetime.datetime.now()
                                                if current_time >= self.timeTracker[index]['time_exp']:
                                                    # print(current_time.time() + ' vs ' + self.timeTracker[index]['time_exp'].time(), end=' ')
                                                    self.objectsToDetectViolation.append(dict_)
                                                    self.timeTracker2.append({'id': objectID, 'time_exp': datetime.datetime.now() + datetime.timedelta(0,5)})
                                                    # delete na yung tracker here?
                                                    self.timeTracker2.pop(index)
                                            else:
                                                # Dont check the time just add new one
                                                self.objectsToDetectViolation.append(dict_)
                                                self.timeTracker2.append({'id': objectID, 'time_exp': datetime.datetime.now() + datetime.timedelta(0,5)})

                                        else:
                                            index = next((index for (index, d) in enumerate(self.objectsToDetectViolation) if d["id"] == objectID), None)
                                            if index is not None:
                                                self.objectsToDetectViolation[index]['coords'] = centroid
                                else:
                                    bbox = []

                                    for x, y, endx, endy in rects:
                                        centroidx = int((x + endx) / 2)
                                        
                                        if centroidx == centroid[0]:
                                            bbox = [x, y, endx, endy]

                                    dict_ = {'coords': centroid, 'violated': False, 'id': objectID, 'bbox': bbox}
                                    self.objectsToDetectViolation.append(dict_)
                                    self.timeTracker2.append({'id': objectID, 'time_exp': datetime.datetime.now() + datetime.timedelta(0,1)})
                                # print('ok')

                    if len(self.objectsToCount) > 0:
                        for i in range(len(self.objectsToDetectViolation)):
                            if self.objectsToDetectViolation[i]['violated'] == False:
                                (x, y) = self.objectsToDetectViolation[i]['coords']
                                
                                if y < 200:
                                    # Violated
                                    self.objectsToDetectViolation[i]['violated'] = True
                                    self.numOfVehiclesViolated += 1
                                    self.violationsDetected.setText(str(self.numOfVehiclesViolated))
                                    index = next((index for (index, d) in enumerate(self.timeTracker2) if d["id"] == self.objectsToDetectViolation[i]['id']), None)
                                    del self.timeTracker2[index]
                                    
                                    print('BEAT')
                                    x, y, endx, endy = self.objectsToDetectViolation[i]['bbox']
                                    

                                    data = {
                                        'violation_type': 'BEATING THE RED LIGHT',
                                        'vehicle_type': 'Car',
                                        'plate_number': 'plate',
                                        'plate_number_img_url': '/data/asdas',
                                        'vehicle_img_url': '/adasda'
                                    }
                                    
                                    # Detect platenumber\
                                    car_roi = self.camera2_frame[int(y * 1.5):int(endy * 1.5), int(x * 1.5):int(endx * 1.5)]
                                        
                                    plate_roi = self.plateDetectionAndOcr(car_roi)

                                    add_violation(data)
                                    self.net.sendRoutineUpdate(self.numOfCarsDetected, self.numOfVehiclesViolated)

                                    # cv.imshow('roasdasdasd', roi)
                                    time_str = str(time.time())
                                    cv.imwrite('/home/lr/Desktop/tvds-integrated-final/data/red_light/car' +  time_str + '.jpg', car_roi)
                                    cv.imwrite('/home/lr/Desktop/tvds-integrated-final/data/red_light/plate_' +   time_str + '.jpg', plate_roi)

                                   
                    
                    if len(self.objectsToCount) > 0:
                        for i in range(len(self.objectsToCount)):
                            if self.objectsToCount[i]['is_counted'] == False:
                                (x, y) = self.objectsToCount[i]['coords']
                                # print('y['+str(i)+']: ' + str(y))
                                if y < 300:
                                    self.numOfCarsDetected += 1
                                    self.objectsToCount[i]['is_counted'] = True
                                    index = next((index for (index, d) in enumerate(self.timeTracker) if d["id"] == self.objectsToCount[i]['id']), None)
                                    del self.timeTracker[index]
                                    self.sendUpdates()

                    img = cv.line(img, (372, 200), (861, 200), (255, 0, 255), 1)
                    img = cv.line(img, (372, 300), (861, 300), (255, 255, 0), 1)

                    self.carsDetected.setText(str(self.numOfCarsDetected))

                    cv.polylines(img, [self.yellow_box], True, (255, 125, 125))
                    displayImage(img, True, self.imageLabel)
                    # cv.imshow('img', img)
                    # cv.waitKey(1)
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
            
        @QtCore.pyqtSlot()
        def detectPlateNumber(self, frame):
            frame_process = frame.copy()
            # Read the graph.
            with tf.gfile.FastGFile('ml/plate_number_inference_graph.pb', 'rb') as f:
                graph_def1 = tf.GraphDef()
                graph_def1.ParseFromString(f.read())
                
                application = QtWidgets.QApplication(sys.argv)

                with tf.Session() as sess:
                    sess = tf.Session()
                    sess.graph.as_default()
                    
                    rows = frame.shape[0]
                    cols = frame.shape[1]
                        
                    # sess.pn_graph.as_default()
                    # tf.import_graph_def(pn_graph_def, name='')
                    tf.import_graph_def(graph_def1, name='')
                
                    inp = cv.resize(frame_process, (300, 300))
                    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
                    # Run the model
                    plate_number = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                                    sess.graph.get_tensor_by_name('detection_scores:0'),
                                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                                    sess.graph.get_tensor_by_name('detection_classes:0')],
                                    feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

                    # Visualize detected bounding boxes.
                    num_detections = int(plate_number[0][0])
                    rois = []
                    rects = []
                    
                    # Loop over the detections
                    for i in range(num_detections):
                        classId = int(plate_number[3][0][i])
                        score = float(plate_number[1][0][i])

                        bbox = [float(v) for v in plate_number[2][0][i]]
                        
                        if score > 0.2:
                            x = bbox[1] * cols
                            y = bbox[0] * rows
                            right = bbox[3] * cols
                            bottom = bbox[2] * rows

                            new_x = int(x * 1.5)
                            new_y = int(y * 1.5)
                            new_right = int(right * 1.5)
                            new_bottom = int(bottom * 1.5)
                            # Set width threshold
                            cv.rectangle(frame_process, (int(x), int(y)), (int(right), int(bottom)),
                                            (0, 255, 0), 2)
                        
                    sess.close()
                    return frame_process
                            
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
    window.setFixedSize(1366, 768)
    # window.show()
    window.sess = sess
    window.graph_def = graph_def
    window.showFullScreen()
    # window.show()
    sys.exit(application.exec_())

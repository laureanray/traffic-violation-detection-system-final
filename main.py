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

# Initialzie the object tracker


# Read the graph.
with tf.gfile.FastGFile('/home/lr/Downloads/new-trained/car_inference_graph/frozen_inference_graph.pb', 'rb') as f:
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
            self.timer = QtCore.QTimer(self, interval=1)
            self.clock_timer = QtCore.QTimer(self, interval=1000)
            self.timer.timeout.connect(self.update_traffic_light_frame)
            self.timer.timeout.connect(self.update_traffic_frame)
            self.clock_timer.timeout.connect(self.update_clock)
            self.camera_selected = True
            self.num_cars_detected = 0
            self.carsDetected.setText(str(self.num_cars_detected))
            self.H = None
            self.W = None
            self.clock_timer.start()
            self.frame = 0
            self.objects_to_count = []
            self.time_tracker = []

            # Load config before starting camera
            config.load_config()
            # Start camera
            self.start_camera()
            self.stopButton.clicked.connect(self.stop)
            self.changeCameraButton.clicked.connect(self.changeCamera)
            self.configButton.clicked.connect(self.showConfig)
            self.camera1_frame = None
            self.camera2_frame = None

            self.contours = np.array([[362, 85], [680, 85], [1280, 368], [1280, 720], [470, 720]])

        def stop(self):
            self.traffic_light.release()
            self.traffic_camera.release()
            # Close the tf session
            sess.close()
            sys.exit()

        @QtCore.pyqtSlot()
        def update_clock(self):
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
            # print('frame: ' + str(self.frame))
            self.frame += 1
            
            # print('###')
            # print(len(ct.disappeared))
            # print('###')
            
            
            flag, self.camera2_frame = self.traffic_camera.read()
            resized = cv.resize(self.camera2_frame, (1280, 720))

            if self.frame < 130:
                return
            else:
                self.timer.setInterval(250)
            
            roi = None

            if self.camera_selected and sess:
                
                img = cv.resize(resized, (1280, 720))
                # Add the roi to the image

                rows = img.shape[0]
                cols = img.shape[1]
                inp = cv.resize(img, (300, 300))
                inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

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
                roi = []
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
                        # print(rect)
                        # print(classId)  # print(rect)
                        # print(classId)
                        # if classId == 1:
                        # print('Car')
                        # elif classId == 2:
                        # print('Truck')
                        # else:
                        #     print('Truck')

                        new_x = int(x * 1.5)
                        new_y = int(y * 1.5)
                        new_right = int(right * 1.5)
                        new_bottom = int(bottom * 1.5)

                        cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)),
                                     (0, 255, 0), 2)

                        roi = self.camera2_frame[new_y:new_y + (new_bottom - new_y), new_x:new_x + (new_right - new_x)]

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
                        if len(self.objects_to_count) > 0:
                            for x in range(len(self.objects_to_count)):
                                # print('[x]: ' + str(x))
                                # print(self.objects_to_count[x]['id'])
                                # if objectID != self.objects_to_count[x]['id']:
                                if not any(d['id'] == objectID for d in self.objects_to_count):
                                    # print(str(objectID) + ' not equal to ' + str(self.objects_to_count[x]['id']))
                                    dict_ = {'coords': centroid, 'is_counted': False, 'id': objectID}
                                    
                                    # Check if the objectid that the system is trying to add already in the self.objects_to_track list
                                    check = next((index for (index, d) in enumerate(self.objects_to_count) if d["id"] == objectID), None)
                                    
                                    # Check if the id has expired before adding to array
                                    index = next((index for (index, d) in enumerate(self.time_tracker) if d["id"] == objectID), None)

                                    if index is not None and check is not None:
                                        current_time = datetime.datetime.now()
                                        
                                        if current_time >= self.time_tracker[index]['time_exp']:
                                            # print(current_time.time() + ' vs ' + self.time_tracker[index]['time_exp'].time(), end=' ')
                                            self.objects_to_count.append(dict_)
                                            self.time_tracker.append({'id': objectID, 'time_exp': datetime.datetime.now() + datetime.timedelta(0,5)})
                                            # delete na yung tracker here?
                                            self.time_tracker.pop(index)                                            
                                        # else:
                                            # print('DI PWEDE')
                                    else:
                                        # print("ELSA???")
                                        # print(self.objects_to_count)
                                        # print(index, check, end=' ')
                                        # Dont check the time just add new one
                                        self.objects_to_count.append(dict_)
                                        self.time_tracker.append({'id': objectID, 'time_exp': datetime.datetime.now() + datetime.timedelta(0,5)})
                                            
                                else:
                                    # print('ELSE')
                                    index = next((index for (index, d) in enumerate(self.objects_to_count) if d["id"] == objectID), None)
                                    #print(index)
                                    if index is not None:
                                        self.objects_to_count[index]['coords'] = centroid      
                                        print('UPDATE CENTROID')                  
                                        
                        else:
                            dict_ = {'coords': centroid, 'is_counted': False, 'id': objectID}
                            self.objects_to_count.append(dict_)    
                            self.time_tracker.append({'id': objectID, 'time_exp': datetime.datetime.now() + datetime.timedelta(0,5)})
      
                        cv.rectangle(img, (centroid[0] - 50, centroid[1] - 50), (centroid[0] + 50, centroid[1] + 50),
                                     (255, 255, 0), 1)

                # if len(roi) > 0:
                #     cv.imshow('roi', cv.resize(roi, (500, 250)))

                # print(len(self.objects_to_count))
                # print(len(self.time_tracker))          
                
                # for x in range(len(self.time_tracker)):
                #     print(self.time_tracker[x]['time_exp'].time(), end=' ')      
                    
                
                if len(self.objects_to_count) > 0:
                    for i in range(len(self.objects_to_count)):
                        if self.objects_to_count[i]['is_counted'] == False:
                            (x, y) = self.objects_to_count[i]['coords']
                            # print('y['+str(i)+']: ' + str(y))
                            if y < 200:
                                self.num_cars_detected += 1
                                self.objects_to_count[i]['is_counted'] = True
                                index = next((index for (index, d) in enumerate(self.time_tracker) if d["id"] == self.objects_to_count[i]['id']), None)
                                del self.time_tracker[index]
                                # del self.objects_to_count[i]
                                
                
                
               # print('o: ' + str(len(self.objects_to_count)) + 't: ' + str(len(self.time_tracker)))
                
                
                # objects_to_count = objects      f
                # if len(self.objects_to_count) > 0:
                #     for i in range(len(self.objects_to_count)):
                #         print(self.objects_to_count[i])
                        # print(str(int(i)) + ' asdas')
                        # print(objects_to_count[i])
                        # if objects_to_count[i]['is_counted'] == False:
                        #     (x, y) = objects_to_count[i]['coords']
                            
                        #     if y < 200:
                        #         self.num_cars_detected += 1
                        #         objects_to_count[i]['is_counted'] = True

                        #     (xstart, ystart, xend, yend)= objects_to_count[i]['coords']
                            
                        #     print(ystart, yend)
                            
                            
                        #     if 200 in range(ystart, yend):
                        #         print('inrange')
                        #         cv.waitKey(0)
                        #         self.num_cars_detected += 1
                        #         # objects_to_count[i]['is_counted'] = True
                        #         objects_to_count.pop(i)
                        #         break
                            
                            # if objects_to_count[i]['centroid'][1] <= 200:
                            #     # count
                            #     self.num_cars_detected += 1
                            #     print(self.num_cars_detected)
                            #     objects_to_count[i]['is_counted'] = True

                # Purge counted centroids
                # print(objects_to_count)
                # for i in range((len(objects_to_count))):
                #     if objects_to_count[i]['is_counted'] == True:
                #         elements_to_pop.append(i)
                     
   
                # print(elements_to_pop)

                # for x in range(int(len(elements_to_pop))):
                #     objects_to_count.pop(int(x))
                #     print(str(int(x)) + ' popped')

                # img = cv.line(img, (372, 200), (861, 200), (255, 0, 0,), 2)
                img = cv.line(img, (372, 100), (861, 100), (255, 0, 255), 1)
                img = cv.line(img, (372, 200), (861, 200), (255, 255, 0), 1)
                


                self.carsDetected.setText(str(self.num_cars_detected))
                # cv.waitKey(0)

                # overlay = im/g.copy()
                # cv.fillPoly(img, pts=[self.contours], color=(255, 255, 255, 125))
                # alpha = 0.4
                # img = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                # cv.imshow('img', img)

                # 1
                # bounding = cv.boundingRect(self.contours)
                # x, y, w, h = bounding
                # cropped = img[y:y+h, x:x+h].copy()
                #
                # # Make mask
                # pts = self.contours - self.contours.min(axis=0)
                #
                # mask = np.zeros(cropped.shape[:2], np.uint8)
                # cv.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv.LINE_AA)
                #
                # # do bit op
                # dst = cv.bitwise_and(cropped, cropped, mask=mask)
                #
                #
                # # add white background
                # bg = np.ones_like(cropped, np.uint8) * 255
                # cv.bitwise_not(bg, bg, mask=mask)
                #
                # dst2 = bg + dst
                #
                # cv.imshow('dst2', dst)
                #

                # cv.waitKey(0)
                displayImage(img, True, self.imageLabel)
                # cv.imshow('img', img)
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

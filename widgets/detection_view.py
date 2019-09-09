# Imports
import os
import sys
import cv2 as cv
import numpy as np
import tensorflow as tf
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)

import tarfile
import zipfile
import matplotlib
import six.moves.urllib as urllib
from matplotlib import pyplot as plt

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops

# MODEL_NAME = 'cars_inference_graph'
#
# # Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph2.pb'
#
# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'label_map.pbtxt')
#

# from object_detection.utils import label_map_util

# from object_detection.utils import visualization_utils as vis_util
#

# detection_graph = tf.Graph()

# with detection_graph.as_default():
#   od_graph_def = tf.GraphDef()
#   with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
#     serialized_graph = fid.read()
#     od_graph_def.ParseFromString(serialized_graph)
#     tf.import_graph_def(od_graph_def, name='')
#
# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



with tf.Session() as sess:
    class Detection(QMainWindow):
        def __init__(self):
            super(Detection, self).__init__()
            # Load GUI
            uic.loadUi('./user_interface/DetectionView.ui', self)
            self.imageLabel.setScaledContents(True)
            self.trafficLight.setScaledContents(True)
            self.traffic_light = None
            self.traffic_camera = None
            self.setWindowTitle('Detection Window')
            self.timer = QtCore.QTimer(self, interval=100)
            self.timer.timeout.connect(self.update_traffic_light_frame)
            self.timer.timeout.connect(self.update_traffic_frame)
            self.camera_selected = True
            # Start camera
            self.start_camera()
            self.stopButton.clicked.connect(self.stop)
            self.changeCameraButton.clicked.connect(self.changeCamera)
            self.camera1_frame = None
            self.camera2_frame = None

        def stop(self):
            self.traffic_light.release()
            self.traffic_camera.release()
            sys.exit()

        @QtCore.pyqtSlot()
        def start_camera(self):

            if self.traffic_light is None:
                self.traffic_light = cv.VideoCapture("/home/lr/Desktop/traffic_light2.avi", 0)
                self.traffic_light.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
                self.traffic_light.set(cv.CAP_PROP_FRAME_WIDTH, 640)

            if self.traffic_camera is None:
                self.traffic_camera = cv.VideoCapture("/home/lr/Desktop/car_1.avi", 0)
                self.traffic_camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
                self.traffic_camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

            self.timer.start()

        @QtCore.pyqtSlot()
        def update_traffic_light_frame(self):

            ret, frame = self.traffic_light.read()
            self.camera1_frame = cv.resize(frame, (1280, 720))
            x = 301*2
            y = 182*2
            w = 7*2
            h = 16*2
            roi = self.camera1_frame[y:y+h, x:x+w]

            self.displayImage(self.camera1_frame, True, self.imageLabel)
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

            image_ = cv.resize(roi, (64, 28))
            self.displayTraffic(image_, True, self.trafficLight)

        @QtCore.pyqtSlot()
        def update_traffic_frame(self):
            flag, self.camera2_frame = self.traffic_camera.read()
            resized = cv.resize(self.camera2_frame, (1280, 720))

            x = 301 * 2
            y = 182 * 2
            w = 7 * 2
            h = 16 * 2

            if self.camera_selected:
                self.displayImage(resized, True, self.imageLabel)
            else:
                self.displayImage(self.camera1_frame, True, self.imageLabel)


        @QtCore.pyqtSlot()
        def capture_image(self):
            flag, frame = self.trafficLightStatus.read()
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

        def displayTraffic(self, img, window=True, label=None):
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

        def changeCamera(self):
            self.start_camera()
            self.camera_selected = not self.camera_selected

        # def run_inference_for_single_image(self, image, graph):
        #     with graph.as_default():
        #         with tf.Session() as sess:
        #             # Get handles to input and output tensors
        #             ops = tf.get_default_graph().get_operations()
        #             all_tensor_names = {output.name for op in ops for output in op.outputs}
        #             tensor_dict = {}
        #             for key in [
        #                 'num_detections', 'detection_boxes', 'detection_scores',
        #                 'detection_classes', 'detection_masks'
        #             ]:
        #                 tensor_name = key + ':0'
        #                 if tensor_name in all_tensor_names:
        #                     tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
        #                         tensor_name)
        #             if 'detection_masks' in tensor_dict:
        #                 # The following processing is only for single image
        #                 detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        #                 detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        #                 # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        #                 real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        #                 detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        #                 detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        #                 detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        #                     detection_masks, detection_boxes, image.shape[1], image.shape[2])
        #                 detection_masks_reframed = tf.cast(
        #                     tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        #                 # Follow the convention by adding back the batch dimension
        #                 tensor_dict['detection_masks'] = tf.expand_dims(
        #                     detection_masks_reframed, 0)
        #             image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        #
        #             # Run inference
        #             output_dict = sess.run(tensor_dict,
        #                                    feed_dict={image_tensor: image})
        #
        #             # all outputs are float32 numpy arrays, so convert types as appropriate
        #             output_dict['num_detections'] = int(output_dict['num_detections'][0])
        #             output_dict['detection_classes'] = output_dict[
        #                 'detection_classes'][0].astype(np.int64)
        #             output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        #             output_dict['detection_scores'] = output_dict['detection_scores'][0]
        #             if 'detection_masks' in output_dict:
        #                 output_dict['detection_masks'] = output_dict['detection_masks'][0]
        #     return output_dict
        #
        #     def run_inference(self):
        #         image = Image.open(image_path)
        #         # the array based representation of the image will be used later in order to prepare the
        #         # result image with boxes and labels on it.
        #         image_np = load_image_into_numpy_array(image)
        #         # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        #         image_np_expanded = np.expand_dims(image_np, axis=0)
        #         # Actual detection.
        #         output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
        #         # Visualization of the results of a detection.
        #         vis_util.visualize_boxes_and_labels_on_image_array(
        #             image_np,
        #             output_dict['detection_boxes'],
        #             output_dict['detection_classes'],
        #             output_dict['detection_scores'],
        #             category_index,
        #             instance_masks=output_dict.get('detection_masks'),
        #             use_normalized_coordinates=True,
        #             line_thickness=8)
                # plt.figure(figsize=IMAGE_SIZE)
                # plt.imshow(image_np)
                # plt.show()

    # if __name__=='__main__':
    #     import sys
    #     app = QtWidgets.QApplication(sys.argv)
    #     window = Detection()
    #     window.setWindowTitle('Detection View')
    #
    #     sys.exit(app.exec_())
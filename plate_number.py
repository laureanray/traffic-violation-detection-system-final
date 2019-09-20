import numpy as np
import tensorflow as tf
import cv2 as cv
from PIL import Image
from utils import visualization_utils as vis_util
from utils import label_map_util

category_index = label_map_util.create_category_index_from_labelmap('/home/lr/plate.pbtxt', use_display_name=True)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile('/home/lr/Downloads/test/platenumber_inference/plate_number_inference_graph.pb', 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
  
  
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def run(image_np):
    # image = Image.open('/home/lr/Downloads/test/platenumber_inference/xz.png').convert('RGB')
    # image_np = load_image_into_numpy_array(image)
    # image_np = cv.imread('/home/lr/Downloads/test/platenumber_inference/xz.png')
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)

    cv.imshow('out', image_np)
    cv.waitKey(0)
    
    
# class PlateDetector:
#     def __init__(self):
#         self.graph = tf.Graph()
#         self.session = tf.Session(graph=self.graph)
        
#         self.f = tf.gfile.FastGFile('/home/lr/Downloads/test/platenumber_inference/plate_number_inference_graph.pb', 'rb')
#         self.graph_def = tf.GraphDef()
#         self.graph_def.ParseFromString(self.f.read())
        
#         self.session.graph.as_default()
#         tf.import_graph_def(self.graph_def, name='')
            
#     def run(self, frame):
#         img = cv.resize(frame, (1280, 720))
#         rows = img.shape[0]
#         cols = img.shape[1]
#         inp = cv.resize(img, (300, 300))
#         inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

#         print(self.session.graph)

#         out = self.session.run([self.session.graph.get_tensor_by_name('num_detections:0'),
#                         self.session.graph.get_tensor_by_name('detection_scores:0'),
#                         self.session.graph.get_tensor_by_name('detection_boxes:0'),
#                         self.session.graph.get_tensor_by_name('detection_classes:0')],
#                     feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

#         # Visualize detected bounding boxes.
#         num_detections = int(out[0][0])
#         roi = []

#         for i in range(num_detections):
#             classId = int(out[3][0][i])
#             score = float(out[1][0][i])
            
#             bbox = [float(v) for v in out[2][0][i]]
#             if score > 0.3:
#                 x = bbox[1] * cols
#                 y = bbox[0] * rows
#                 right = bbox[3] * cols
#                 bottom = bbox[2] * rows
#                 print(classId)
                
#                 # else:
#                 #     print('Truck')
                
#                 # print(score * 100)
#                 cv.rectangle(img, (int(x), int(y)), (int(right),
#                                                     int(bottom)), (125, 255, 51), thickness=2)
                
                
#                 roi = img[int(y):int(y)+(int(bottom) - int(y)), int(x):int(x)+(int(right) - int(x))]

#                 if roi.any():
#                     roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
#                     blur = cv.GaussianBlur(roi,(5,5),0)
#                     ret3,th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

#                     cv.imshow('roxxxi', roi)
#                     cv.waitKey(0)
#                     return roi
            
# #     def detect(frame):

# #         with tf.gfile.FastGFile('/home/lr/Downloads/test/platenumber_inference/plate_number_inference_graph.pb', 'rb') as f:
# #             graph_def = tf.GraphDef()
# #             graph_def.ParseFromString(f.read())

# #             session.graph.as_default()
# #             tf.import_graph_def(graph_def, name='')

# #             # Read and preprocess an image.
# #             img = cv.resize(frame, (1280, 720))
# #             rows = img.shape[0]
# #             cols = img.shape[1]
# #             inp = cv.resize(img, (300, 300))
# #             inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

# #             # Run the model
# #             out = session.run([session.graph.get_tensor_by_name('num_detections:0'),
# #                             session.graph.get_tensor_by_name('detection_scores:0'),
# #                             session.graph.get_tensor_by_name('detection_boxes:0'),
# #                             session.graph.get_tensor_by_name('detection_classes:0')],
# #                         feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

# #             # Visualize detected bounding boxes.
# #             num_detections = int(out[0][0])
# #             roi = []

# #             for i in range(num_detections):
# #                 classId = int(out[3][0][i])
# #                 score = float(out[1][0][i])
                
# #                 bbox = [float(v) for v in out[2][0][i]]
# #                 if score > 0.3:
# #                     x = bbox[1] * cols
# #                     y = bbox[0] * rows
# #                     right = bbox[3] * cols
# #                     bottom = bbox[2] * rows
# #                     print(classId)
                    
# #                     # else:
# #                     #     print('Truck')
                    
# #                     # print(score * 100)
# #                     cv.rectangle(img, (int(x), int(y)), (int(right),
# #                                                         int(bottom)), (125, 255, 51), thickness=2)
                    
                    
# #                     roi = img[int(y):int(y)+(int(bottom) - int(y)), int(x):int(x)+(int(right) - int(x))]

# #                     if roi.any():
# #                         roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
# #                         blur = cv.GaussianBlur(roi,(5,5),0)
# #                         ret3,th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

# #                         cv.imshow('roxxxi', roi)
# #                         cv.waitKey(0)
# #                         return roi




# a = PlateDetector() 
# a.run(cv.imread('/home/lr/Downloads/test/platenumber_inference/xz.png'))
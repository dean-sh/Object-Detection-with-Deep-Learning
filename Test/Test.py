import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
sys.path.append('research/object_detection')
import matplotlib.pyplot as plt
import time
import cv2


plt.switch_backend('agg')

from collections import defaultdict
from io import StringIO
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
from object_detection.utils import ops as utils_ops
import label_map_util
import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'C:/Users/Dean/Documents/TensorFlows/models/research/object_detection/faster_rcnn_inception_v2_coco'
MODEL_FILE = 'C:/Users/Dean/Documents/TensorFlows/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('C:/Users/Dean/Documents/TensorFlows/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
NUMBER_OF_IMAGES = 5

#opener = urllib.request.URLopener()
#opener.retrieve('http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz')
#download and open the graph
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

#load the frozen TF graph into the memory
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'C:/Users/Dean/Documents/TensorFlows/models/research/object_detection/test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, NUMBER_OF_IMAGES + 1) ]

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for image_path in TEST_IMAGE_PATHS:
            # Preprocessing begins
            #begin_img_tonumpy = time.time()
            # imageio module will load directly to numpy array

            image_np = cv2.imread(image_path)
            #imageio.imread(image_path)
            #print("Time to convert img to numpy: {} seconds".format(round(time.time() - begin_img_tonumpy, 3)))
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            now = time.time()
            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
            print("Time for forward pass: {} seconds".format(round(time.time() - now, 3)))
            # Visualization of the results of a detection.
            now = time.time()
            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=7
              )
            print("Time for postprocessing: {} seconds".format(round(time.time() - now, 3)))
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow('image', image_np)
            cv2.waitKey(0)

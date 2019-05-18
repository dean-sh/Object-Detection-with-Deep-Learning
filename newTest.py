import numpy as np
import os
import sys
import tensorflow as tf
sys.path.append('research/object_detection')
import time
import cv2
import  math
from PIL import Image as img


# This is needed since the notebook is stored in the object_detection folder.
import label_map_util
import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'C:/Users/Dean/Documents/GitHub/models/research/object_detection/mac_n_cheese_graph'
# MODEL_FILE = 'C:/Users/Itay/Documents/models-master/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('C:/Users/Dean/Documents/GitHub/models/research/object_detection/training', 'object-detection.pbtxt')
# PATH_TO_LABELS = os.path.join('C:/Users/Dean/Documents/TensorFlows/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 6
NUMBER_OF_IMAGES = 1


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

PATH_TO_TEST_IMAGES_DIR = 'C:/Users/Dean/Documents/GitHub/models/research/object_detection/test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, NUMBER_OF_IMAGES + 1) ]

#load the frozen TF graph into the memory
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

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
        new_boxes = []
        for image_path in TEST_IMAGE_PATHS:
            # Preprocessing begins
            #begin_img_tonumpy = time.time()
            # imageio module will load directly to numpy array
            image = img.open(image_path)
            width, height = image.size
            image_name = os.path.basename(image_path)

            image_np = cv2.imread(image_path)
            #imageio.imread(image_path)
            #print("Time to convert img to numpy: {} seconds".format(round(time.time() - begin_img_tonumpy, 3)))
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            #now = time.time()
            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            now = time.time()


            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=7,
              min_score_thresh=0.6
              )
            print("Time for postprocessing: {} seconds".format(round(time.time() - now, 3)))
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow('image', image_np)
            cv2.waitKey(0)
            for i, box in enumerate(np.squeeze(boxes)):
                if (np.squeeze(scores)[i] > 0.5):
                    box[0] = math.floor(box[0] * height)
                    box[1] = math.floor(box[1] * width)
                    box[2] = math.floor(box[2] * height)
                    box[3] = math.floor(box[3] * width)
                    box_score =  100*np.squeeze(scores)[i]
                    box_class = np.squeeze(classes)[i]
                    csv_line = np.append(image_name, box)
                    csv_line = np.append(csv_line, box_score)
                    csv_line = np.append(csv_line, box_class)
                    new_boxes.append(csv_line)
        np.savetxt('yourfile.csv', new_boxes, delimiter=',')

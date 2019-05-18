import numpy as np
import os
import sys
import tensorflow as tf
sys.path.append('research/object_detection')
import time
import cv2
import  math
from PIL import Image as img

import ast
import os

PATH_TO_CKPT = 'mac_n_cheese_graph/frozen_inference_graph.pb'
PATH_TO_LABELS ='training/object-detection.pbtxt'
NUM_CLASSES = 6
#Remember to get rid of NUM OF IMGS
NUMBER_OF_IMAGES = 4


class BusesClassifier(object):
    def __init__(self):
        PATH_TO_CKPT = 'mac_n_cheese_graph/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

def get_classification(self, img):
    # Bounding Box Detection.
    with self.detection_graph.as_default():
        # Expand dimension since the model expects image to have shape [1, None, None, 3].
        img_expanded = np.expand_dims(img, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
            feed_dict={self.image_tensor: img_expanded})
    return boxes, scores, classes, num


image_np = cv2.imread('test_images/image1.JPG')
Classifier = BusesClassifier()
boxes = Classifier.d_boxes(image_np)


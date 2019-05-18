import numpy as np
import os
import tensorflow as tf
import cv2
import math
import glob

def bb_intersection_over_union(new_box, boxes_for_iou_eval, pred_class, NUM_CLASSES):
    total_iou = 0
    for i in range(0,NUM_CLASSES):
        if pred_class[i] == 1:  # if prediction is exists
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxes_for_iou_eval[i,0], new_box[0])
            yA = max(boxes_for_iou_eval[i,1], new_box[1])
            xB = min(boxes_for_iou_eval[i,2], new_box[2])
            yB = min(boxes_for_iou_eval[i,3], new_box[3])

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxes_for_iou_eval[i,2] - boxes_for_iou_eval[i,0] + 1) * (boxes_for_iou_eval[i,3] - boxes_for_iou_eval[i,1] + 1)
            boxBArea = (new_box[2] - new_box[0] + 1) * (new_box[3] - new_box[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = interArea / float(boxAArea + boxBArea - interArea)
            total_iou = max(total_iou,iou)

    # return the intersection over union value
    return total_iou


def run(myAnnFileName, buses):

    # What model to use.
    MODEL_NAME = 'C:/Users/Itay/Documents/models-master/research/object_detection/mac_n_cheese_graph'
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    NUM_CLASSES = 6
    MAX_IOU = 0.6

    # place file name to save
    annFileEstimations = open(myAnnFileName, 'w+')

    # load the frozen TF graph into the memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    TEST_IMAGE_PATHS = buses + '/*jpg'
    TEST_IMAGE_PATHS = glob.glob(TEST_IMAGE_PATHS)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            new_boxes = []
            for image_path in TEST_IMAGE_PATHS:
                # Preprocessing begins

                image_np = cv2.imread(image_path)
                height = np.size(image_np, 0)
                width = np.size(image_np, 1)
                image_name = os.path.basename(image_path)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # Getting the detection parameters from the bounding boxes and writing to txt file.
                pred_class = np.zeros(NUM_CLASSES, dtype=int)
                boxes_for_iou_eval = np.zeros((NUM_CLASSES, 4), dtype=int)    # 4 = xmin,ymin,xmax,ymax, total number of boxes can be 6
                for i, box in enumerate(np.squeeze(boxes)):
                    if np.squeeze(scores)[i] > 0.5:
                        box_class = np.squeeze(classes)[i]
                        box_class = box_class.astype(int)
                        # find only one box per class:
                        # remember that scores are arranged from max to min
                        if pred_class[box_class - 1] == 0:
                            ymin = math.floor(box[0] * height)
                            xmin = math.floor(box[1] * width)
                            ymax = math.floor(box[2] * height)
                            xmax = math.floor(box[3] * width)
                            new_box = [xmin, ymin, xmax, ymax]
                            if i == 0:      # first box - don't need to clc iou
                                iou = 0
                            else:
                                iou = bb_intersection_over_union(new_box, boxes_for_iou_eval, pred_class, NUM_CLASSES)
                            if iou < MAX_IOU:       # if no box is overlap with another
                                boxes_for_iou_eval[box_class - 1] = new_box
                                # define line to show:
                                csv_line = [xmin, ymin, xmax - xmin, ymax - ymin, box_class]
                                new_boxes.append(csv_line)
                                csv_line = map(str, csv_line)
                                csv_line = ','.join(csv_line)
                                csv_line = '[' + csv_line + ']'
                                if i == 0:
                                    posStr = csv_line
                                else:
                                    posStr += ',' + csv_line
                            pred_class[box_class - 1] = 1

                strToWrite = image_name + ':'
                strToWrite += posStr + '\n'

                annFileEstimations.write(strToWrite)

"""Yolo v3 detection script.

Saves the detections in the `detection` folder.

Usage:
    python detect.py <images/video> <iou threshold> <confidence threshold> <filenames>

Example:
    python detect.py images 0.5 0.5 data/images/dog.jpg data/images/office.jpg
    python detect.py video 0.5 0.5 data/video/shinjuku.mp4

Note that only one video can be processed at one run.
"""

import tensorflow as tf
import sys
import cv2

from core.yolo_v3 import Yolo_v3
from core.utils import load_images, load_class_names, draw_boxes, draw_frame

_MODEL_SIZE = (416, 416)
_CLASS_NAMES_FILE = './data/labels/coco.names'
_MAX_OUTPUT_SIZE = 20


def main(type, iou_threshold, confidence_threshold, input_names):
    if type != 'images':
        raise ValueError("Inappropriate data type.")

    class_names = load_class_names(_CLASS_NAMES_FILE)
    n_classes = len(class_names)

    batch_size = len(input_names)
    batch = load_images(input_names, model_size=_MODEL_SIZE)
    inputs = tf.placeholder(tf.float32, [batch_size, *_MODEL_SIZE, 3])

    model = Yolo_v3(inputs, n_classes=n_classes, model_size=_MODEL_SIZE,
                    max_output_size=_MAX_OUTPUT_SIZE,
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold, trainable=False)
    
    model.eval()
    saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))

    with tf.Session() as sess:
        saver.restore(sess, './weights/model.ckpt')
        detection_result = sess.run(model.boxes_dicts, 
            feed_dict={inputs: batch})

    draw_boxes(input_names, detection_result, class_names, _MODEL_SIZE)

    print('Detections have been saved successfully.')

if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4:])

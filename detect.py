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
import time
import os
import numpy as np
import core.utils as utils

from core.yolo_v3 import Yolo_v3
from core.yolo_components import build_boxes,\
    non_max_suppression, eval_precision_recall
from core.config import cfg
from core.dataset import Dataset

_MODEL_SIZE = (416, 416)
_CLASS_NAMES_FILE = './data/labels/coco.names'
_MAX_OUTPUT_SIZE = 20

_STRIDES = np.array(cfg.YOLO.STRIDES)
_OUTPUT_SIZE = 416 // _STRIDES
_CHECKPOINT_FN_200 = './checkpoint/rn-rn-yolov3_B200.ckpt'
_CHECKPOINT_FN_500 = './checkpoint/rn-rn-yolov3_B500.ckpt'
_MODEL_CKPT = './checkpoint/yolov3_coco_demo.ckpt'

def detect_dataset(iou_threshold, confidence_threshold, checkpoint_fn):
    
    t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    detection_dir = './detections/'+t+'/'
    if not os.path.exists(detection_dir):
        os.mkdir(detection_dir)
    test_detection_dir = detection_dir+'test/'
    baseline_detection_dir = detection_dir+'baseline/'
    if not os.path.exists(test_detection_dir):
        os.mkdir(test_detection_dir)
    if not os.path.exists(baseline_detection_dir):
        os.mkdir(baseline_detection_dir)

    class_names = utils.load_class_names(_CLASS_NAMES_FILE)
    n_classes = len(class_names)

    inputs = tf.placeholder(dtype=tf.float32, shape=[None, *_MODEL_SIZE, 3])
    #mask_placeholders = utils.build_mask_placeholders(_OUTPUT_SIZE)

    label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
    label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
    label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
    true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
    true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
    true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')

    model = Yolo_v3(inputs=inputs, 
                    #mask_placeholders=mask_placeholders,
                    trainable=False, 
                    n_classes=n_classes, 
                    model_size=_MODEL_SIZE)
        
    obj_conf_loss, no_obj_conf_loss, obj_class_loss, \
        obj_loc_loss = \
        model.compute_loss(label_sbbox, label_mbbox, label_lbbox,\
                           true_sbboxes, true_mbboxes, true_lbboxes)
    
    loss = obj_conf_loss + no_obj_conf_loss + obj_class_loss + obj_loc_loss

    #model.eval()
    
    with tf.name_scope('ema'):
        ema_obj = tf.train.ExponentialMovingAverage(0.9995)
    #saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))
    

    #saver = tf.compat.v1.train.Saver(ema_obj.variables_to_restore())
    #saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))
    #saver.restore(sess, checkpoint_fn)

    batches = Dataset('test')
    batch_size = batches.batch_size

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        i = 0
        for batch in batches:
            if i >= 50:
                break
            
            saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))
            saver.restore(sess, checkpoint_fn)
            
            batch_img, label_boxes, boxes, img_paths = batch
            model.eval(batch_size, _MAX_OUTPUT_SIZE, iou_threshold, confidence_threshold)
            
            #feed_dict = utils.construct_feed_dict(mask_placeholders, *noobj_masks)
            feed_dict = dict()
            feed_dict.update({inputs:   batch_img,
                              label_sbbox:  label_boxes[0],
                              label_mbbox:  label_boxes[1],
                              label_lbbox:  label_boxes[2],
                              true_sbboxes: boxes[0],
                              true_mbboxes: boxes[1],
                              true_lbboxes: boxes[2]})
            boxes_dicts, detect_loss = sess.run([model.boxes_dicts, loss], feed_dict=feed_dict)

            recall, precision = eval_precision_recall(batch_size, boxes_dicts, label_boxes, n_classes)
            print("Batch: {0}; Loss: {1:.2f}; Precision: {2:.2f}; Recall: {3:.2f}"\
                    .format(i, detect_loss, precision, recall))
            utils.draw_boxes_new(img_paths, boxes_dicts, class_names, _MODEL_SIZE, test_detection_dir, shape='e')

############################### Bar Between Test and Baseline solutions ######################

            saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))
            saver.restore(sess, _MODEL_CKPT)
            
            model.eval(batch_size, _MAX_OUTPUT_SIZE, iou_threshold, confidence_threshold)

            baseline_boxes_dicts, baseline_detect_loss = sess.run([model.boxes_dicts, loss], feed_dict=feed_dict)
            
            bl_recall, bl_precision = eval_precision_recall(batch_size, baseline_boxes_dicts,\
                                     label_boxes, n_classes)
            print("Batch: {0}; Baseline Loss: {1:.2f}; Precision: {2:.2f}; Recall: {3:.2f}"\
                    .format(i, baseline_detect_loss, bl_precision, bl_recall))
            #print("Batch: {0}; Baseline Loss: {1:.2f}".format(i, baseline_detect_loss))
            utils.draw_boxes_new(img_paths, baseline_boxes_dicts, class_names, _MODEL_SIZE, baseline_detection_dir,shape='e')

            i += 1

    print('Detections have been saved successfully.')


def detect_images(iou_threshold, confidence_threshold, checkpoint_fn, select_fn):
    with open(select_fn, 'r') as f:
        txt = f.readlines()
        img_fns = [line.strip() for line in txt]

    if len(img_fns)==0:
        raise KeyError("No input images to detect.")

    t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    detection_dir = './detections/'+t+'/'
    if not os.path.exists(detection_dir):
        os.mkdir(detection_dir)
    test_detection_dir = detection_dir+'test/'
    baseline_detection_dir = detection_dir+'baseline/'
    if not os.path.exists(test_detection_dir):
        os.mkdir(test_detection_dir)
    if not os.path.exists(baseline_detection_dir):
        os.mkdir(baseline_detection_dir)

    class_names = utils.load_class_names(_CLASS_NAMES_FILE)
    n_classes = len(class_names)

    inputs = tf.placeholder(dtype=tf.float32, shape=[None, *_MODEL_SIZE, 3])

    model = Yolo_v3(inputs=inputs, 
                    #mask_placeholders=mask_placeholders,
                    trainable=False, 
                    n_classes=n_classes, 
                    model_size=_MODEL_SIZE)
        
    with tf.name_scope('ema'):
        ema_obj = tf.train.ExponentialMovingAverage(0.9995)
    #saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))
    

    #saver = tf.compat.v1.train.Saver(ema_obj.variables_to_restore())
    #saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))
    #saver.restore(sess, checkpoint_fn)

    batch_size = len(img_fns)
    batch = utils.load_images(img_fns, model_size=_MODEL_SIZE)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            
        saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))
        saver.restore(sess, checkpoint_fn)
        
        model.eval(batch_size, _MAX_OUTPUT_SIZE, iou_threshold, confidence_threshold)

        feed_dict = {inputs: batch}
       
        boxes_dicts = sess.run(model.boxes_dicts, feed_dict=feed_dict)
        
        utils.draw_boxes_new(img_fns, boxes_dicts, class_names, _MODEL_SIZE, test_detection_dir, shape='e')

############################### Bar Between Test and Baseline solutions ######################

        saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))
        saver.restore(sess, _MODEL_CKPT)
        
        model.eval(batch_size, _MAX_OUTPUT_SIZE, iou_threshold, confidence_threshold)
        
        feed_dict = {inputs: batch}
        baseline_boxes_dicts= sess.run(model.boxes_dicts, feed_dict=feed_dict)
        
        utils.draw_boxes_new(img_fns, baseline_boxes_dicts, class_names, _MODEL_SIZE, baseline_detection_dir, shape='e')

    print('Detections have been saved successfully.')


def detect_video(iou_threshold, confidence_threshold, checkpoint_fn, video_fn):

    t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    detection_dir = './detections/'+t+'/'
    if not os.path.exists(detection_dir):
        os.mkdir(detection_dir)
    filename = os.path.basename(video_fn)

    class_names = utils.load_class_names(_CLASS_NAMES_FILE)
    n_classes = len(class_names)

    inputs = tf.placeholder(dtype=tf.float32, shape=[1, *_MODEL_SIZE, 3])

    model = Yolo_v3(inputs=inputs, 
                    #mask_placeholders=mask_placeholders,
                    trainable=False, 
                    n_classes=n_classes, 
                    model_size=_MODEL_SIZE)
    model.eval(1, _MAX_OUTPUT_SIZE, iou_threshold, confidence_threshold)
    with tf.name_scope('ema'):
        ema_obj = tf.train.ExponentialMovingAverage(0.9995)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            
        saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))
        saver.restore(sess, checkpoint_fn)
        #model.eval(batch_size, _MAX_OUTPUT_SIZE, iou_threshold, confidence_threshold)

    ###########################################

        win_name = 'Video detection'
        cv2.namedWindow(win_name)
        cap = cv2.VideoCapture(video_fn)
        frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                      cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #fourcc = cv2.VideoWriter_fourcc(*'X264')
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(detection_dir+'detection_'+filename, fourcc, fps,
                              (int(frame_size[0]), int(frame_size[1])))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                resized_frame = cv2.resize(frame, dsize=_MODEL_SIZE[::-1],
                                           interpolation=cv2.INTER_NEAREST)
                resized_frame = resized_frame / 255.
                detection_result = sess.run(model.boxes_dicts,
                                            feed_dict={inputs: [resized_frame]})

                utils.draw_frame_new(frame, frame_size, detection_result,
                           class_names, _MODEL_SIZE, shape='e')

                cv2.imshow(win_name, frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                out.write(frame)
        finally:
            cv2.destroyAllWindows()
            cap.release()
            print('Detections have been saved successfully.')


'''
    python detect.py dataset .5 .5 ckpt
    python detect.py images .5 .5 ckpt voc.txt
    python detect.py video .5 .5 ckpt video_fn
'''
if __name__ == '__main__':
    # use the preset dataset
    # compute loss and detect
    if sys.argv[1]=='dataset':
        detect_dataset(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
    
    # use the input images
    # don't compute loss, just detect
    elif sys.argv[1]=='images':
        detect_images(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5])


    elif sys.argv[1]=='video':
        detect_video(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5])
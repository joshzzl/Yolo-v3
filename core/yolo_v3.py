
# Author: Zhili Zhang
# Date: Sept 12, 2019

import numpy as np
import tensorflow as tf
import core.utils as utils
from core.yolo_components import yolo_convolutional,\
    yolo_detection, upsample, build_boxes,\
    non_max_suppression
from core.darknet import darknet53
from core.layers import convolutional
from core.config import cfg

_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]


class Yolo_v3:
    # You only look once v3

    def __init__(self, inputs, trainable, n_classes, model_size, 
                 #max_output_size, iou_threshold, confidence_threshold, 
                 data_format='channels_last'):
        '''
        Args:
            n_classes: # of class labels
            model_size: input size of the model
            max_output_size: max number of boxes to be selected for each class.
            iou_threshold: threshold of the IOU (intersect over union)
            confidence_threshold: Threshold for the confidence score
            data_format: The input format
        '''
        if not data_format:
            if tf.test.is_built_with_cuda():
                data_format = 'channels_first'
            else:
                data_format = 'channels_last'

        self.n_classes = n_classes
        self.model_size = model_size
        #self.max_output_size = max_output_size
        #self.iou_threshold = iou_threshold
        #self.confidence_threshold = confidence_threshold
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh  = cfg.YOLO.IOU_LOSS_THRESH
        self.data_format = data_format
        self.trainable = trainable

        #self.__build_placeholders()

        try:
            conv_lbbox, conv_mbbox, conv_sbbox, pred_lbbox, pred_mbbox, pred_sbbox \
                = self.__build(inputs)
            self.conv_lbbox = conv_lbbox
            self.conv_mbbox = conv_mbbox
            self.conv_sbbox = conv_sbbox
            self.pred_lbbox = pred_lbbox
            self.pred_mbbox = pred_mbbox
            self.pred_sbbox = pred_sbbox
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

    '''
    def __build_placeholders(self):
        self.inputs = tf.placeholder(tf.float32, [None, *self.model_size, 3])
    '''
    
    def __build(self, inputs):
        with tf.variable_scope('yolo_v3_model'):
            if self.data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            # mark this step
            #inputs = inputs / 255

            route1, route2, inputs = darknet53(inputs=inputs, trainable=self.trainable,
                                               data_format=self.data_format)

            route, inputs = yolo_convolutional(inputs=inputs, filters=512, 
                trainable=self.trainable, data_format=self.data_format, name='yolo_conv0')

            conv_lbbox, pred_lbbox = yolo_detection(inputs=inputs, 
                                 n_classes=self.n_classes,
                                 anchors=_ANCHORS[6:9],
                                 img_size=self.model_size, trainable=self.trainable,
                                 data_format=self.data_format, name='conv_lbbox')

            inputs = convolutional(inputs=route, filters=256, kernel_size=1, 
                                    trainable=self.trainable, name='conv57',
                                    data_format=self.data_format)

            upsample_size = route2.get_shape().as_list()
            inputs = upsample(inputs=inputs, out_shape=upsample_size,
                              data_format=self.data_format, name='upsample0')
            
            if self.data_format=='channels_first':
                axis = 1
            else:
                axis = 3

            with tf.variable_scope('route_1'):
                inputs = tf.concat([inputs, route2], axis=axis)
            
            route, inputs = yolo_convolutional(inputs=inputs, filters=256, 
                trainable=self.trainable, data_format=self.data_format, name='yolo_conv1')
            
            conv_mbbox, pred_mbbox = yolo_detection(inputs=inputs, 
                                 n_classes=self.n_classes,
                                 anchors=_ANCHORS[3:6],
                                 img_size=self.model_size, trainable=self.trainable,
                                 data_format=self.data_format, name='conv_mbbox')

            inputs = convolutional(inputs=route, filters=128, kernel_size=1, 
                                    trainable=self.trainable, name='conv63',
                                    data_format=self.data_format)

            upsample_size = route1.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_size,
                              data_format=self.data_format, name='upsample1')
            
            with tf.variable_scope('route_2'):
                inputs = tf.concat([inputs, route1], axis=axis)
            
            route, inputs = yolo_convolutional(inputs=inputs, filters=128, 
                trainable=self.trainable, data_format=self.data_format, name='yolo_conv2')

            conv_sbbox, pred_sbbox = yolo_detection(inputs=inputs, 
                                 n_classes=self.n_classes,
                                 anchors=_ANCHORS[0:3],
                                 img_size=self.model_size, trainable=self.trainable,
                                 data_format=self.data_format, name='conv_sbbox')

            return conv_lbbox, conv_mbbox, conv_sbbox,\
                pred_lbbox, pred_mbbox, pred_sbbox

    '''
    def eval(self):
        pred_bbox_shape = self.pred_lbbox.get_shape().as_list()
        batch_size = pred_bbox_shape[0]

        pred_lbbox = tf.reshape(self.pred_lbbox, [batch_size, -1, 5+self.n_classes])
        pred_mbbox = tf.reshape(self.pred_mbbox, [batch_size, -1, 5+self.n_classes])
        pred_sbbox = tf.reshape(self.pred_sbbox, [batch_size, -1, 5+self.n_classes])

        inputs = tf.concat([pred_lbbox, pred_mbbox, pred_sbbox], axis=1)

        inputs = build_boxes(inputs)

        boxes_dicts = non_max_suppression(
            inputs, n_classes=self.n_classes,
            max_output_size=self.max_output_size,
            iou_threshold=self.iou_threshold,
            confidence_threshold=self.confidence_threshold)

        self.boxes_dicts = boxes_dicts
    '''


    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    # this function calculates ground truth IOU, which is IOU - (Enclose - Union)/Enclose
    def bbox_giou(self, boxes1, boxes2):

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    '''
    This layer is under the condition that we focus on a certain class of anchors (small, med, large)
    size = (13, 26, 52) <=> stride = (32, 16, 8)
    Input arguments:
        conv: output tensor of Yolo, not being decoded  - shape (n, size, size, 255)
        pred: decoded output tensor, with bbox info - shape (n, size, size, 3, 85)
        label: tensor storing label info of training data, shape (n, size, size, 3, 85)
        bboxes: tensor storing bbox infor of training data, shape (n, 150, 4)
        anchors: 3 anchors in that specific category (small / med / large)
        stride: the stride for that category, (s 8; m 16; l 32)
    '''
    def loss_layer(self, conv, pred, label, bboxes, stride):

        # 
        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.n_classes))
        # raw confidence
        conv_raw_conf = conv[:, :, :, :, 4:5]
        # raw prob for 80 classes
        conv_raw_prob = conv[:, :, :, :, 5:]

        # pred x,y (center) and width, height
        pred_xywh     = pred[:, :, :, :, 0:4]
        # pred confidence = sigmoid(raw)
        pred_conf     = pred[:, :, :, :, 4:5]

        # xywh ground truth
        label_xywh    = label[:, :, :, :, 0:4]
        # 1 if (i, yind, xind, j) (j=0,1,2 of three anchors) the label box best closed to the anchor; 0 otherwise
        respond_bbox  = label[:, :, :, :, 4:5]
        # all 0 if above value is 0, smooth one-hot if above value is 1
        label_prob    = label[:, :, :, :, 5:]

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1) 
        input_size = tf.cast(input_size, tf.float32)

        # this is a scaling method to strengthen the influence of small bbox's giou
        # basically if the bbox is small, then this scale is greater (2 - box_area/total_area)
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < self.iou_loss_thresh, tf.float32 )

        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

        return giou_loss, conf_loss, prob_loss



    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         stride=self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         stride=self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         stride=self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        return giou_loss, conf_loss, prob_loss
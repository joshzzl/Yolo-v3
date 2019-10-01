
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

    def __init__(self, inputs, #mask_placeholders, 
                 trainable, n_classes, model_size, 
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

        #self.mask_placeholders = mask_placeholders
        self.n_classes = n_classes
        self.model_size = model_size
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh  = cfg.YOLO.IOU_LOSS_THRESH
        self.data_format = data_format
        self.trainable = trainable

        try:
            conv_boxes, pred_boxes, xy_offsets = self.__build(inputs)
            conv_lbbox, conv_mbbox, conv_sbbox = conv_boxes
            pred_lbbox, pred_mbbox, pred_sbbox = pred_boxes
            xy_offset_l, xy_offset_m, xy_offset_s = xy_offsets
                
            self.conv_lbbox = conv_lbbox
            self.conv_mbbox = conv_mbbox
            self.conv_sbbox = conv_sbbox
            self.pred_lbbox = pred_lbbox
            self.pred_mbbox = pred_mbbox
            self.pred_sbbox = pred_sbbox
            self.xy_offset_l = xy_offset_l
            self.xy_offset_m = xy_offset_m
            self.xy_offset_s = xy_offset_s
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

    
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

            conv_lbbox, pred_lbbox, xy_offset_l = yolo_detection(inputs=inputs, 
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
            
            conv_mbbox, pred_mbbox, xy_offset_m = yolo_detection(inputs=inputs, 
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

            conv_sbbox, pred_sbbox, xy_offset_s = yolo_detection(inputs=inputs, 
                                 n_classes=self.n_classes,
                                 anchors=_ANCHORS[0:3],
                                 img_size=self.model_size, trainable=self.trainable,
                                 data_format=self.data_format, name='conv_sbbox')

            return [conv_lbbox, conv_mbbox, conv_sbbox],\
                   [pred_lbbox, pred_mbbox, pred_sbbox],\
                   [xy_offset_l, xy_offset_m, xy_offset_s]

    
    def eval(self, batch_size, max_output_size, iou_threshold, confidence_threshold):

        pred_lbbox = tf.reshape(self.pred_lbbox, [batch_size, -1, 5+self.n_classes])
        pred_mbbox = tf.reshape(self.pred_mbbox, [batch_size, -1, 5+self.n_classes])
        pred_sbbox = tf.reshape(self.pred_sbbox, [batch_size, -1, 5+self.n_classes])

        inputs = tf.concat([pred_lbbox, pred_mbbox, pred_sbbox], axis=1)

        inputs = build_boxes(inputs)

        boxes_dicts = non_max_suppression(
            inputs, n_classes=self.n_classes,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold)

        self.boxes_dicts = boxes_dicts
    


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
        iou = tf.div_no_nan(inter_area, union_area)
        area_ar = [boxes1_area, boxes2_area, inter_area, union_area]


        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * tf.div_no_nan((enclose_area - union_area), enclose_area)

        return giou, iou, area_ar, enclose_area

    # boxes1: (N, 13, 13, 3, 1, 4); boxes2: (N, 1, 1, 1, 150, 4)
    # return (N, 13, 13, 3, 150)
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
        iou = 1.0 * tf.div_no_nan(inter_area, union_area)

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
    def loss_layer(self, conv, pred, label, bboxes, stride, xy_offset, anchors):

        # 
        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        # 32 * 13 = 16 * 26 = 8 * 52 = 416
        input_size  = stride * output_size
        # (N, 13, 13, 3, 85)
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
        
        # compute the label_xy and pred_xy by divide by stride and remove offset
        # compute the label_wh and pred_wh by divide by anchors
        pred_xy = pred[:, :, :, :, 0:2] / stride - xy_offset
        label_xy = label[:, :, :, :, 0:2] / stride - xy_offset
        pred_wh = pred[:, :, :, :, 2:4] / anchors
        label_wh = label[:, :, :, :, 2:4] / anchors

        label_wh = tf.where(condition=tf.equal(label_wh, 0),
                            x=tf.ones_like(label_wh), y=label_wh)
        pred_wh = tf.where(condition=tf.equal(pred_wh, 0),
                            x=tf.ones_like(pred_wh), y=pred_wh)
        label_wh = tf.log(tf.clip_by_value(label_wh, 1e-9, 1e9))
        pred_wh = tf.log(tf.clip_by_value(pred_wh, 1e-9, 1e9))

        # 1 if (i, yind, xind, j) (j=0,1,2 of three anchors) the label box best closed to the anchor; 0 otherwise
        obj_mask = label[:, :, :, :, 4:5]

        # all 0 if above value is 0, smooth one-hot if above value is 1
        label_prob    = label[:, :, :, :, 5:]

        giou, mid_iou, area_ar, enclose_area = self.bbox_giou(pred_xywh, label_xywh)
        giou = tf.expand_dims(giou, axis=-1) 

        input_size = tf.cast(input_size, tf.float32)

        # this is a scaling method to strengthen the influence of small bbox's giou
        # basically if the bbox is small, then this scale is greater (2 - box_area/total_area)
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)

        ###############################################################################################################
        # (N, 13, 13, 3, 150)
        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        # (N, 13, 13, 3, 1)
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
        no_obj_mask = (1.0 - obj_mask) * tf.cast( max_iou < self.iou_loss_thresh, tf.float32 )

        #obj_loc_loss = obj_mask * bbox_loss_scale * (label_xywh - pred_xywh)**2
        obj_xy_loss = tf.reduce_sum(tf.square(label_xy - pred_xy) * obj_mask * bbox_loss_scale) / tf.cast(batch_size, dtype=tf.float32)
        obj_wh_loss = tf.reduce_sum(tf.square(label_wh - pred_wh) * obj_mask * bbox_loss_scale) / tf.cast(batch_size, dtype=tf.float32)
        obj_loc_loss = obj_xy_loss + obj_wh_loss
        
        # !!! Notice this part is gonna be modified
        #obj_conf_loss = obj_mask * tf.square(1- giou)
        true_conf = obj_mask * giou
        obj_conf_loss = obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_conf, logits=conv_raw_conf)
        #obj_conf_loss = obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_mask, logits=conv_raw_conf)
        obj_conf_loss = tf.reduce_mean(tf.reduce_sum(obj_conf_loss, axis=[1,2,3,4]))

        no_obj_conf_loss = no_obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_mask, logits=conv_raw_conf)
        no_obj_conf_loss = tf.reduce_mean(tf.reduce_sum(no_obj_conf_loss, axis=[1,2,3,4]))
        
        obj_class_loss = obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
        obj_class_loss = tf.reduce_mean(tf.reduce_sum(obj_class_loss, axis=[1,2,3,4]))
        
        '''
        conf_focal = self.focal(respond_bbox, pred_conf)
        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )
        obj_conf_loss = tf.reduce_mean(tf.reduce_sum(obj_conf_loss, axis=[1,2,3,4]))
        no_obj_conf_loss = tf.reduce_mean(tf.reduce_sum(no_obj_conf_loss, axis=[1,2,3,4]))
        obj_loc_loss = tf.reduce_mean(tf.reduce_sum(obj_loc_loss, axis=[-1]))
        '''

        return obj_conf_loss, no_obj_conf_loss, obj_class_loss, obj_loc_loss #, mid_iou, area_ar, enclose_area



    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         stride=self.strides[0], xy_offset=self.xy_offset_s, 
                                         anchors=_ANCHORS[0:3])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         stride=self.strides[1], xy_offset=self.xy_offset_m, 
                                         anchors=_ANCHORS[3:6])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         stride=self.strides[2], xy_offset=self.xy_offset_l, 
                                         anchors=_ANCHORS[6:9])

        with tf.name_scope('obj_conf_loss'):
            obj_conf_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('no_obj_conf_loss'):
            no_obj_conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('obj_class_loss'):
            obj_class_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        with tf.name_scope('obj_loc_loss'):
            obj_loc_loss = loss_sbbox[3] + loss_mbbox[3] + loss_lbbox[3]

        #obj_loc_loss = loss_sbbox[3] + loss_mbbox[3] + loss_lbbox[3]
        #iou_mid = [loss_sbbox[4], loss_mbbox[4], loss_lbbox[4]]
        #areas = [loss_sbbox[5], loss_mbbox[5], loss_lbbox[5]]
        #enc_areas = [loss_sbbox[6], loss_mbbox[6], loss_lbbox[6]]

        return obj_conf_loss, no_obj_conf_loss, obj_class_loss, obj_loc_loss #, iou_mid, areas, enc_areas
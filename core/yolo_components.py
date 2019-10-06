
# Author: Zhili Zhang
# Date: Sept 12, 2019

import tensorflow as tf 
import numpy as np
from core.layers import convolutional
from collections import Counter
from tensorflow.keras.layers import Conv2D


def yolo_convolutional(inputs, filters, trainable, data_format, name):

    with tf.variable_scope(name):
    
        inputs = convolutional(inputs=inputs, filters=filters, kernel_size=1,
            trainable=trainable, name='conv0', data_format=data_format)
        
        inputs = convolutional(inputs=inputs, filters=2*filters, kernel_size=3,
            trainable=trainable, name='conv1', data_format=data_format)
        
        inputs = convolutional(inputs=inputs, filters=filters, kernel_size=1,
            trainable=trainable, name='conv2', data_format=data_format)
        
        inputs = convolutional(inputs=inputs, filters=2*filters, kernel_size=3,
            trainable=trainable, name='conv3', data_format=data_format)
        
        inputs = convolutional(inputs=inputs, filters=filters, kernel_size=1,
            trainable=trainable, name='conv4', data_format=data_format)

        route = inputs

        inputs = convolutional(inputs=inputs, filters=2*filters, kernel_size=3,
            trainable=trainable, name='conv5', data_format=data_format)

        return route, inputs


def yolo_detection(inputs, n_classes, anchors, img_size, 
                    trainable, data_format, name):

    '''
    Args:
        inputs: tensor input
        n_classes: number of labels
        anchors: a list of anchor sizes
        img_size: the input size of the model
        data_format: input format
    '''

    n_anchors = len(anchors)
    filters = n_anchors * (5 + n_classes)

    inputs = convolutional(inputs=inputs, filters=filters, kernel_size=1,
                            trainable=trainable, name=name, 
                            data_format=data_format, act=False, bn=False)

    # raw output of detection conv layer
    raw_output = inputs

    shape = inputs.get_shape().as_list()

    # channels_first: NCHW
    # channels_last: NHWC
    if data_format=='channels_first':
        grid_shape = shape[2:4]
        # reshape to NHWC
        inputs = tf.transpose(inputs, [0,2,3,1])
        raw_output = tf.transpose(raw_output, [0,2,3,1])
    else:
        grid_shape = shape[1:3]

    inputs = tf.reshape(inputs, [-1, n_anchors * grid_shape[0] * grid_shape[1],
                                 5 + n_classes])
    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])

    # split & get the 4 components of output
    box_centers, box_shapes, confidence, classes = \
        tf.split(inputs, [2, 2, 1, n_classes], axis=-1)

    x = tf.range(grid_shape[0], dtype=tf.float32)
    y = tf.range(grid_shape[1], dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(x, y)
    x_offset = tf.reshape(x_offset, (-1, 1))
    y_offset = tf.reshape(y_offset, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    
    xy_offset_output = tf.identity(x_y_offset)
    xy_offset_output = tf.reshape(xy_offset_output, [grid_shape[0], grid_shape[1], 1, 2])

    x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
    x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
    box_centers = tf.nn.sigmoid(box_centers)
    box_centers = (box_centers + x_y_offset) * strides

    anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
    box_shapes = tf.exp(box_shapes) * tf.cast(anchors, tf.float32)

    confidence = tf.nn.sigmoid(confidence)

    classes = tf.nn.sigmoid(classes)

    inputs = tf.concat([box_centers, box_shapes,
                        confidence, classes], axis=-1)

    inputs = tf.reshape(inputs, [-1, grid_shape[0], grid_shape[1], 
                                 n_anchors, 5+n_classes])

    return raw_output, inputs, xy_offset_output


def upsample(inputs, out_shape, data_format, name):
    """Upsamples to `out_shape` using nearest neighbor interpolation."""
    if data_format == 'channels_first':
        # if NCHW then convert to NHWC
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        new_height = out_shape[3]
        new_width = out_shape[2]
    else:
        new_height = out_shape[2]
        new_width = out_shape[1]

    with tf.variable_scope(name):
        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    return inputs


def build_boxes(inputs):
    """Computes top left and bottom right points of the boxes."""
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    top_left_x = center_x - width / 2
    top_left_y = center_y - height / 2
    bottom_right_x = center_x + width / 2
    bottom_right_y = center_y + height / 2

    boxes = tf.concat([top_left_x, top_left_y,
                       bottom_right_x, bottom_right_y,
                       confidence, classes], axis=-1)

    return boxes


def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold,
                        confidence_threshold):
    """Performs non-max suppression separately for each class.

    Args:
        inputs: Tensor input.
        n_classes: Number of classes.
        max_output_size: Max number of boxes to be selected for each class.
        iou_threshold: Threshold for the IOU.
        confidence_threshold: Threshold for the confidence score.
    Returns:
        A list containing class-to-boxes dictionaries
            for each sample in the batch.
    """
    batch = tf.unstack(inputs)
    boxes_dicts = []
    for boxes in batch:
        boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)
        classes = tf.argmax(boxes[:, 5:], axis=-1)
        classes = tf.expand_dims(tf.to_float(classes), axis=-1)
        boxes = tf.concat([boxes[:, :5], classes], axis=-1)

        boxes_dict = dict()
        for cls in range(n_classes):
            mask = tf.equal(boxes[:, 5], cls)
            mask_shape = mask.get_shape()
            if mask_shape.ndims != 0:
                class_boxes = tf.boolean_mask(boxes, mask)
                boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes,
                                                              [4, 1, -1],
                                                              axis=-1)
                boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
                indices = tf.image.non_max_suppression(boxes_coords,
                                                       boxes_conf_scores,
                                                       max_output_size,
                                                       iou_threshold)
                class_boxes = tf.gather(class_boxes, indices)
                boxes_dict[cls] = class_boxes[:, :5]

        boxes_dicts.append(boxes_dict)

    return boxes_dicts

# based on the nms results, further compute the precision & recall
# ypred: [3] * [N, 13, 13, 3, 85]
# ytrue: [3] * [N, 13, 13, 3, 85]
def eval_precision_recall(batch_size, batch_pred_boxes, y_true, num_classes, iou_thresh=0.5):
    #num_images = y_true[0].shape[0]
    #num_images = y_true[0].shape[0]

    true_labels_dict = {i: 0 for i in range(num_classes)}  # {class: count}
    pred_labels_dict = {i: 0 for i in range(num_classes)}
    true_positive_dict = {i: 0 for i in range(num_classes)}

    for i in range(batch_size):
        true_labels_list, true_boxes_list = [], []
        for j in range(3):  # three feature maps
            # shape: [13, 13, 3, 80]
            true_probs_temp = y_true[j][i][..., 5:-1]
            # shape: [13, 13, 3, 4] (x_center, y_center, w, h)
            true_boxes_temp = y_true[j][i][..., 0:4]

            # [13, 13, 3]
            object_mask = true_probs_temp.sum(axis=-1) > 0

            # [V, 80] V: Ground truth number of the current image
            true_probs_temp = true_probs_temp[object_mask]
            # [V, 4]
            true_boxes_temp = true_boxes_temp[object_mask]

            # [V], labels
            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            # [V, 4] (x_center, y_center, w, h)
            true_boxes_list += true_boxes_temp.tolist()

        if len(true_labels_list) != 0:
            for cls, count in Counter(true_labels_list).items():
                true_labels_dict[cls] += count

        # [V, 4] (xmin, ymin, xmax, ymax)
        true_boxes = np.array(true_boxes_list)
        box_centers, box_sizes = true_boxes[:, 0:2], true_boxes[:, 2:4]
        true_boxes[:, 0:2] = box_centers - box_sizes / 2.
        true_boxes[:, 2:4] = true_boxes[:, 0:2] + box_sizes

        #box_pred [batch_size]*{cls: [:, (xmin, ymin, xmax, ymax, conf)]}
        pred_boxes = batch_pred_boxes[i]
        pred_xy = []
        pred_conf = []
        pred_labels = []
        for k, boxes in pred_boxes.items():
            # append labels
            pred_labels += [k]*boxes.shape[0]
            pred_conf += boxes[:, 4].tolist()
            pred_xy += boxes[:, 0:4].tolist()
        # pred_xy: [N, 4]
        # pred_conf: [N]
        # pred_labels: [N]
        # N: Detected box number of the current image


        # len: N
        #pred_labels_list = [] if pred_labels is None else pred_labels.tolist()
        if pred_labels == []:
            continue
        pred_xy = np.array(pred_xy)
        pred_conf = np.array(pred_conf)

        # calc iou
        # [N, V]
        iou_matrix = calc_iou(pred_xy, true_boxes)
        # [N]
        max_iou_idx = np.argmax(iou_matrix, axis=-1)

        correct_idx = []
        correct_conf = []
        for k in range(max_iou_idx.shape[0]):
            pred_labels_dict[pred_labels[k]] += 1
            match_idx = max_iou_idx[k]  # V level
            if iou_matrix[k, match_idx] > iou_thresh and true_labels_list[match_idx] == pred_labels[k]:
                if match_idx not in correct_idx:
                    correct_idx.append(match_idx)
                    correct_conf.append(pred_conf[k])
                else:
                    same_idx = correct_idx.index(match_idx)
                    if pred_conf[k] > correct_conf[same_idx]:
                        correct_idx.pop(same_idx)
                        correct_conf.pop(same_idx)
                        correct_idx.append(match_idx)
                        correct_conf.append(pred_conf[k])

        for t in correct_idx:
            true_positive_dict[true_labels_list[t]] += 1

    recall = sum(true_positive_dict.values()) / (sum(true_labels_dict.values()) + 1e-6)
    precision = sum(true_positive_dict.values()) / (sum(pred_labels_dict.values()) + 1e-6)

    return recall, precision



def calc_iou(pred_boxes, true_boxes):
    '''
    Maintain an efficient way to calculate the ios matrix using the numpy broadcast tricks.
    shape_info: pred_boxes: [N, 4]
                true_boxes: [V, 4]
    return: IoU matrix: shape: [N, V]
    '''

    # [N, 1, 4]
    pred_boxes = np.expand_dims(pred_boxes, -2)
    # [1, V, 4]
    true_boxes = np.expand_dims(true_boxes, 0)

    # [N, 1, 2] & [1, V, 2] ==> [N, V, 2]
    intersect_mins = np.maximum(pred_boxes[..., :2], true_boxes[..., :2])
    intersect_maxs = np.minimum(pred_boxes[..., 2:], true_boxes[..., 2:])
    intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)

    # shape: [N, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # shape: [N, 1, 2]
    pred_box_wh = pred_boxes[..., 2:] - pred_boxes[..., :2]
    # shape: [N, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    # [1, V, 2]
    true_boxes_wh = true_boxes[..., 2:] - true_boxes[..., :2]
    # [1, V]
    true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]
    # shape: [N, V]
    union_area = pred_box_area + true_boxes_area - intersect_area

    # shape: [N, V]
    #iou = np.divide(intersect_area, union_area,\
    #                     out=np.zeros_like(union_area.shape, dtype=np.float32),\
    #                     where=(union_area!=0))
    iou = intersect_area / (pred_box_area + true_boxes_area - intersect_area + 1e-10)

    return iou

# Author: Zhili Zhang
# Date: Sept 12, 2019

import tensorflow as tf 
from core.layers import convolutional
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

    return raw_output, inputs


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

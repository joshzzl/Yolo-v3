
# Author: Zhili Zhang
# Date: Sept 12, 2019

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1


def _padding(inputs, kernel_size, data_format):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        return tf.pad(inputs, [[0,0], [0,0], 
            [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        return tf.pad(inputs, [[0,0], [pad_beg, pad_end], 
            [pad_beg, pad_end], [0,0]])

def batch_norm(inputs, trainable, data_format):
    if data_format=='channels_first':
        ax = 1
    else:
        ax = 3
    return tf.layers.batch_normalization(inputs, axis=ax, momentum=_BATCH_NORM_DECAY, 
        epsilon=_BATCH_NORM_EPSILON, scale=True, trainable=True, training=trainable)

'''
def batch_norm(inputs, trainable, data_format):
    if data_format=='channels_first':
        ax = 1
    else:
        ax = 3
    return BatchNormalization(axis=ax, momentum=_BATCH_NORM_DECAY, 
        epsilon=_BATCH_NORM_EPSILON, scale=True)(inputs=inputs, training=trainable)
'''

def convolutional(inputs, filters, kernel_size, trainable, name, strides=1, 
        data_format='channels_first', act=True, bn=True):
    
    with tf.variable_scope(name):
        if strides > 1:
            inputs = _padding(inputs, kernel_size, data_format)
            padding = 'VALID'
        else:
            padding = 'SAME'

        conv = Conv2D(filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding, 
            use_bias=False, data_format=data_format)(inputs=inputs)

        if bn:
            conv = batch_norm(conv, trainable, data_format)
        else:
            bias = tf.get_variable(name='bias', shape=filters, trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)


        if act:
            conv = tf.nn.leaky_relu(conv, alpha=_LEAKY_RELU)

    return conv


def residual(inputs, filters, trainable, data_format, name, strides=1):
    shortcut = inputs

    with tf.variable_scope(name):
        inputs = convolutional(inputs=inputs, filters=filters, kernel_size=1,
            trainable=trainable, name='conv1', strides=strides, data_format=data_format)
        inputs = convolutional(inputs=inputs, filters=2*filters, kernel_size=3,
            trainable=trainable, name='conv2', strides=strides, data_format=data_format)
        residual_out = inputs + shortcut

    return residual_out
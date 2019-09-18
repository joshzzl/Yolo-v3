
# Author: Zhili Zhang
# Date: Sept 12, 2019

import tensorflow as tf 
from core.layers import convolutional, residual


def darknet53(inputs, trainable, data_format):

    with tf.variable_scope('darknet'):
        inputs = convolutional(inputs=inputs, filters=32, kernel_size=3, 
            trainable=trainable, name='conv0', data_format=data_format)

        inputs = convolutional(inputs=inputs, filters=64, kernel_size=3,
            trainable=trainable, name='conv1', strides=2, data_format=data_format)

        for i in range(1):
            inputs = residual(inputs=inputs, filters=32, trainable=trainable,
                data_format=data_format, name='residual%d' %(i+0))

        inputs = convolutional(inputs=inputs, filters=128, kernel_size=3, 
            trainable=trainable, name='conv4', strides=2, data_format=data_format)

        for i in range(2):
            inputs = residual(inputs=inputs, filters=64, trainable=trainable,
                data_format=data_format, name='residual%d' %(i+1))

        inputs = convolutional(inputs=inputs, filters=256, kernel_size=3, 
            trainable=trainable, name='conv9', strides=2, data_format=data_format)

        for i in range(8):
            inputs = residual(inputs=inputs, filters=128, trainable=trainable,
                data_format=data_format, name='residual%d' %(i+3))

        route1 = inputs

        inputs = convolutional(inputs=inputs, filters=512, kernel_size=3, 
            trainable=trainable, name='conv26', strides=2, data_format=data_format)

        for i in range(8):
            inputs = residual(inputs=inputs, filters=256, trainable=trainable,
                data_format=data_format, name='residual%d' %(i+11))

        route2 = inputs

        inputs = convolutional(inputs=inputs, filters=1024, kernel_size=3, 
            trainable=trainable, name='conv43', strides=2, data_format=data_format)

        for i in range(4):
            inputs = residual(inputs=inputs, filters=512, trainable=trainable,
                data_format=data_format, name='residual%d' %(i+19))

        return route1, route2, inputs
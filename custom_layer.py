# -*- coding: utf-8 -*-
from keras import *
from keras.layers import *
import tensorflow as tf
kernel_regularizer = regularizers.l2(1e-5)
bias_regularizer = regularizers.l2(1e-5)

def conv_(inputs, filter_num, kernel_size=(3, 3), strides=(1,1)):
    conv_ = Conv2D(
        filters=filter_num,
        kernel_size=kernel_size,
        strides=strides,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=kernel_regularizer
    )(inputs)
    conv_ = BatchNormalization()(conv_)
    return conv_

def dilate_conv(inputs, filter_num, dilation_rate):
    conv_ = Conv2D(
        filters=filter_num,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        dilation_rate=dilation_rate,
        kernel_initializer='he_normal',
        kernel_regularizer=kernel_regularizer
    )(inputs)
    conv_ = BatchNormalization()(conv_)
    return conv_

def concat_dwt(conv, dwt, filter_num, strides=(2, 2)):
    conv_downsample = conv_(conv, filter_num, (3, 3), strides=strides)
    concat_dwt_ = Concatenate()([conv_downsample, dwt])
    return concat_dwt_

class BilinearUpsampling(Layer):
    def __init__(self, upsampling=(2, 2), **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)       
        self.upsampling = upsampling
        
    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
                 input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs, **kwargs):
        return tf.image.resize(inputs, 
                               (int(inputs.shape[1] * self.upsampling[0]),
                                int(inputs.shape[2] * self.upsampling[1])),
                               method=tf.image.ResizeMethod.BILINEAR)



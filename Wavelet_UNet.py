from keras.optimizers import Adam
import keras.backend as K
from custom_layer import *
import tensorflow as tf
from DWT import DWT_Pooling, IWT_UpSampling

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)
    
def Wavelet_UNet(input_shape=(256, 256, 3), num_class=1):
    inputs = Input(shape=input_shape)

    conv1 = conv_(inputs, 32)
    conv1 = conv_(conv1, 32)
    conv1 = conv_(conv1, 32)
    dwt1 = DWT_Pooling()(conv1)
    concat_dwt11 = concat_dwt(conv1, dwt1, 32, strides=(2, 2))
    fusion1 = conv_(concat_dwt11, 64 * 4, (1, 1))

    conv2 = conv_(fusion1, 64)
    conv2 = conv_(conv2, 64)
    conv2 = conv_(conv2, 64)
    dwt2 = DWT_Pooling()(conv2)
    concat_dwt12 = concat_dwt(conv1, dwt2, 64, strides=(4, 4))
    concat_dwt22 = concat_dwt(conv2, concat_dwt12, 64, strides=(2, 2))
    fusion2 = conv_(concat_dwt22, 128 * 4, (1, 1))

    conv3 = conv_(fusion2, 128)
    conv3 = conv_(conv3, 128)
    conv3 = conv_(conv3, 128)
    dwt3 = DWT_Pooling()(conv3)
    concat_dwt13 = concat_dwt(conv1, dwt3, 128, strides=(8, 8))
    concat_dwt23 = concat_dwt(conv2, concat_dwt13, 128, strides=(4, 4))
    concat_dwt33 = concat_dwt(conv3, concat_dwt23, 128, strides=(2, 2))
    fusion3 = conv_(concat_dwt33, 256 * 4, (1, 1))

    conv4 = conv_(fusion3, 256)
    conv4 = conv_(conv4, 256)
    conv4 = conv_(conv4, 256)
    dwt4 = DWT_Pooling()(conv4)
    concat_dwt14 = concat_dwt(conv1, dwt4, 256, strides=(16, 16))
    concat_dwt24 = concat_dwt(conv2, concat_dwt14, 256, strides=(8, 8))
    concat_dwt34 = concat_dwt(conv3, concat_dwt24, 256, strides=(4, 4))
    concat_dwt44 = concat_dwt(conv4, concat_dwt34, 256, strides=(2, 2))
    fusion4 = conv_(concat_dwt44, 512 * 4, (1, 1))

    conv5 = conv_(fusion4, 512, (3, 3))
    conv5 = Dropout(0.5)(conv5)

    clf_aspp = CLF_ASPP(conv5, conv1, conv2, conv3, conv4, input_shape)

    up_conv1 = IWT_UpSampling()(clf_aspp)
    skip_conv4 = conv_(conv4, 256, (1, 1))
    context_inference1 = Concatenate()([up_conv1, skip_conv4])
    conv6 = conv_(context_inference1, 256)
    conv6 = conv_(conv6, 256)

    up_conv2 = IWT_UpSampling()(conv6)
    up_conv2 = conv_(up_conv2, 128, (2, 2))
    skip_conv3 = conv_(conv3, 128, (1, 1))
    context_inference2 = Concatenate()([up_conv2, skip_conv3])
    conv7 = conv_(context_inference2, 128)
    conv7 = conv_(conv7,128)

    up_conv3 = IWT_UpSampling()(conv7)
    up_conv3 = conv_(up_conv3, 64, (2, 2))
    skip_conv2 = conv_(conv2, 64, (1, 1))
    context_inference3 = Concatenate()([up_conv3, skip_conv2])
    conv8 = conv_(context_inference3, 64)
    conv8 = conv_(conv8, 64)

    up_conv4 = IWT_UpSampling()(conv8)
    up_conv4 = conv_(up_conv4, 32, (2, 2))
    skip_conv1 = conv_(conv1, 32, (1, 1))
    context_inference4 = Concatenate()([up_conv4, skip_conv1])
    conv9 = conv_(context_inference4, 32)
    conv9 = conv_(conv9, 32)

    if num_class == 1:
        conv10 = Conv2D(num_class, (1, 1), activation='sigmoid')(conv9)
    else:
        conv10 = Conv2D(num_class, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model


def CLF_ASPP(conv5, conv1, conv2, conv3, conv4, input_shape):

    b0 = conv_(conv5, 256, (1, 1))
    b1 = dilate_conv(conv5, 256, dilation_rate=(2, 2))
    b2 = dilate_conv(conv5, 256, dilation_rate=(4, 4))
    b3 = dilate_conv(conv5, 256, dilation_rate=(6, 6))

    out_shape0 = input_shape[0] // pow(2, 4)
    out_shape1 = input_shape[1] // pow(2, 4)
    b4 = AveragePooling2D(pool_size=(out_shape0, out_shape1))(conv5)
    b4 = conv_(b4, 256, (1, 1))
    b4 = BilinearUpsampling((out_shape0, out_shape1))(b4)

    clf1 = conv_(conv1, 256, kernel_size=(3,3), strides=(16, 16))
    clf2 = conv_(conv2, 256, kernel_size=(3,3), strides=(8, 8))
    clf3 = conv_(conv3, 256, kernel_size=(3,3), strides=(4, 4))
    clf4 = conv_(conv4, 256, kernel_size=(3,3), strides=(2, 2))

    outs = Concatenate()([clf1, clf2, clf3, clf4, b0, b1, b2, b3, b4])

    outs = conv_(outs, 256 * 4, (1, 1))
    outs = Dropout(0.5)(outs)

    return outs

if __name__ == '__main__':
    import os
    model = Wavelet_UNet()
    model.summary()
    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])




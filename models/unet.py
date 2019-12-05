"""
U-Net model specification

Taken from https://github.com/zhixuhao/unet
"""

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

def build_model(input_shape, n_classes=3):

    # Input layer
    inputs = layers.Input(input_shape)

    # Common conv layer arguments
    conv_args = dict(activation='relu', padding='same', kernel_initializer='he_normal')

    # Down-sizing convolutional layer blocks
    conv1 = layers.Conv2D(64, 3, **conv_args)(inputs)
    conv1 = layers.Conv2D(64, 3, **conv_args)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, **conv_args)(pool1)
    conv2 = layers.Conv2D(128, 3, **conv_args)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, **conv_args)(pool2)
    conv3 = layers.Conv2D(256, 3, **conv_args)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, **conv_args)(pool3)
    conv4 = layers.Conv2D(512, 3, **conv_args)(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(1024, 3, **conv_args)(pool4)
    conv5 = layers.Conv2D(1024, 3, **conv_args)(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    # Up-sampling convolutional layer blocks with shortcuts
    up6 = layers.Conv2D(512, 2, **conv_args)(layers.UpSampling2D(size=(2,2))(drop5))
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, **conv_args)(merge6)
    conv6 = layers.Conv2D(512, 3, **conv_args)(conv6)

    up7 = layers.Conv2D(256, 2, **conv_args)(layers.UpSampling2D(size=(2,2))(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, **conv_args)(merge7)
    conv7 = layers.Conv2D(256, 3, **conv_args)(conv7)

    up8 = layers.Conv2D(128, 2, **conv_args)(layers.UpSampling2D(size=(2,2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, **conv_args)(merge8)
    conv8 = layers.Conv2D(128, 3, **conv_args)(conv8)

    up9 = layers.Conv2D(64, 2, **conv_args)(layers.UpSampling2D(size=(2,2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, **conv_args)(merge9)
    conv9 = layers.Conv2D(64, 3, **conv_args)(conv9)
    #conv9 = layers.Conv2D(2, 3, **conv_args)(conv9)

    conv10 = layers.Conv2D(n_classes, 1, activation='softmax')(conv9)

    return models.Model(inputs=inputs, outputs=conv10)

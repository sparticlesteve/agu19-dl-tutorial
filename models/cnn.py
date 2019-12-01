"""Simple CNN model definition"""

import tensorflow as tf
import tensorflow.keras.layers as layers

def build_model(input_shape, n_classes=3):
    """Construct the Keras model"""
    model = tf.keras.models.Sequential([
        layers.Conv2D(16, kernel_size=3, padding='same', activation='relu',
                      input_shape=input_shape),
        layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        layers.Conv2D(n_classes, kernel_size=1, activation='softmax')
    ])
    return model

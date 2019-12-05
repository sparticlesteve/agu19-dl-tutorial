"""TF Keras metrics for segmentation"""

from tensorflow.python.keras.metrics import MeanMetricWrapper, MeanIoU
import tensorflow.keras.backend as K

class PixelIoU(MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(PixelIoU, self).update_state(
            y_true=y_true,
            y_pred=K.cast(K.argmax(y_pred, axis=-1), y_true.dtype),
            sample_weight=sample_weight)

class PixelAccuracy(MeanMetricWrapper):
    def __init__(self, name='accuracy', dtype=None):
        super(PixelAccuracy, self).__init__(pixel_accuracy, name, dtype=dtype)

def pixel_accuracy(y_true, y_pred):
    # Convert prediction into labels by choosing the highest-scored class
    y_pred = K.cast(K.argmax(y_pred, axis=-1), y_true.dtype)
    return K.mean(K.equal(K.flatten(y_true), K.flatten(y_pred)))

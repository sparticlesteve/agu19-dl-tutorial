"""
Hardware/device configuration
"""

# System
import os

# Externals
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, Session, keras

def configure_session(intra_threads=4, inter_threads=2,
                      blocktime=1, affinity='granularity=fine,compact,1,0',
                      gpu=None):
    """Sets the thread knobs in the TF backend"""

    os.environ['KMP_BLOCKTIME'] = str(blocktime)
    os.environ['KMP_AFFINITY'] = affinity
    os.environ['OMP_NUM_THREADS'] = str(intra_threads)
    config = ConfigProto(
        inter_op_parallelism_threads=inter_threads,
        intra_op_parallelism_threads=intra_threads
    )
    if gpu is not None:
        config.gpu_options.visible_device_list = str(gpu)
    keras.backend.set_session(Session(config=config))

    # For some reason TF is trying to put things on the wrong GPUs
    # unless we also specify device this way. Seems like a bug.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')

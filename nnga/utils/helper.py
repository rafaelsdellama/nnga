# https://forums.fast.ai/t/gpu-memory-not-being-freed-after-training-is-over/10265/6
import tensorflow as tf
# from keras import backend as k
import gc
from tensorflow.python.framework import ops


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    ops.reset_default_graph()
    # k.clear_session()
    tf.keras.backend.clear_session()
    print(gc.collect())
    gc.collect()

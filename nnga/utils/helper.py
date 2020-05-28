import tensorflow as tf
import gc


def dump_tensors():
    """
    https://forums.fast.ai/t/gpu-memory-not-being-freed-after-training-is-over/10265/6
    """
    tf.keras.backend.clear_session()
    gc.collect()

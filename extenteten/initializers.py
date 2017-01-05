import numpy as np
import tensorflow as tf


def identity_initializer(dtype=tf.float32):
    def initializer(shape, dtype=dtype):
        if len(shape) == 1:
            return tf.zeros(shape, dtype)
        elif len(shape) == 2 and shape[0] == shape[1]:
            return tf.constant(np.eye(shape[0]), dtype)
        raise ValueError("Invalid shape for identity_initializer.")
    return initializer

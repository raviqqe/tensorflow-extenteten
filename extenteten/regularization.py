import tensorflow as tf

from .util import func_scope


@func_scope()
def l2_regularization_loss(scale=1e-8):
    return tf.contrib.layers.apply_regularization(
        tf.contrib.layers.l2_regularizer(scale))

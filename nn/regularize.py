import tensorflow as tf

from .util import func_scope



@func_scope()
def regularize_with_l2_loss(scale):
  return tf.contrib.layers.apply_regularization(
      tf.contrib.layers.l2_regularizer(scale))

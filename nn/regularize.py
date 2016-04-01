import tensorflow as tf


def regularize_with_l2_loss(scale):
  return scale * tf.reduce_sum(tf.pack([
      tf.nn.l2_loss(weight)
      for weight in tf.get_collection(tf.GraphKeys.WEIGHTS)]))

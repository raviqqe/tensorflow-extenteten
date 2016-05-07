import tensorflow as tf



def variable(shape):
  return tf.Variable(_normal(shape))


def _normal(shape):
  return tf.truncated_normal(shape, stddev=0.1)

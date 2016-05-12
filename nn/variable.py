import tensorflow as tf



def variable(shape, name=None):
  return tf.Variable(_normal(shape), name=name)


def _normal(shape):
  return tf.truncated_normal(shape, stddev=0.1)

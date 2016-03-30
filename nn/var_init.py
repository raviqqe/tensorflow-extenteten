import tensorflow as tf



def normal(shape):
  return tf.truncated_normal(shape, stddev=0.1)

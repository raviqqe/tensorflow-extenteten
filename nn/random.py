import tensorflow as tf

from .util import static_shape



def sample_crop(xs, n):
  return tf.random_crop(xs, [n, *static_shape(xs)[1:]])

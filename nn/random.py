import tensorflow as tf

from .assertion import is_natural_num
from .util import funcname_scope, static_shape



@funcname_scope
def sample_crop(xs, n):
  assert is_natural_num(n)
  return tf.random_crop(
      xs,
      [tf.minimum(n, tf.shape(xs)[0]), *static_shape(xs)[1:]])

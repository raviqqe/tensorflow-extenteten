import tensorflow as tf

from .assertion import is_natural_num
from .util import func_scope, static_shape


__all__ = ['sample_crop']


@func_scope()
def sample_crop(xs, n):
    return tf.random_crop(
        xs,
        tf.concat([[tf.minimum(n, tf.shape(xs)[0])], tf.shape(xs)[1:]], 0))

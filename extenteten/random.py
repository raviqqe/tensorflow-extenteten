import tensorflow as tf

from .util import func_scope


__all__ = ['sample_crop']


@func_scope()
def sample_crop(xs, n):
    return tf.random_crop(
        xs,
        tf.concat([[tf.minimum(n, tf.shape(xs)[0])], tf.shape(xs)[1:]], 0))

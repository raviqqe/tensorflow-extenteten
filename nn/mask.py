import tensorflow as tf

from .util import static_rank, funcname_scope, dimension_indices



@funcname_scope
def mask(length, max_length):
  """ That's a spicy meat ball! """
  assert static_rank(length) == 1

  return tf.to_int32(tf.less(
      _range(max_length, batch_size=tf.shape(length)[0]),
      _tile_length(length, max_length)))


@funcname_scope
def _tile_length(length, max_length):
  return tf.tile(tf.transpose(tf.expand_dims(length, 0)), [1, max_length])


@funcname_scope
def _range(limit, *, batch_size):
  return tf.tile(tf.expand_dims(tf.range(limit), 0), [batch_size, 1])


@funcname_scope
def max_mask(x, reduction_indices=None):
  assert static_rank(x) >= 2
  max_value = tf.reduce_max(x,
                            reduction_indices or dimension_indices(x, 1),
                            keep_dims=True)
  return x * tf.cast(tf.equal(x, max_value), x.dtype)


@funcname_scope
def mean_mask(x, reduction_indices=None):
  assert static_rank(x) >= 2
  mean_value = tf.reduce_mean(x,
                              reduction_indices or dimension_indices(x, 1),
                              keep_dims=True)
  return x * tf.cast(tf.greater_equal(x, mean_value), x.dtype)

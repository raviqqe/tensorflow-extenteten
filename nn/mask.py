import tensorflow as tf

from .util import static_rank, func_scope, dimension_indices
from .flags import FLAGS



@func_scope
def mask(length, max_length, dtype=tf.bool):
  assert static_rank(length) == 1
  return tf.sequence_mask(length, max_length, dtype)


@func_scope
def max_mask(x, reduction_indices=None):
  assert static_rank(x) >= 2
  max = tf.reduce_max(x,
                      reduction_indices or dimension_indices(x, 1),
                      keep_dims=True)
  return tf.cast(tf.equal(x, max), x.dtype)


@func_scope
def mean_mask(x, reduction_indices=None):
  assert static_rank(x) >= 2
  mean = tf.reduce_mean(x,
                        reduction_indices or dimension_indices(x, 1),
                        keep_dims=True)
  return tf.cast(tf.greater_equal(x, mean), x.dtype)


@func_scope
def entity_index_mask(x, dtype=tf.bool):
  return tf.cast(tf.logical_and(x >= FLAGS.first_entity_index,
                                x <= FLAGS.last_entity_index),
                 dtype)

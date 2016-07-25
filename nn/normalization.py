import tensorflow as tf

from .util import funcname_scope, static_rank, dimension_indices
from .variable import variable



@funcname_scope
def layer_normalization(x: ("batch", ...)):
  assert static_rank(x) >= 2
  return (x - _batchwise_mean(x)) / _stddev(x) * variable([], "stddev")


@funcname_scope
def _batchwise_mean(x):
  return tf.reduce_mean(x, dimension_indices(x, 1), keep_dims=True)


@funcname_scope
def _stddev(x):
  return tf.sqrt(tf.reduce_sum(tf.square(x - _batchwise_mean(x)),
                               dimension_indices(x, 1),
                               keep_dims=True))

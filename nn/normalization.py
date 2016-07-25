import functools
import tensorflow as tf

from .util import funcname_scope, static_shape, static_rank, dimension_indices
from .variable import variable



@funcname_scope
def layer_normalization(x: ("batch", ...), add_bias=True):
  assert static_rank(x) >= 2
  param_shape = [1, *static_shape(x)[1:]]
  a = (x - _batchwise_mean(x)) \
      / _stddev(x) * variable(tf.ones(param_shape), "gain")
  bias = variable(tf.zeros(param_shape), "bias")
  tf.add_to_collection(tf.GraphKeys.BIASES, bias)
  return a + bias if add_bias else a


@funcname_scope
def _batchwise_mean(x):
  return tf.reduce_mean(x, dimension_indices(x, 1), keep_dims=True)


@funcname_scope
def _stddev(x):
  return tf.sqrt(tf.reduce_mean(tf.square(x - _batchwise_mean(x)),
                                dimension_indices(x, 1),
                                keep_dims=True))

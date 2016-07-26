import functools
import tensorflow as tf

from .util import funcname_scope, static_shape, static_rank, dimension_indices
from .variable import variable



_GAIN = "gain"
_BIAS = "bias"


@funcname_scope
def layer_normalization(x: ("batch", ...),
                        add_bias=True,
                        share_variables=False):
  assert static_rank(x) >= 2

  shape = [1, *static_shape(x)[1:]]
  gain = (tf.get_variable(_GAIN, shape, initializer=tf.ones_initializer)
          if share_variables else
          variable(tf.ones(shape), _GAIN))
  bias = (tf.get_variable(_BIAS, shape, initializer=tf.zeros_initializer)
          if share_variables else
          variable(tf.zeros(shape), _BIAS))
  a = (x - _batchwise_mean(x)) / (_stddev(x) * gain)

  return a + bias if add_bias else a


@funcname_scope
def _batchwise_mean(x):
  return tf.reduce_mean(x, dimension_indices(x, 1), keep_dims=True)


@funcname_scope
def _stddev(x):
  return tf.sqrt(tf.reduce_mean(tf.square(x - _batchwise_mean(x)),
                                dimension_indices(x, 1),
                                keep_dims=True))

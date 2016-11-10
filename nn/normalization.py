import functools
import tensorflow as tf

from .util import func_scope, static_shape, static_rank, dimension_indices
from .variable import variable



_GAIN = "gain"
_BIAS = "bias"
_EPSILON = 1e-5 # refer to the implementation of batch norm' in Chainer


@func_scope
def layer_normalization(x: ("batch", ...),
                        add_bias=True,
                        share_variables=False):
  assert static_rank(x) >= 2

  shape = [1, *static_shape(x)[1:]]
  scale = _inverted_stddev(x) * _gain(shape, share_variables)

  # refer to the implementation of tf.nn.batch_normalization()
  a = x * scale - _batchwise_mean(x) * scale

  return a + _bias(shape, share_variables) if add_bias else a


def _gain(shape, share_variables):
  return (tf.get_variable(_GAIN, shape, initializer=tf.ones_initializer)
          if share_variables else
          variable(tf.ones(shape), _GAIN))


def _bias(shape, share_variables):
  return (tf.get_variable(_BIAS, shape, initializer=tf.zeros_initializer)
          if share_variables else
          variable(tf.zeros(shape), _BIAS))


@func_scope
def _batchwise_mean(x):
  return tf.reduce_mean(x, dimension_indices(x, 1), keep_dims=True)


@func_scope
def _inverted_stddev(x):
  return tf.rsqrt(_variance(x) + _EPSILON)


@func_scope
def _variance(x):
  return tf.reduce_mean(tf.square(x - _batchwise_mean(x)),
                        dimension_indices(x, 1),
                        keep_dims=True)

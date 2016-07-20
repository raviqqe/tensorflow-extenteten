import functools
import tensorflow as tf

from .util import funcname_scope, static_rank, static_shape
from .variable import variable
from .assertion import is_natural_num, is_natural_num_sequence



@funcname_scope
def multi_conv_and_pool(x,
                        *,
                        nums_of_channels,
                        conv_kernel_shape,
                        pool_kernel_shape):
  assert is_natural_num_sequence(nums_of_channels)
  assert _is_kernel_shape(conv_kernel_shape)
  assert _is_kernel_shape(pool_kernel_shape)

  def layer(x, num_of_channels):
    return max_pool(tf.tanh(conv2d(x, conv_kernel_shape, num_of_channels)),
                    kernel_shape=pool_kernel_shape)

  return functools.reduce(layer, nums_of_channels, x)


@funcname_scope
def multi_conv(x,
               *,
               nums_of_channels,
               kernel_shape):
  assert is_natural_num_sequence(nums_of_channels)
  assert _is_kernel_shape(kernel_shape)

  def layer(x, num_of_channels):
    return tf.tanh(conv2d(x, kernel_shape, num_of_channels))

  return functools.reduce(layer, nums_of_channels, x)


@funcname_scope
def conv2d(x, kernel_shape, num_of_channels):
  assert static_rank(x) == 4
  assert _is_kernel_shape(kernel_shape)
  assert is_natural_num(num_of_channels)

  return tf.nn.conv2d(
      x,
      variable(list(kernel_shape) + [static_shape(x)[-1], num_of_channels],
               name="kernel"),
      strides=[1, 1, 1, 1],
      padding="SAME")


@funcname_scope
def max_pool(x, kernel_shape):
  assert _is_kernel_shape(kernel_shape)

  strides = [1] + list(kernel_shape) + [1]
  return tf.nn.max_pool(x, ksize=strides, strides=strides, padding="SAME")


def _is_kernel_shape(shape):
  return is_natural_num_sequence(shape, 2)

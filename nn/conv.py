import functools
import tensorflow as tf

from .util import funcname_scope, static_rank, static_shape
from .variable import variable
from .assertion import check_natural_num



@funcname_scope
def lenet(x,
          nums_of_channels=[20, 50],
          conv_kernel_shape=[5, 5],
          pool_kernel_shape=[2, 2]):
  def layer(x, num_of_channels):
    c = tf.tanh(conv2d(x,
                       kernel_shape=conv_kernel_shape,
                       num_of_channels=num_of_channels))
    return max_pool(c, kernel_shape=pool_kernel_shape)

  return functools.reduce(layer, nums_of_channels, x)


@funcname_scope
def conv2d(x, kernel_shape, num_of_channels):
  assert static_rank(x) == 4
  assert _check_kernel_shape(kernel_shape)
  assert check_natural_num(num_of_channels)

  return tf.nn.conv2d(
      x,
      variable(list(kernel_shape) + [static_shape(x)[-1], num_of_channels],
               name="kernel"),
      strides=[1, 1, 1, 1],
      padding="SAME")


@funcname_scope
def max_pool(x, kernel_shape):
  assert _check_kernel_shape(kernel_shape)

  strides = [1] + list(kernel_shape) + [1]
  return tf.nn.max_pool(x, ksize=strides, strides=strides, padding="SAME")


def _check_kernel_shape(kernel_shape):
  return len(kernel_shape) == 2 and \
         all(check_natural_num(size) for size in kernel_shape)

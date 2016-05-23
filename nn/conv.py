import functools
import tensorflow as tf

from .util import funcname_scope
from .variable import variable
from .assertion import check_natural_num



@funcname_scope
def lenet(x):
  conv = functools.partial(conv2d, kernel_shape=[5, 5])
  pool = functools.partial(max_pool, kernel_shape=[2, 2])
  activate = tf.tanh

  h = activate(conv(x, num_of_output_channels=20))
  h = pool(h)
  h = activate(conv(h, num_of_output_channels=50))
  return pool(h)


def conv2d(x, kernel_shape, num_of_output_channels):
  assert _check_kernel_shape(kernel_shape)
  assert check_natural_num(num_of_output_channels)

  return tf.nn.conv2d(
      x,
      variable(list(kernel_shape) + [None, num_of_output_channels],
               name="kernel"),
      strides=[1, 1, 1, 1],
      padding="SAME")


def max_pool(x, kernel_shape):
  assert _check_kernel_shape(kernel_shape)

  strides = [1] + list(kernel_shape) + [1]
  return tf.nn.max_pool(x, ksize=strides, strides=strides, padding="SAME")


def _check_kernel_shape(kernel_shape):
  return len(kernel_shape) == 2 and \
         all(check_natural_num(size) for size in kernel_shape)

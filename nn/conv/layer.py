import tensorflow as tf

from ..util import funcname_scope, static_rank, static_shape
from ..variable import variable
from ..summary import image_summary, num_of_summary_images
from ..assertion import is_natural_num
from .assertion import is_kernel_shape




@funcname_scope
def conv2d(x, kernel_shape, num_of_channels):
  assert static_rank(x) == 4
  assert is_kernel_shape(kernel_shape)
  assert is_natural_num(num_of_channels)

  filter_ = variable([*kernel_shape, static_shape(x)[-1], num_of_channels],
                     name="filter")
  _summarize_filter(filter_)

  return tf.nn.conv2d(
      x,
      filter_,
      strides=[1, 1, 1, 1],
      padding="SAME")


@funcname_scope
def max_pool(x, kernel_shape):
  assert is_kernel_shape(kernel_shape)

  strides = [1, *kernel_shape, 1]
  return tf.nn.max_pool(x, ksize=strides, strides=strides, padding="SAME")


def _summarize_filter(filter_):
  image_summary(tf.transpose(
      filter_[:, :, 0, :min(static_shape(filter_)[-1], num_of_summary_images)],
      [2, 0, 1]))

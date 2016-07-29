import tensorflow as tf

from ..invertible import InvertibleLayer
from ..util import funcname_scope, static_rank, static_shape
from ..variable import variable
from ..summary import image_summary, num_of_summary_images
from ..assertion import is_natural_num
from .assertion import is_kernel_shape



_DEFAULT_CONV_STRIDES = [1, 1, 1, 1]
_DEFAULT_PADDING = "SAME"


@funcname_scope
def conv2d(x, kernel_shape, num_of_channels):
  assert static_rank(x) == 4
  assert is_kernel_shape(kernel_shape)
  assert is_natural_num(num_of_channels)

  return _conv2d_with_filter(
      x,
      _create_filter(kernel_shape, static_shape(x)[-1], num_of_channels))


def _create_filter(kernel_shape,
                   num_of_input_channels,
                   num_of_output_channels):
  filter_ = variable(
      [*kernel_shape, num_of_input_channels, num_of_output_channels],
      name="filter")
  _summarize_filter(filter_)
  return filter_


@funcname_scope
def _conv2d_with_filter(x, filter_):
  return tf.nn.conv2d(x, filter_, _DEFAULT_CONV_STRIDES, _DEFAULT_PADDING)


@funcname_scope
def max_pool(x, kernel_shape):
  assert is_kernel_shape(kernel_shape)

  kernel_shape = [1, *kernel_shape, 1]
  return tf.nn.max_pool(x, kernel_shape, kernel_shape, _DEFAULT_PADDING)


def _summarize_filter(filter_):
  image_summary(tf.transpose(
      filter_[:, :, 0, :min(static_shape(filter_)[-1], num_of_summary_images)],
      [2, 0, 1]))


class InvertibleReLU(InvertibleLayer):
  def __init__(self):
    pass

  def forward(self, x):
    return tf.nn.relu(x)

  def backward(self, x):
    return tf.nn.relu(x)


class InvertibleConv2d(InvertibleLayer):
  def __init__(self, kernel_shape, num_of_channels):
    assert is_kernel_shape(kernel_shape), kernel_shape
    assert is_natural_num(num_of_channels), num_of_channels
    self._kernel_shape = kernel_shape
    self._num_of_channels = num_of_channels

  def forward(self, x):
    self._input_shape = tf.shape(x)
    self._filter =  _create_filter(self._kernel_shape,
                                   static_shape(x)[-1],
                                   self._num_of_channels)
    return _conv2d_with_filter(x, self._filter)

  def backward(self, x):
    return tf.nn.conv2d_transpose(
        x,
        self._filter,
        output_shape=tf.concat(0, [tf.shape(x)[:1], self._input_shape[1:]]),
        strides=_DEFAULT_CONV_STRIDES)


class InvertibleMaxPool(InvertibleLayer):
  def __init__(self, kernel_shape):
    assert is_kernel_shape(kernel_shape)
    self._kernel_shape = [1, *kernel_shape, 1]

  def forward(self, x):
    h, self._max_indices = tf.nn.max_pool_with_argmax(x,
                                                      self._kernel_shape,
                                                      self._kernel_shape,
                                                      _DEFAULT_PADDING)
    return h

  def backward(self, x):
    return _max_unpool(x, self._max_indices, self._kernel_shape)


@funcname_scope
def _max_unpool(x, max_indices, kernel_shape):
  return NotImplemented

import functools
import tensorflow as tf

from ..util import funcname_scope
from ..random import sample_crop
from ..summary import image_summary, num_of_summary_images
from ..assertion import is_natural_num_sequence
from .assertion import is_kernel_shape
from .layer import conv2d, max_pool



ACTIVATE = tf.nn.relu


@funcname_scope
def multi_conv_and_pool(x,
                        *,
                        nums_of_channels,
                        conv_kernel_shape,
                        pool_kernel_shape):
  assert is_natural_num_sequence(nums_of_channels)
  assert is_kernel_shape(conv_kernel_shape)
  assert pool_kernel_shape is None or is_kernel_shape(pool_kernel_shape)

  @funcname_scope
  def layer(x, num_of_channels):
    h = ACTIVATE(conv2d(x, conv_kernel_shape, num_of_channels))
    image_summary(tf.transpose(
        sample_crop(h, 1)[0, :, :, :num_of_summary_images],
        [2, 0, 1]))
    return h if pool_kernel_shape is None else \
           max_pool(h, pool_kernel_shape)

  return functools.reduce(layer, nums_of_channels, x)


@funcname_scope
def multi_conv(x,
               *,
               nums_of_channels,
               kernel_shape):
  return multi_conv_and_pool(x,
                             nums_of_channels=nums_of_channels,
                             conv_kernel_shape=kernel_shape,
                             pool_kernel_shape=None)

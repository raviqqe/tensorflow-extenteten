import tensorflow as tf

from ..util import func_scope, static_rank, static_shape
from ..variable import variable
from ..assertion import is_natural_num, is_natural_num_sequence


__all__ = ['conv2d', 'max_pool']


_DEFAULT_PADDING = "SAME"


@func_scope()
def conv2d(x, kernel_shape, num_channels):
    assert static_rank(x) == 4
    assert _is_kernel_shape(kernel_shape)
    assert is_natural_num(num_channels)

    return tf.nn.conv2d(
        x,
        variable([*kernel_shape, static_shape(x)[-1], num_channels], "kernel"),
        [1, 1, 1, 1],
        _DEFAULT_PADDING)


@func_scope()
def max_pool(x, kernel_shape):
    assert _is_kernel_shape(kernel_shape)
    return tf.nn.max_pool(x, *(2 * [1, *kernel_shape, 1]), _DEFAULT_PADDING)


def _is_kernel_shape(shape):
    return is_natural_num_sequence(shape, 2)

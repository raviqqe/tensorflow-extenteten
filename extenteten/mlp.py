import functools
import tensorflow as tf

from .assertion import is_natural_num_sequence
from .util import func_scope
from . import layer


__all__ = ['mlp']


@func_scope()
def mlp(x, *, layer_sizes, dropout_keep_prob=1, activate=tf.nn.elu):
    assert is_natural_num_sequence(layer_sizes)

    return layer.linear(
        functools.reduce(functools.partial(layer.fully_connected,
                                           dropout_keep_prob=dropout_keep_prob,
                                           activate=activate),
                         layer_sizes[:-1],
                         x),
        layer_sizes[-1])

import functools
import tensorflow as tf

from .assertion import is_natural_num_sequence
from .util import static_rank, funcname_scope
from .variable import variable
from .layer import fully_connected



@funcname_scope
def mlp(x, *, layer_sizes, dropout_prob, activate=tf.nn.elu):
  assert static_rank(x) == 2
  assert is_natural_num_sequence(layer_sizes)

  def activated_fully_connected(x, output_layer_size):
    return fully_connected(
        x,
        output_layer_size=output_layer_size,
        activate=activate,
        dropout_prob=dropout_prob)

  return fully_connected(
      functools.reduce(activated_fully_connected, layer_sizes[:-1], x),
      output_layer_size=layer_sizes[-1],
      dropout_prob=0)

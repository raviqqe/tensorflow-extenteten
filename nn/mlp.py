import functools
import tensorflow as tf

from .util import static_rank, funcname_scope
from .variable import variable
from .layer import fully_connected



@funcname_scope
def mlp(x, *, layer_sizes, dropout_prob, activate=tf.nn.elu):
  assert static_rank(x) == 2
  assert all(isinstance(size, int) for size in layer_sizes)

  fully_connected_ = functools.partial(fully_connected,
                                       dropout_prob=dropout_prob)

  def activated_fully_connected(x, output_layer_size):
    return fully_connected_(
        x,
        output_layer_size=output_layer_size,
        activate=activate)

  return fully_connected_(
      functools.reduce(activated_fully_connected, layer_sizes[:-1], x),
      output_layer_size=layer_sizes[-1])

import functools
import tensorflow as tf

from .util import static_shape, funcname_scope
from .variable import variable
from .dropout import dropout



@funcname_scope
def linear(x, output_layer_size):
  weight = variable([static_shape(x)[1], output_layer_size], name="weight")
  bias = variable([output_layer_size], name="bias")
  tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)
  tf.add_to_collection(tf.GraphKeys.BIASES, bias)
  return tf.matmul(x, weight) + bias


@funcname_scope
def fully_connected(x,
                    *,
                    output_layer_size,
                    dropout_prob=None,
                    activate=None):
  may_dropout = _identity if dropout_prob == None else \
                functools.partial(dropout, dropout_prob=dropout_prob)
  may_activate = _identity if activate == None else activate
  return may_dropout(may_activate((linear(x, output_layer_size))))


def _identity(x):
  return x

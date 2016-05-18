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
                    dropout_prob,
                    output_layer_size,
                    activate=None):
  return dropout(
      ((lambda x: x) if activate is None else activate)
          (linear(x, output_layer_size)),
      dropout_prob)

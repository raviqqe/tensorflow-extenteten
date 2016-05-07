import tensorflow as tf

from .util import static_shape
from .variable import variable



def linear(x, output_layer_size):
  weight = variable([static_shape(x)[1], output_layer_size])
  bias = variable([output_layer_size])
  tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)
  tf.add_to_collection(tf.GraphKeys.BIASES, bias)
  return tf.matmul(x, weight) + bias

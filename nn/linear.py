import tensorflow as tf

from . import var_init



def linear(x, output_layer_size, regularizer_scale=1e-8):
  weight = _weight([x.get_shape().as_list()[1], output_layer_size])
  bias = _bias(output_layer_size)
  tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)
  tf.add_to_collection(tf.GraphKeys.BIASES, bias)
  return tf.matmul(x, weight) + bias


def _weight(shape):
  return tf.Variable(var_init.normal(shape))


def _bias(output_layer_size):
  return tf.Variable(var_init.normal([output_layer_size]))

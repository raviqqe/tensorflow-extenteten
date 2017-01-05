import tensorflow as tf

from .util import static_shape, func_scope
from .variable import variable


@func_scope()
def linear(x, output_layer_size):
    weight = variable([static_shape(x)[1], output_layer_size], name="weight")
    bias = variable([output_layer_size], name="bias")

    tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)
    tf.add_to_collection(tf.GraphKeys.BIASES, bias)

    return tf.matmul(x, weight) + bias


@func_scope()
def fully_connected(x,
                    output_layer_size,
                    *,
                    dropout_keep_prob=None,
                    activate=tf.nn.elu):
    h = linear(x, output_layer_size)

    if activate is not None:
        h = activate(h)

    return (h
            if dropout_keep_prob is None else
            tf.nn.dropout(h, dropout_keep_prob))

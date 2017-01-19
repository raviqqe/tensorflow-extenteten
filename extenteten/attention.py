import tensorflow as tf

from . import collections, summary
from .util import static_shape, func_scope
from .layer import linear
from .variable import variable
from .random import sample_crop
from .softmax import softmax


__all__ = ['attention_please']


@func_scope()
def attention_please(xs, context_vector_size, sequence_length=None, name=None):
    attention = _calculate_attention(xs, context_vector_size, sequence_length)
    summary.tensor(attention)
    summary.image(tf.expand_dims(
        sample_crop(attention, static_shape(attention)[1]),
        0))
    collections.add_attention(attention)
    return _give_attention(xs, attention)


@func_scope()
def _calculate_attention(xs: ("batch", "sequence", "embedding"),
                         context_vector_size,
                         sequence_length=None):
    context_vector = variable([context_vector_size, 1], name="context_vector")
    summary.tensor(context_vector)

    attention_logits = tf.reshape(
        tf.matmul(tf.tanh(linear(tf.reshape(xs, [-1, static_shape(xs)[2]]),
                                 context_vector_size)),
                  context_vector),
        [-1, static_shape(xs)[1]])

    return softmax(attention_logits, sequence_length)


@func_scope()
def _give_attention(xs, attention):
    return tf.squeeze(tf.batch_matmul(tf.transpose(xs, [0, 2, 1]),
                                      tf.expand_dims(attention, 2)),
                      [2])

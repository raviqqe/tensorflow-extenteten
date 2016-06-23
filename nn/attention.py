import tensorflow as tf

from .util import static_shape, funcname_scope
from .layer import linear
from .variable import variable
from .summary import summarize
from .dynamic_length import dynamic_softmax



@funcname_scope
def attention_please(xs, context_vector_size, sequence_length=None):
  return _give_attention(
      xs,
      _calculate_attention(xs, context_vector_size, sequence_length))


@funcname_scope
def _calculate_attention(xs : ("batch", "sequence", "embedding"),
                         context_vector_size,
                         sequence_length=None):
  context_vector = variable([context_vector_size, 1], name="context_vector")
  summarize(context_vector)

  attention_logits = tf.reshape(
      tf.matmul(tf.tanh(linear(tf.reshape(xs, [-1, static_shape(xs)[2]]),
                        context_vector_size)),
                context_vector),
      [-1, static_shape(xs)[1]])

  return tf.nn.softmax(attention_logits) if sequence_length is None else \
         dynamic_softmax(attention_logits, sequence_length)


@funcname_scope
def _give_attention(xs, attention):
  return tf.squeeze(tf.batch_matmul(tf.transpose(xs, [0, 2, 1]),
                                    tf.expand_dims(attention, 2)), [2])

import tensorflow as tf

from .util import static_shape, funcname_scope
from .layer import linear
from .variable import variable
from .summary import summarize



@funcname_scope
def attention_please(xs, context_vector_size):
  return _give_attention(xs, _calculate_attention(xs, context_vector_size))


@funcname_scope
def _calculate_attention(xs : ("batch", "sequence", "embedding"),
                         context_vector_size):
  sequence_length = static_shape(xs)[1]
  embedding_size = static_shape(xs)[2]

  context_vector = variable([context_vector_size, 1], name="context_vector")
  summarize(context_vector)

  attention_logits = tf.reshape(
      tf.matmul(tf.tanh(linear(tf.reshape(xs, [-1, embedding_size]),
                        context_vector_size)),
                context_vector),
      [-1, sequence_length])

  return tf.nn.softmax(attention_logits)


@funcname_scope
def _give_attention(xs, attention):
  return tf.squeeze(tf.batch_matmul(tf.transpose(xs, [0, 2, 1]),
                                    tf.expand_dims(attention, 2)), [2])

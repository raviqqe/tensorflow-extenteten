import tensorflow as tf

from ..util import static_shape, static_rank
from ..linear import linear
from ..variable import variable



def embeddings_to_embedding(child_embeddings,
                            output_embedding_size,
                            context_vector_size):
  assert static_rank(child_embeddings) == 3

  with tf.variable_scope("embeddings_to_embedding"):
    embedding_size = static_shape(child_embeddings)[2]
    rnn_cell = tf.nn.rnn_cell.GRUCell(output_embedding_size, embedding_size)

    cell_outputs = []
    cell_state = rnn_cell.zero_state(tf.shape(child_embeddings)[0], tf.float32)
    for iteration, child_embedding \
        in enumerate(_split_child_embeddings(child_embeddings)):
      if iteration != 0: tf.get_variable_scope().reuse_variables()
      cell_output, cell_state = rnn_cell(child_embedding, cell_state)
      cell_outputs.append(cell_output)

    return _attention_please(tf.pack(cell_outputs), context_vector_size)


def _attention_please(xs, context_vector_size):
  sequence_length = static_shape(xs)[0]
  batch_size = tf.shape(xs)[1]
  embedding_size = static_shape(xs)[2]

  context_vector = variable([context_vector_size, 1])

  attention = tf.nn.softmax(tf.transpose(tf.reshape(
      tf.matmul(
        tf.tanh(linear(
          tf.reshape(
            xs,
            [-1, embedding_size]), # -1 denotes sequence_length * batch_size
          context_vector_size)),
        context_vector),
      [sequence_length, -1]))) # -1 denotes batch_size

  return tf.transpose(tf.pack([_inner_product(x, attention)
                               for x in tf.unpack(tf.transpose(xs))]))


def _inner_product(x, y):
  return tf.squeeze(tf.batch_matmul(tf.expand_dims(x, 1),
                                    tf.expand_dims(y, 2)),
                    [1, 2])


def _split_child_embeddings(child_embeddings):
  return tf.unpack(tf.transpose(
      child_embeddings,
      [1, 0] + _dimension_indices(child_embeddings)[2:]))


def _dimension_indices(tensor):
  return list(range(static_rank(tensor)))

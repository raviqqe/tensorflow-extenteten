import tensorflow as tf

from .util import static_shape, static_rank



def embeddings_to_embedding(child_embeddings, output_embedding_size):
  assert static_rank(child_embeddings) == 3

  embedding_size = static_shape(child_embeddings)[2]
  rnn_cell = tf.nn.rnn_cell.GRUCell(output_embedding_size, embedding_size)

  state = rnn_cell.zero_state(tf.shape(child_embeddings)[0], tf.float32)
  for iteration, child_embedding \
      in enumerate(_split_child_embeddings(child_embeddings)):
    if iteration != 0: tf.get_variable_scope().reuse_variables()
    parent_embedding, state = rnn_cell(child_embedding, state)
  return parent_embedding


def _split_child_embeddings(child_embeddings):
  return tf.unpack(tf.transpose(
      child_embeddings,
      [1, 0] + _dimension_indices(child_embeddings)[2:]))


def _dimension_indices(tensor):
  return list(range(static_rank(tensor)))

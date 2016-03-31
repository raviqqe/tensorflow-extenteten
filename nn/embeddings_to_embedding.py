import tensorflow as tf

from .util import static_shape, static_rank



def embeddings_to_embedding(child_embeddings):
  assert static_rank(child_embeddings) == 3

  child_shape = static_shape(child_embeddings)
  batch_size = tf.shape(child_embeddings)[0]
  embedding_size = child_shape[2]
  rnn_cell = tf.nn.rnn_cell.GRUCell(embedding_size, embedding_size)

  state = rnn_cell.zero_state(batch_size, tf.float32)
  for iteration, child_embedding \
      in enumerate(_split_child_embeddings(child_embeddings)):
    if iteration != 0: tf.get_variable_scope().reuse_variables()
    parent_embedding, state = rnn_cell(child_embedding, state)
  return parent_embedding


def _split_child_embeddings(child_embeddings):
  num_of_childs = static_shape(child_embeddings)[1]
  permutation = [1, 0] + _dimension_indices(static_rank(child_embeddings))[2:]
  return tf.unpack(tf.transpose(child_embeddings, permutation))


def _dimension_indices(rank):
  return list(range(rank))

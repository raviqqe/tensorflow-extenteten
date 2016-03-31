import tensorflow as tf

from .util import static_rank



def embeddings_to_embedding(child_embeddings):
  assert static_rank(child_embeddings) == 3

  child_shape = child_embeddings.get_shape().as_list()
  batch_size = child_shape[0]
  embedding_size = child_shape[2]
  rnn_cell = tf.nn.rnn_cell.GRUCell(embedding_size, embedding_size)

  state = rnn_cell.zero_state(batch_size, tf.float32)
  for iteration, child_embedding \
      in enumerate(_split_child_embeddings(child_embeddings)):
    if iteration != 0: tf.get_variable_scope().reuse_variables()
    parent_embedding, state = rnn_cell(child_embedding, state)
  return parent_embedding


def _split_child_embeddings(child_embeddings):
  child_shape = child_embeddings.get_shape()
  num_of_childs = child_shape[1]
  permutation = [1, 0] + _dimension_indices(child_shape)[2:]
  splitted_child_embeddings = tf.split(
      0,
      num_of_childs,
      tf.transpose(child_embeddings, permutation))
  return map(_reshape_child_embedding, splitted_child_embeddings)


def _dimension_indices(shape):
  return list(range(shape.ndims))


def _reshape_child_embedding(child_embedding):
  return tf.reshape(child_embedding, child_embedding.get_shape()[1:])

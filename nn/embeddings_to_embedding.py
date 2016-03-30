import tensorflow as tf



def embeddings_to_embedding(child_embeddings):
  assert len(child_embeddings.get_shape()) == 3

  child_shape = child_embeddings.get_shape()
  batch_size = child_shape[0]
  embedding_size = child_shape[2]
  rnn_cell = tf.nn.rnn_cell.GRUCell(embedding_size)

  state = tf.zeros([batch_size, embedding_size])
  for child_embedding in _split_child_embeddings(child_embeddings):
    parent_embedding, state = rnn_cell(child_embedding, state)
  return parent_embedding


def _split_child_embeddings(child_embeddings):
  child_shape = child_embeddings.get_shape()
  num_of_childs = child_shape[1]
  permutation = [1, 0] + _dimension_indices(child_shape)[2:]
  return tf.split(0,
                  num_of_childs,
                  tf.transpose(child_embeddings, permutation))


def _dimension_indices(shape):
  return list(range(shape.ndims))

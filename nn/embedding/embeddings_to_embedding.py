import tensorflow as tf

from ..util import static_rank, funcname_scope, dimension_indices
from ..linear import linear
from ..variable import variable
from ..attention import attention_please



@funcname_scope
def embeddings_to_embedding(child_embeddings,
                            *,
                            output_embedding_size,
                            context_vector_size,
                            dropout_prob):
  assert static_rank(child_embeddings) == 3

  cell_outputs = _rnn(child_embeddings,
                      output_embedding_size=output_embedding_size,
                      dropout_prob=dropout_prob)

  return attention_please(cell_outputs, context_vector_size)


@funcname_scope
def _rnn(child_embeddings, *, output_embedding_size, dropout_prob):
  rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
      tf.nn.rnn_cell.GRUCell(output_embedding_size),
      1 - dropout_prob)

  cell_outputs, _ = tf.nn.rnn(
      rnn_cell,
      _split_child_embeddings(child_embeddings),
      initial_state=rnn_cell.zero_state(tf.shape(child_embeddings)[0],
                                        tf.float32))

  return _pack_cell_outputs_in_batch_seq_embed_order(cell_outputs)


@funcname_scope
def _pack_cell_outputs_in_batch_seq_embed_order(cell_outputs):
  return tf.transpose(tf.pack(cell_outputs), [1, 0, 2])


@funcname_scope
def _split_child_embeddings(child_embeddings):
  return tf.unpack(tf.transpose(
      child_embeddings,
      [1, 0] + dimension_indices(child_embeddings, 2)))

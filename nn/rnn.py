import tensorflow as tf

from .util import funcname_scope, dimension_indices



@funcname_scope
def rnn(input_embeddings, *, output_embedding_size, dropout_prob):
  rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
      tf.nn.rnn_cell.GRUCell(output_embedding_size),
      1 - dropout_prob)

  cell_outputs, _ = tf.nn.rnn(
      rnn_cell,
      _split_input_embeddings(input_embeddings),
      initial_state=rnn_cell.zero_state(tf.shape(input_embeddings)[0],
                                        tf.float32))

  return _pack_cell_outputs_in_batch_seq_embed_order(cell_outputs)


@funcname_scope
def _pack_cell_outputs_in_batch_seq_embed_order(cell_outputs):
  return tf.transpose(tf.pack(cell_outputs), [1, 0, 2])


@funcname_scope
def _split_input_embeddings(input_embeddings):
  return tf.unpack(tf.transpose(
      input_embeddings,
      [1, 0] + dimension_indices(input_embeddings, 2)))

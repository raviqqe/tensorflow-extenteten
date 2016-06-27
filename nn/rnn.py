import tensorflow as tf

from .util import funcname_scope, dimension_indices



@funcname_scope
def rnn(input_embeddings,
        *,
        output_embedding_size,
        dropout_prob,
        sequence_length=None):
  return _only_outputs(tf.nn.rnn(
      gru_cell(output_embedding_size, dropout_prob),
      _split_input_embeddings(input_embeddings),
      sequence_length=sequence_length,
      dtype=input_embeddings.dtype))


@funcname_scope
def bidirectional_rnn(input_embeddings,
                      *,
                      output_embedding_size,
                      dropout_prob,
                      sequence_length=None):
  assert output_embedding_size % 2 == 0
  cell = lambda: gru_cell(output_embedding_size // 2, dropout_prob)

  return _only_outputs(tf.nn.bidirectional_rnn(
      cell(),
      cell(),
      _split_input_embeddings(input_embeddings),
      sequence_length=sequence_length,
      dtype=input_embeddings.dtype))


@funcname_scope
def gru_cell(output_embedding_size, dropout_prob):
  return _dropout_cell(tf.nn.rnn_cell.GRUCell(output_embedding_size),
                       dropout_prob)


def _dropout_cell(cell, dropout_prob):
  return tf.nn.rnn_cell.DropoutWrapper(cell, 1 - dropout_prob)


def _only_outputs(rnn_return_values):
  return _pack_cell_outputs_in_batch_seq_embed_order(rnn_return_values[0])


@funcname_scope
def _pack_cell_outputs_in_batch_seq_embed_order(cell_outputs):
  return tf.transpose(tf.pack(cell_outputs), [1, 0, 2])


@funcname_scope
def _split_input_embeddings(input_embeddings):
  return tf.unpack(tf.transpose(
      input_embeddings,
      [1, 0] + dimension_indices(input_embeddings, 2)))

import tensorflow as tf

from ..util import funcname_scope



@funcname_scope
def gru_cell(output_embedding_size, dropout_prob):
  return _dropout_cell(tf.nn.rnn_cell.GRUCell(output_embedding_size),
                       dropout_prob)


def _dropout_cell(cell, dropout_prob):
  return tf.nn.rnn_cell.DropoutWrapper(cell, 1 - dropout_prob)

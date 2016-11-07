import tensorflow as tf

from ..flags import FLAGS
from ..util import funcname_scope



@funcname_scope
def ln_lstm(output_size, dropout_prob=FLAGS.dropout_prob):
  return tf.contrib.rnn.LayerNormBasicLSTMCell(
      output_size,
      dropout_keep_prob=1-dropout_prob)


@funcname_scope
def gru(output_size, dropout_prob=FLAGS.dropout_prob):
  return _dropout_cell(tf.nn.rnn_cell.GRUCell(output_size), dropout_prob)


def _dropout_cell(cell, dropout_prob):
  return tf.nn.rnn_cell.DropoutWrapper(cell, 1 - dropout_prob)

import tensorflow as tf

from ..util import funcname_scope, dimension_indices
from .. import flags
from ..flags import FLAGS



@funcname_scope
def rnn(inputs,
        *,
        output_embedding_size,
        dropout_prob=FLAGS.dropout_prob,
        sequence_length=None,
        cell=flags.rnn_cell()):
  return tf.nn.dynamic_rnn(
      cell(output_embedding_size, dropout_prob),
      inputs,
      sequence_length=sequence_length,
      dtype=inputs.dtype)[0]


@funcname_scope
def bidirectional_rnn(inputs,
                      *,
                      output_embedding_size,
                      dropout_prob=FLAGS.dropout_prob,
                      sequence_length=None,
                      cell=flags.rnn_cell()):
  assert output_embedding_size % 2 == 0
  create_cell = lambda: cell(output_embedding_size, dropout_prob)

  return tf.nn.bidirectional_dynamic_rnn(
      create_cell(),
      create_cell(),
      inputs,
      sequence_length=sequence_length,
      dtype=inputs.dtype)[0]

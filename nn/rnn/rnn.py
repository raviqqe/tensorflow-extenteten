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
        cell=flags.rnn_cell(),
        output_state=False):
  outputs, state = tf.nn.dynamic_rnn(
      cell(output_embedding_size, dropout_prob),
      inputs,
      sequence_length=sequence_length,
      dtype=inputs.dtype)

  return _unpack_state_tuple(state) if output_state else outputs


@funcname_scope
def bidirectional_rnn(inputs,
                      *,
                      output_embedding_size,
                      dropout_prob=FLAGS.dropout_prob,
                      sequence_length=None,
                      cell=flags.rnn_cell(),
                      output_state=False):
  assert output_embedding_size % 2 == 0
  create_cell = lambda: cell(output_embedding_size, dropout_prob)

  outputs, states = tf.nn.bidirectional_dynamic_rnn(
      create_cell(),
      create_cell(),
      inputs,
      sequence_length=sequence_length,
      dtype=inputs.dtype)

  return (tf.concat(1, [_unpack_state_tuple(state) for state in states])
          if output_state else \
          tf.concat(2, outputs))


@funcname_scope
def _unpack_state_tuple(state):
  return state.h if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple) else state

import tensorflow as tf

from . import cell
from ..util import func_scope


_DEFAULT_CELL = cell.gru


__all__ = ['rnn', 'bidirectional_rnn']


@func_scope()
def rnn(inputs,
        output_size,
        *,
        sequence_length=None,
        cell=_DEFAULT_CELL,
        output_state=False):
    outputs, state = tf.nn.dynamic_rnn(
        cell(output_size),
        inputs,
        sequence_length=sequence_length,
        dtype=inputs.dtype,
        swap_memory=True)

    return _unpack_state_tuple(state) if output_state else outputs


@func_scope()
def bidirectional_rnn(inputs,
                      output_size,
                      *,
                      sequence_length=None,
                      cell=_DEFAULT_CELL,
                      output_state=False):
    def create_cell(): return cell(output_size)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        create_cell(),
        create_cell(),
        inputs,
        sequence_length=sequence_length,
        dtype=inputs.dtype,
        swap_memory=True)

    return (tf.concat([_unpack_state_tuple(state) for state in states], 1)
            if output_state else
            tf.concat(outputs, 2))


@func_scope()
def _unpack_state_tuple(state):
    return state.h if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple) else state

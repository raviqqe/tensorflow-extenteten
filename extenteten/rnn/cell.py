import tensorflow as tf

from ..util import func_scope
from ..initializers import identity_initializer


@func_scope()
def ln_lstm(output_size):
    return tf.contrib.rnn.LayerNormBasicLSTMCell(output_size)


@func_scope(initializer=identity_initializer)
def gru(output_size):
    return tf.nn.rnn_cell.GRUCell(output_size)

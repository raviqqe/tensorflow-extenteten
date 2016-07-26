import tensorflow as tf

from ..assertion import is_sequence
from ..util import funcname_scope, static_shape, static_rank
from ..normalization import layer_normalization



class LNLSTMCell(tf.nn.rnn_cell.RNNCell):
  def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh):
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._activation = activation

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      c, h = state
      concat = _ln_linear([inputs, h], 4 * self._num_units, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(1, 4, concat)

      new_c = layer_normalization(c * tf.sigmoid(f + self._forget_bias)
                                    + tf.sigmoid(i) * self._activation(j),
                                  share_variables=True)
      new_h = self._activation(new_c) * tf.sigmoid(o)
      return new_h, tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)


@funcname_scope
def ln_lstm_cell(output_embedding_size, dropout_prob):
  return _dropout_cell(LNLSTMCell(output_embedding_size), dropout_prob)


@funcname_scope
def gru_cell(output_embedding_size, dropout_prob):
  return _dropout_cell(tf.nn.rnn_cell.GRUCell(output_embedding_size),
                       dropout_prob)


def _dropout_cell(cell, dropout_prob):
  return tf.nn.rnn_cell.DropoutWrapper(cell, 1 - dropout_prob)


@funcname_scope
def _ln_linear(inputs, output_size, add_bias=True, initial_bias=0.0):
  if not is_sequence(inputs):
    inputs = [inputs]
  assert is_sequence(inputs)

  result = 0.
  for index, input_ in enumerate(inputs):
    input_size = static_shape(input_)[1]
    assert static_rank(input_) == 2 and input_size != None

    with tf.variable_scope("weighted_{}".format(index)):
      weight = tf.get_variable("Weight{}".format(index),
                               [input_size, output_size])
      result += layer_normalization(tf.matmul(input_, weight),
                                    share_variables=True)

  if add_bias:
    result += tf.get_variable(
        "Bias",
        [output_size],
        initializer=tf.constant_initializer(initial_bias))

  return result

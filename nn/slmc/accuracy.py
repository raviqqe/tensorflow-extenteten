import tensorflow as tf

from ..util import static_shape, static_rank, funcname_scope



@funcname_scope
def accuracy(output_layer, true_label):
  assert static_rank(output_layer) == 2
  #assert static_shape(output_layer)[0] == (batch size)
  #assert static_shape(output_layer)[1] == (number of classes)
  assert static_rank(true_label) == 1
  #assert static_shape(true_label)[0] == (batch size)
  assert static_shape(output_layer)[0] == static_shape(true_label)[0]

  correct_prediction = tf.equal(tf.argmax(output_layer, 1), true_label)
  return tf.reduce_mean(tf.to_float(correct_prediction))

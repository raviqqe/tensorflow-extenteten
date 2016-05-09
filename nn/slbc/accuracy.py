import tensorflow as tf

from ..util import static_shape, static_rank, funcname_scope
from .predicted_label import predicted_label



@funcname_scope
def accuracy(output_layer, true_label):
  assert static_rank(output_layer) == static_rank(true_label) == 1
  assert static_shape(output_layer) == static_shape(true_label)
  assert true_label.dtype == tf.bool

  return tf.reduce_mean(tf.to_float(tf.equal(predicted_label(output_layer),
                                             true_label)))

import tensorflow as tf

from ..util import funcname_scope, static_shape



@funcname_scope
def loss(output_layer, true_labels):
  assert static_shape(output_layer) == static_shape(true_labels)

  return tf.nn.sigmoid_cross_entropy_with_logits(
      output_layer,
      tf.to_float(true_labels))

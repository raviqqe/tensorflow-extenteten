import tensorflow as tf

from .util import static_shape, static_rank, funcname_scope



@funcname_scope
def loss(output_layer, true_label):
  return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      output_layer,
      true_label))


@funcname_scope
def accuracy(output_layer: ("batch", "class"), true_label: ("batch",)):
  assert static_rank(output_layer) == 2
  assert static_rank(true_label) == 1
  assert static_shape(output_layer)[0] == static_shape(true_label)[0]

  return tf.reduce_mean(tf.to_float(tf.equal(predicted_label(output_layer),
                                             true_label)))


@funcname_scope
def predicted_label(output_layer):
  assert static_rank(output_layer) == 2
  return tf.argmax(output_layer, 1)

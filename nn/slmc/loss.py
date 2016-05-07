import tensorflow as tf

from ..util import funcname_scope



@funcname_scope
def loss(output_layer, true_labels):
  return tf.nn.sparse_softmax_cross_entropy_with_logits(output_layer,
                                                        true_labels)

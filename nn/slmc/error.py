import tensorflow as tf



def error(output_layer, true_labels):
  return tf.nn.sparse_softmax_cross_entropy_with_logits(output_layer,
                                                        true_labels)

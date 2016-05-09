import tensorflow as tf

from ..util import static_shape, static_rank, funcname_scope
from .accuracy import accuracy
from .loss import loss
from .predicted_labels import predicted_labels



@funcname_scope
def classify(output_layer, true_labels):
  assert static_rank(output_layer) == static_rank(true_labels) == 1
  assert static_shape(output_layer) == static_shape(true_labels)
  assert true_labels.dtype == tf.bool

  return loss(output_layer, true_labels), \
         accuracy(output_layer, true_labels), \
         predicted_labels(output_layer)

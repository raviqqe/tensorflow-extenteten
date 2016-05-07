import tensorflow as tf

from ..util import static_shape, static_rank, funcname_scope
from .. import slmc



@funcname_scope
def classify(output_layer, true_labels):
  assert static_rank(output_layer) == 2
  #assert static_shape(output_layer)[0] == (batch size)
  #assert static_shape(output_layer)[1] == (num of labels) * (num of classes)
  assert static_rank(true_labels) == 2
  #assert static_shape(true_labels)[0] == (batch size)

  num_of_labels = static_shape(true_labels)[1]

  losses, accuracies, predicted_labels = _transpose_list([
    [slmc.loss(output_layer_per_label, true_label),
     slmc.accuracy(output_layer_per_label, true_label),
     slmc.predicted_label(output_layer_per_label)]
    for output_layer_per_label, true_label
    in zip(_split_by_labels(output_layer, num_of_labels),
           _split_labels(true_labels))
  ])

  return (
    tf.reduce_sum(_concat_by_labels(losses)),
    tf.reduce_mean(tf.pack(accuracies)),
    _concat_by_labels(predicted_labels),
  )


def _split_labels(labels_tensor):
  return tf.unpack(tf.transpose(labels_tensor))


def _split_by_labels(tensor, num_of_labels):
  assert static_rank(tensor) >= 2
  #assert static_shape(tensor)[0] == (batch size)
  #assert static_shape(tensor)[1] == (values concatenated by labels)

  return tf.split(1, num_of_labels, tensor)


def _concat_by_labels(tensors):
  #assert static_shape(tensors[0])[0] == (batch size)

  return tf.transpose(tf.pack(tensors),
                      [1, 0] + list(range(static_rank(tensors[0])))[1:])


def _transpose_list(list_):
  return zip(*list_)

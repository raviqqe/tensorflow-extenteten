import tensorflow as tf

from ..util import static_shape, static_rank
from .. import slmc
from . import util



def classify(output_layer, true_labels, num_of_labels):
  assert static_rank(output_layer) == 2
  #assert static_shape(output_layer)[0] == (batch size)
  #assert static_shape(output_layer)[1] == (num of labels) * (num of classes)
  assert static_rank(true_labels) == 2
  #assert static_shape(true_labels)[0] == (batch size)
  assert static_shape(true_labels)[1] == num_of_labels

  triples_per_label = [
    [slmc.loss(output_layer_per_label, true_label),
     slmc.accuracy(output_layer_per_label, true_label),
     slmc.predicted_label(output_layer_per_label)]
    for output_layer_per_label, true_label
    in zip(util.split_by_labels(output_layer, num_of_labels),
           _split_labels(true_labels, num_of_labels))
  ]

  losses, accuracies, predicted_labels = zip(*triples_per_label)

  return (
    tf.reduce_sum(util.concat_by_labels(losses)),
    tf.reduce_mean(util.concat_by_labels(accuracies)),
    util.concat_by_labels(predicted_labels),
  )


def _split_labels(labels_tensor, num_of_labels):
  def reshape_label_tensor(label_tensor):
    return tf.reshape(label_tensor, [static_shape(labels_tensor)[0]])

  return map(reshape_label_tensor,
             util.split_by_labels(labels_tensor, num_of_labels))

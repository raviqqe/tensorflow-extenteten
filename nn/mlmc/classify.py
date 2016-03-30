import tensorflow as tf

from .. import slmc
from . import util



def classify(output_layer, true_labels, num_of_labels):
  assert len(output_layer.get_shape()) == 2
  #assert output_layer.get_shape()[0] == (batch size)
  #assert output_layer.get_shape()[1] == (num of labels) * (num of classes)
  assert len(true_labels.get_shape()) == 2
  #assert true_labels.get_shape()[0] == (batch size)
  assert true_labels.get_shape()[1] == num_of_labels

  triples_per_label = [
    [slmc.error(output_layer_per_label, true_label),
     slmc.accuracy(output_layer_per_label, true_label),
     slmc.predicted_label(output_layer_per_label)]
    for output_layer_per_label, true_label
    in zip(util.split_by_labels(output_layer, num_of_labels),
           _split_labels(true_labels, num_of_labels))
  ]

  errors, accuracies, predicted_labels = zip(*triples_per_label)

  return (
    tf.reduce_sum(util.concat_by_labels(errors)),
    tf.reduce_mean(util.concat_by_labels(accuracies)),
    util.concat_by_labels(predicted_labels),
  )


def _split_labels(labels_tensor, num_of_labels):
  def reshape_label_tensor(label_tensor):
    batch_size = labels_tensor.get_shape()[0]
    return tf.reshape(label_tensor, [batch_size])

  return map(reshape_label_tensor,
             util.split_by_labels(labels_tensor, num_of_labels))

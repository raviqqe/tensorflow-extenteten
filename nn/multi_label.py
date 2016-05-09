import tensorflow as tf

from .util import static_shape, static_rank, funcname_scope



@funcname_scope
def classify_with_single_label_module(
    output_layer: ("batch", "label * class"),
    true_labels: ("batch", "label"),
    *,
    single_label_module):
  assert static_rank(output_layer) == 2
  assert static_rank(true_labels) == 2

  losses, accuracies, predicted_labels = _transpose_2d_list([
    [single_label_module.loss(output_layer_per_label, true_label),
     single_label_module.accuracy(output_layer_per_label, true_label),
     single_label_module.predicted_label(output_layer_per_label)]
    for output_layer_per_label, true_label
    in _zip_by_labels(output_layer, true_labels)
  ])

  return tf.reduce_sum(tf.pack(losses)), \
         tf.reduce_mean(tf.pack(accuracies)), \
         _concat_by_labels(predicted_labels)


def _transpose_2d_list(list_):
  return [list(tuple_) for tuple_ in zip(*list_)]


def _zip_by_labels(output_layer, true_labels):
  assert static_rank(output_layer) == 2
  assert static_rank(true_labels) == 2

  num_of_labels = static_shape(true_labels)[1]
  assert static_shape(output_layer)[1] % num_of_labels == 0
  return zip(_split_output_layer_by_labels(output_layer, num_of_labels),
             _split_labels(true_labels))


def _split_labels(tensor):
  return tf.unpack(tf.transpose(tensor))


def _split_output_layer_by_labels(tensor, num_of_labels):
  assert static_rank(tensor) >= 2
  assert static_shape(tensor)[1] % num_of_labels == 0
  return tf.split(1, num_of_labels, tensor)


def _concat_by_labels(tensors: [("batch", ...)]):
  packed_tensor = tf.pack(tensors) # (label, batch, ...)
  return tf.transpose(packed_tensor,
                      [1, 0] + _dimensions(packed_tensor, start=2))


def _dimensions(tensor, start=0):
  return list(range(static_rank(tensor)))[start:]

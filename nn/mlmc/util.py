import tensorflow as tf

from ..util import static_shape, static_rank



def split_by_labels(tensor, num_of_labels):
  assert static_rank(tensor) >= 2
  #assert static_shape(tensor)[0] == (batch size)
  #assert static_shape(tensor)[1] == (values concatenated by labels)

  return tf.split(1, num_of_labels, tensor)


def concat_by_labels(tensors):
  if all(static_rank(tensor) == 0 for tensor in tensors):
    return tf.concat(0, tensors)

  #assert static_shape(tensors[0])[0] == (batch size)

  return tf.transpose(tf.pack(tensors),
                      [1, 0] + list(range(static_rank(tensors[0])))[1:])

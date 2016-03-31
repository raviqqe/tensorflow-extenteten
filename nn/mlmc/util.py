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

  def reshape_tensor(tensor):
    tensor_shape = static_shape(tensor)
    return tf.reshape(tensor, tensor_shape[0:1] + [1] + tensor_shape[1:])

  return tf.concat(1, [reshape_tensor(tensor) for tensor in tensors])

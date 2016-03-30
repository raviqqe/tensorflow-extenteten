import tensorflow as tf



def split_by_labels(tensor, num_of_labels):
  assert len(x.get_shape()) >= 2
  #assert x.get_shape()[0] == (batch size)
  #assert x.get_shape()[1] == (values concatenated by labels)

  return tf.split(1, num_of_labels, tensor)


def concat_by_labels(tensors):
  assert len(x.get_shape()) >= 1
  #assert x.get_shape()[0] == (batch size)

  def reshape_tensor(tensor):
    return tf.reshape(tensor,
                      [tensors.get_shape()[0], 1]
                      + list(tensors.get_shape()[1:]))

  return tf.concat(1, map(reshape_tensor, tensors))

import tensorflow as tf



def split_by_labels(tensor, num_of_labels):
  assert len(tensor.get_shape()) >= 2
  #assert tensor.get_shape()[0] == (batch size)
  #assert tensor.get_shape()[1] == (values concatenated by labels)

  return tf.split(1, num_of_labels, tensor)


def concat_by_labels(tensors):
  assert len(tensors.get_shape()) >= 1
  #assert tensors.get_shape()[0] == (batch size)

  tensors_shape = tensors.get_shape().as_list()

  def reshape_tensor(tensor):
    return tf.reshape(tensor, tensors_shape[0:1] + [1] + tensors_shape[1:])

  return tf.concat(1, map(reshape_tensor, tensors))

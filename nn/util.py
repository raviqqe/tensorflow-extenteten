import tensorflow as tf



def static_shape(tensor):
  return tensor.get_shape().as_list()


def static_rank(tensor):
  return len(static_shape(tensor))

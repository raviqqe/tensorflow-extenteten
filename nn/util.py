import functools
import tensorflow as tf



def static_shape(tensor):
  return tensor.get_shape().as_list()


def static_rank(tensor):
  return len(static_shape(tensor))


def funcname_scope(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    with tf.variable_scope(func.__name__):
      return func(*args, **kwargs)
  return wrapper


def dimension_indices(tensor, start=0):
  return list(range(static_rank(tensor)))[start:]

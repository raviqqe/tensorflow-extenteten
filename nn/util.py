import functools
import numpy
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


@funcname_scope
def machine_epsilon(dtype):
  if dtype in {tf.float32, tf.float64}:
    return tf.constant(_numpy_epsilon(dtype.as_numpy_dtype))
  else:
    raise "Machine epsilon for {} is not defined.".format(dtype)


def _numpy_epsilon(dtype):
  return numpy.finfo(dtype).eps

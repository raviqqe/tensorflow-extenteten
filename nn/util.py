import functools
import numpy
import tensorflow as tf



def static_shape(tensor):
  return tf.convert_to_tensor(tensor).get_shape().as_list()


def static_rank(tensor):
  return len(static_shape(tf.convert_to_tensor(tensor)))


def func_scope(func_or_name):
  if isinstance(func_or_name, str):
    def wrapper(func):
      func.__name__ = func_or_name
      return func_scope(func)
    return wrapper

  func = func_or_name

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    with tf.variable_scope(func.__name__):
      return func(*args, **kwargs)
  return wrapper


def on_device(device_name):
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      with tf.device(device_name):
        return func(*args, **kwargs)
    return wrapper
  return decorator


def dimension_indices(tensor, start=0):
  return [*range(static_rank(tensor))][start:]


@func_scope
def dtype_min(dtype):
  return tf.constant(_numpy_min(dtype.as_numpy_dtype))


def _numpy_min(dtype):
  return numpy.finfo(dtype).min


@func_scope
def dtype_epsilon(dtype):
  return tf.constant(_numpy_epsilon(dtype.as_numpy_dtype))


def _numpy_epsilon(dtype):
  return numpy.finfo(dtype).eps


def flatten(x):
  return tf.reshape(x, [-1])

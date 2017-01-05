import functools
import tensorflow as tf

from .util import func_scope



@func_scope()
def unpack_to_array(tensor):
  return tf.TensorArray(tensor.dtype, tf.shape(tensor)[0]).unpack(tensor)


@func_scope()
def with_dependencies(dependencies, tensor):
  """
  This function is documented partially in tensorflow.org.
  But, it cannot be found in a library.
  """
  with tf.control_dependencies(dependencies):
    if isinstance(tensor, tf.Tensor):
      return tf.identity(tensor)
    elif isinstance(tensor, tf.Operation):
      return tf.group(tensor)
    raise ValueError("{} must be tf.Tensor or tf.Operation.".format(tensor))


@func_scope()
def sequential(*ops):
  return functools.reduce(lambda x, y: with_dependencies([x], y), ops)

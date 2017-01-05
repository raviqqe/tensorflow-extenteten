import functools
import tensorflow as tf

from .assertion import is_natural_num_sequence



def variable(shape_or_initial, name=None):
  create_variable = functools.partial(tf.Variable, name=name)

  if is_natural_num_sequence(shape_or_initial):
    shape = shape_or_initial
    return create_variable(
        (tf.contrib.layers.xavier_initializer()
         if len(shape) == 2 else
         tf.truncated_normal_initializer(stddev=0.1))(shape))

  initial = shape_or_initial
  return create_variable(initial)

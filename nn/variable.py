import tensorflow as tf

from . import var_init



def variable(shape):
  return tf.Variable(var_init.normal(shape))

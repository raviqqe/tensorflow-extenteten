import tensorflow as tf

from .util import func_scope, static_rank, dtype_epsilon



@func_scope
def scalar_to_vec(scalar):
  assert static_rank(scalar) == 1
  return tf.expand_dims(scalar, [1])


@func_scope
def vec_to_mat(vec):
  assert static_rank(vec) == 2
  return tf.expand_dims(vec, [2])


@func_scope
def softmax_inverse(x):
  return tf.log(x + dtype_epsilon(x.dtype))

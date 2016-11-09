import tensorflow as tf

from .util import funcname_scope, static_rank



@funcname_scope
def scalar_to_vec(scalar):
  assert static_rank(scalar) == 1
  return tf.expand_dims(scalar, [1])


@funcname_scope
def vec_to_mat(vec):
  assert static_rank(vec) == 2
  return tf.expand_dims(vec, [2])

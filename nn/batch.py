import tensorflow as tf

from .util import funcname_scope, static_rank, dimension_indices
from .math import vec_to_mat



@funcname_scope
def dynamic_partition(data, partitions, num_partitions, name=None):
  return tf.map_fn(
      lambda args: tf.dynamic_partition(*args, num_partitions, name=name),
      [data, partitions])


@funcname_scope
def mat_vec_mul(matrix, vector):
  assert static_rank(matrix) == 3
  assert static_rank(vector) == 2

  return tf.squeeze(tf.batch_matmul(matrix, vec_to_mat(vector)), [2])


@funcname_scope
def max(x, keep_dims=False, name=None):
  return tf.reduce_max(x, dimension_indices(x, 1),
                       keep_dims=keep_dims, name=name)


@funcname_scope
def min(x, keep_dims=False, name=None):
  return tf.reduce_min(x, dimension_indices(x, 1),
                       keep_dims=keep_dims, name=name)


@funcname_scope
def sum(x, keep_dims=False, name=None):
  return tf.reduce_sum(x, dimension_indices(x, 1),
                       keep_dims=keep_dims, name=name)

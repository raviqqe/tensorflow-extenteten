import tensorflow as tf

from .util import funcname_scope, static_rank



@funcname_scope
def dynamic_partition(data, partitions, num_partitions, name=None):
  return tf.map_fn(
      lambda args: tf.dynamic_partition(*args, num_partitions, name=name),
      [data, partitions])


@funcname_scope
def mat_vec_mul(matrix, vector):
  assert static_rank(matrix) == 3
  assert static_rank(vector) == 2

  return tf.squeeze(tf.batch_matmul(matrix, tf.expand_dims(vector, [2])), [2])

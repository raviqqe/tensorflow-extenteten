import tensorflow as tf

from .util import funcname_scope



@funcname_scope
def dynamic_partition(data, partitions, num_partitions, name=None):
  return tf.map_fn(
      lambda args: tf.dynamic_partition(*args, num_partitions, name=name),
      [data, partitions])

import tensorflow as tf

from .util import static_rank, dimension_indices, func_scope



@func_scope()
def id_tree_to_root_width(id_tree, null_id=0):
  return id_sequence_to_length(tf.reduce_sum(_not_equal(id_tree, null_id),
                                             dimension_indices(id_tree, 2)))


@func_scope()
def id_sequence_to_length(id_sequence, null_id=0):
  assert static_rank(id_sequence) == 2
  return tf.reduce_sum(_not_equal(id_sequence, null_id), 1)


@func_scope()
def _not_equal(tensor, scalar):
  return tf.to_int32(tf.not_equal(tensor, scalar))
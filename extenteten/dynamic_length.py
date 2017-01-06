import tensorflow as tf

from .util import static_rank, dimension_indices, func_scope


__all__ = ['id_tensor_to_root_width', 'id_vector_to_length']


@func_scope()
def id_tensor_to_root_width(id_tensor, null_id=0):
    return id_vector_to_length(tf.reduce_sum(_not_equal(id_tensor, null_id),
                                             dimension_indices(id_tensor, 2)))


@func_scope()
def id_vector_to_length(id_vector, null_id=0):
    assert static_rank(id_vector) == 2
    return tf.reduce_sum(_not_equal(id_vector, null_id), 1)


@func_scope()
def _not_equal(tensor, scalar):
    return tf.to_int32(tf.not_equal(tensor, scalar))

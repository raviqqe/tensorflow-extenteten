import functools
import tensorflow as tf

from .util import static_shape, static_rank, funcname_scope, \
                  dimension_indices, dtype_min, dtype_epsilon
from .control import unpack_to_array
from .mask import mask



@funcname_scope
def softmax(vector, sequence_length=None):
  assert static_rank(vector) == 2

  return tf.nn.softmax(vector) if sequence_length is None else \
         _dynamic_softmax(vector, sequence_length)


@funcname_scope
def _dynamic_softmax(vector, sequence_length):
  vector_length = static_shape(vector)[1]
  mask_ = tf.cast(mask(sequence_length, vector_length), vector.dtype)
  vector_filled_with_min = mask_ * vector \
                           + (1 - mask_) * dtype_min(vector.dtype)
  tile = functools.partial(_tile_column_vector, width=vector_length)

  unnormal_dist = tf.exp(vector_filled_with_min
                         - tile(_batchwise_max(vector_filled_with_min))) \
                  * mask_
  return _batchwise_vector_scalar_div(
      unnormal_dist,
      tf.reduce_sum(unnormal_dist, [1]) + dtype_epsilon(unnormal_dist.dtype))


@funcname_scope
def _batchwise_vector_scalar_div(vector, scalar):
  assert static_rank(vector) == 2 and static_rank(scalar) == 1
  return vector / _tile_column_vector(scalar, static_shape(vector)[1])


@funcname_scope
def _batchwise_max(x):
  return tf.reduce_max(x, dimension_indices(x)[1:])


@funcname_scope
def _tile_column_vector(vector, width):
  assert static_rank(vector) == 1
  return tf.tile(tf.transpose(tf.expand_dims(vector, 0)), [1, width])

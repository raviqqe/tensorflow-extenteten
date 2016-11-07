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
  vector_length = tf.shape(vector)[1]
  mask_ = tf.cast(mask(sequence_length, vector_length), vector.dtype)
  vector_with_min = mask_ * vector + (1 - mask_) * dtype_min(vector.dtype)

  unnormal_dist = tf.exp(vector_with_min - _batch_max(vector_with_min)) * mask_
  return _batch_vector_scalar_div(
      unnormal_dist,
      tf.reduce_sum(unnormal_dist, [1]) + dtype_epsilon(unnormal_dist.dtype))


@funcname_scope
def _batch_vector_scalar_div(vector, scalar):
  assert static_rank(vector) == 2 and static_rank(scalar) == 1
  return vector / tf.expand_dims(scalar, 1)


@funcname_scope
def _batch_max(x):
  return tf.reduce_max(x, dimension_indices(x)[1:])

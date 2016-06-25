import tensorflow as tf

from .util import static_shape, static_rank, funcname_scope, machine_epsilon
from .control import unpack_to_array
from .mask import mask



@funcname_scope
def softmax(vector, sequence_length=None):
  assert static_rank(vector) == 2

  return tf.nn.softmax(vector) if sequence_length is None else \
         _dynamic_softmax(vector, sequence_length)


@funcname_scope
def _dynamic_softmax(vector, sequence_length):
  unnormal_dist = tf.exp(vector) * tf.to_float(mask(sequence_length,
                                                    static_shape(vector)[1]))
  return _batchwise_vector_scalar_div(
      unnormal_dist,
      tf.reduce_sum(unnormal_dist, [1]) + machine_epsilon(unnormal_dist.dtype))


@funcname_scope
def _batchwise_vector_scalar_div(vector, scalar):
  assert static_rank(vector) == 2 and static_rank(scalar) == 1
  return vector / tf.tile(tf.transpose(tf.expand_dims(scalar, 0)),
                          [1, static_shape(vector)[1]])

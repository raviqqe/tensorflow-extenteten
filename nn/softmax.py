import tensorflow as tf

from .util import static_shape, static_rank, funcname_scope
from .control import unpack_to_array



@funcname_scope
def softmax(vector, sequence_length=None):
  return tf.nn.softmax(vector) if sequence_length is None else \
         _dynamic_softmax(vector, sequence_length)


@funcname_scope
def _dynamic_softmax(vector, sequence_length):
  assert static_rank(vector) == 2

  vector_array = unpack_to_array(vector)
  length_array = unpack_to_array(sequence_length)

  def body(batch_index, dist_array):
    return batch_index + 1, dist_array.write(
        batch_index,
        _right_pad(
            tf.nn.softmax(tf.expand_dims(
                tf.slice(vector_array.read(batch_index),
                         [0],
                         [length_array.read(batch_index)]),
                0)),
            static_shape(vector)[1]))

  return tf.while_loop(
    lambda batch_index, _: batch_index < tf.shape(vector)[0],
    body,
    [tf.constant(0, tf.int32),
     tf.TensorArray(vector.dtype, tf.shape(vector)[0])],
  )[1].concat()


@funcname_scope
def _right_pad(tensor, length):
  assert static_rank(tensor) == 2
  result = tf.pad(tensor,
                  [[0, 0], [0, length - tf.shape(tensor)[1]]],
                  "CONSTANT")
  result.set_shape([1, length])
  return result

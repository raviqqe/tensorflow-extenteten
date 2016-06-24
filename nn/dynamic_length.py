import tensorflow as tf

from .util import static_shape, static_rank, dimension_indices
from .control import unpack_to_array



def id_tree_to_first_width(id_tree, null_id=0):
  return id_sequence_to_length(tf.reduce_sum(_not_equal(id_tree, null_id),
                                             dimension_indices(ids, 2)))


def id_sequence_to_length(id_sequence, null_id=0):
  assert static_rank(id_sequence) == 2
  return tf.reduce_sum(_not_equal(id_sequence, null_id), 1)


def _not_equal(tensor, scalar):
  return tf.to_int32(tf.not_equal(tensor, scalar))


def dynamic_softmax(vector, sequence_length):
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


def _right_pad(tensor, length):
  assert static_rank(tensor) == 2
  result = tf.pad(tensor,
                  [[0, 0], [0, length - tf.shape(tensor)[1]]],
                  "CONSTANT")
  result.set_shape([1, length])
  return result

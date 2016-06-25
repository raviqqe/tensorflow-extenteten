import tensorflow as tf

from .util import static_rank, funcname_scope



@funcname_scope
def mask(length, max_length):
  """ That's a spicy meat ball! """
  assert static_rank(length) == 1

  return tf.to_int32(tf.less(
      _range(max_length, batch_size=tf.shape(length)[0]),
      tf.tile(tf.transpose(tf.expand_dims(length, 0)), [1, max_length])))


@funcname_scope
def _range(limit, *, batch_size):
  return tf.tile(tf.expand_dims(tf.range(limit), 0), [batch_size, 1])

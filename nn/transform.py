import tensorflow as tf

from .util import func_scope, static_rank, static_shape



@func_scope()
def dynamic_pack(*tensors):
  if tensors[0].dtype == tf.string:
    return tf.pack(tensors)

  shape = _max_shape(tensors)
  return tf.pack([_pad_to_shape(tensor, shape) for tensor in tensors])


@func_scope()
def _pad_to_shape(x, shape):
  shape = tf.to_int32(shape) # for scalars
  assert static_rank(shape) == 1
  assert static_rank(x) == static_shape(shape)[0]

  paddings = tf.concat(
      1,
      [tf.expand_dims(paddings, 1) for paddings
       in [tf.zeros_like(shape),  shape - tf.shape(x)]])

  return tf.pad(x, paddings)


@func_scope()
def _max_shape(tensors):
  return [_max_dim(i, tensors) for i in range(_rank_of_tensors(tensors))]


@func_scope()
def _rank_of_tensors(tensors):
  rank = static_rank(tensors[0])
  assert all(rank == static_rank(tensor) for tensor in tensors)
  return rank


@func_scope()
def _max_dim(i, tensors):
  return tf.reduce_max([tf.shape(tensor)[i] for tensor in tensors])
